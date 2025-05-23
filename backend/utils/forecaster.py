from PyQt6.QtCore import QThread, pyqtSignal
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model 
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import logging
from datetime import datetime
import pytz
from utils.mt5_trading import MT5Trader
from utils.database import Database
import config
import talib
import pandas_ta as ta
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# In forecaster.py
class ForecastingThread(QThread):
    signal_generated = pyqtSignal(dict)
    metrics_updated = pyqtSignal(dict)

    def __init__(self, symbol="XAUUSD", timeframe="M15"):
        super().__init__()
        if symbol not in config.config["SYMBOL_OPTIONS"]:
            raise ValueError(f"Invalid symbol: {symbol}")
        if timeframe not in config.config["TIMEFRAME_OPTIONS"]:
            raise ValueError(f"Invalid timeframe: {timeframe}")
        
        self.symbol = symbol
        self.timeframe = timeframe
        self.timezone = pytz.timezone('Africa/Nairobi')
        self.model = None
        self.time_steps = 60
        self.mt5_trader = MT5Trader()
        self.prediction_history = []
        self.actual_prices = []
        self.metrics_window = 20
        self.volatility_threshold = 0.02
        self.min_confidence_long = 80.0
        self.min_confidence_short = 70.0
        self.db = Database()
        self.latest_signal = None  # Initialize latest_signal
        self.initialize_model()
        logger.info(f"ForecastingThread initialized for {symbol} on {timeframe}")
        
    def initialize_model(self):
        try:
            self.model = load_model(config.config["MODEL_PATH"])
            self.model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
            logger.info("-" * 75)
            logger.info("LSTM model loaded and compiled successfully")
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            raise

    def run(self):
        while True:
            try:
                data = self.mt5_trader.fetch_data(self.symbol, self.timeframe)
                if data is None or data.empty:
                    logger.warning(f"No data received for {self.symbol} on {self.timeframe}")
                    self.msleep(30000)
                    continue

                if not data['close'].empty:
                    latest_price = float(data['close'].iloc[-1])
                    self.actual_prices.append(latest_price)
                    if len(self.actual_prices) > self.metrics_window * 2:
                        self.actual_prices = self.actual_prices[-self.metrics_window * 2:]

                metrics = self.calculate_metrics(data)
                if metrics:
                    self.metrics_updated.emit(metrics)

                signals = self.generate_signals(data, self.symbol)
                for signal in signals:
                    logger.info(f"Generated signal: {signal}")
                    self.db.save_signal(signal)
                    self.signal_generated.emit(signal)
                
                self.msleep(30000)
            except Exception as e:
                logger.error(f"Error in forecasting loop: {str(e)}")
                self.msleep(30000)

    def calculate_all_indicators(self, data: pd.DataFrame) -> dict:
        try:
            # Require minimum 200 bars for long-term indicators, 50 for short-term
            min_bars = 50
            if len(data) < min_bars:
                logger.warning(f"Insufficient data rows ({len(data)}) for indicators. Need at least {min_bars}.")
                return {}

            data = data.dropna(subset=['close', 'high', 'low']).copy()
            if data.empty:
                logger.error("Data is empty after dropping NaNs")
                return {}

            close_prices = data['close'].astype(np.float64).values
            high_prices = data['high'].astype(np.float64).values
            low_prices = data['low'].astype(np.float64).values

            if np.any(np.isnan(close_prices)) or np.any(np.isnan(high_prices)) or np.any(np.isnan(low_prices)):
                logger.error("NaN values found in price data after conversion")
                return {}

            # Calculate indicators that can work with available data
            indicators = {}
        
            # Short-term indicators (work with less data)
            if len(close_prices) >= 7:
                indicators['ema7'] = float(talib.EMA(close_prices, timeperiod=7)[-1])
            if len(close_prices) >= 13:
                indicators['ema13'] = float(talib.EMA(close_prices, timeperiod=13)[-1])
        
            # Medium-term indicators
            if len(close_prices) >= 50:
                indicators['ema50'] = float(talib.EMA(close_prices, timeperiod=50)[-1])
        
            # Long-term indicators (only calculate if we have enough data)
            if len(close_prices) >= 200:
                indicators['ema200'] = float(talib.EMA(close_prices, timeperiod=200)[-1])
        
            # Other indicators
            if len(close_prices) >= 14:
                macd, macd_signal, _ = talib.MACD(close_prices, fastperiod=12, slowperiod=26, signalperiod=9)
                indicators['macd'] = float(macd[-1])
                indicators['macd_signal'] = float(macd_signal[-1])
                indicators['rsi'] = float(talib.RSI(close_prices, timeperiod=14)[-1])
                indicators['atr'] = float(talib.ATR(high_prices, low_prices, close_prices, timeperiod=14)[-1])
                indicators['atr_pct'] = float(indicators['atr'] / close_prices[-1])
                indicators['adx'] = float(talib.ADX(high_prices, low_prices, close_prices, timeperiod=14)[-1])
        
            # Pivot points
            indicators['pivot'] = float((high_prices[-1] + low_prices[-1] + close_prices[-1]) / 3)
            indicators['resistance1'] = 2 * indicators['pivot'] - low_prices[-1]
            indicators['support1'] = 2 * indicators['pivot'] - high_prices[-1]
        
            # Bollinger Bands (require at least 20 periods)
            if len(close_prices) >= 20:
                upper, middle, lower = talib.BBANDS(close_prices, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
                indicators.update({
                    'bb_upper': float(upper[-1]),
                    'bb_middle': float(middle[-1]),
                    'bb_lower': float(lower[-1])
                })
        
            # Detect market regime with available indicators
            indicators['regime'] = self.detect_market_regime(indicators)
        
            # Fill any missing values with defaults
            for key in ['ema7', 'ema13', 'ema50', 'ema200', 'macd', 'macd_signal', 
                    'rsi', 'atr', 'atr_pct', 'adx', 'bb_upper', 'bb_middle', 'bb_lower']:
                if key not in indicators:
                    indicators[key] = 0.0
        
            logger.debug(f"Calculated indicators: { {k: round(v, 4) if isinstance(v, float) else v for k, v in indicators.items()} }")
            return indicators

        except Exception as e:
            logger.error(f"Error calculating indicators: {str(e)}", exc_info=True)
            return {}
        
    def detect_market_regime(self, indicators: dict) -> str:
        """Enhanced market regime detection incorporating multiple factors"""
        try:
            adx = indicators.get('adx', 0.0)
            atr_pct = indicators.get('atr_pct', 0.0) * 100
            ema_diff = indicators.get('ema50', 0) - indicators.get('ema200', 0)
            rsi = indicators.get('rsi', 50)
            
            # Calculate trend strength score (0-1)
            trend_score = min(1.0, adx / 25.0) if adx > 0 else 0.0
            
            # Calculate volatility score (0-1)
            volatility_score = min(1.0, atr_pct / 1.5)
            
            # Determine market regime based on scores
            if trend_score > 0.7 and abs(ema_diff) > (indicators.get('atr', 0) * 3):
                return "Strong Trend"
            elif trend_score > 0.5:
                return "Moderate Trend"
            elif volatility_score > 0.8:
                return "High Volatility"
            elif rsi > 70 or rsi < 30:
                return "Extreme Conditions"
            elif volatility_score < 0.3 and trend_score < 0.3:
                return "Range-bound"
            else:
                return "Neutral"
                
        except Exception as e:
            logger.error(f"Error detecting market regime: {str(e)}")
            return "Neutral"

    def generate_signals(self, data: pd.DataFrame, symbol: str) -> list:
        try:
            data = data.dropna(subset=['close']).copy()
            if len(data) < self.time_steps:
                logger.warning(f"Insufficient data for prediction: {len(data)} rows available")
                return []

            indicators = self.calculate_all_indicators(data)
            if not indicators:
                logger.warning("No indicators calculated, skipping signal generation")
                return []
            
            prediction = self.predict_price(data)
            if prediction is None:
                return []
                
            price_data = self.mt5_trader.get_current_price(symbol)
            if not price_data:
                logger.error(f"Failed to fetch current price for {symbol}")
                return []
                
            current_price = price_data['ask']
            prediction_pct = (prediction - current_price) / current_price
            
            # Incorporate market regime into signal generation
            market_regime = indicators.get('regime', 'Neutral')
            is_bullish_long, is_bullish_short = self.is_bullish_signal(prediction, current_price, indicators, market_regime)
            is_bearish_long, is_bearish_short = self.is_bearish_signal(prediction, current_price, indicators, market_regime)
            
            if is_bullish_long or is_bullish_short:
                logger.info(f"Long signal detected for {symbol}: {prediction_pct:.2%} (Bullish)")
            elif is_bearish_long or is_bearish_short:
                logger.info(f"Short signal detected for {symbol}: {prediction_pct:.2%} (Bearish)")      
            else:
                logger.info(f"No valid signals detected for {symbol}: {prediction_pct:.2%},  Market Regime: {market_regime}")
                return []
            # Generate signals based on conditions
            signals = []
            for signal_type, is_valid, is_bullish in [
                ("Long-Term", is_bullish_long, True),
                ("Short-Term", is_bullish_short, True),
                ("Long-Term", is_bearish_long, False),
                ("Short-Term", is_bearish_short, False)
            ]:
                if not is_valid:
                    continue

                current_price = price_data['ask'] if is_bullish else price_data['bid']
                
                stop_loss, take_profit, risk_reward = self.calculate_sl_tp(
                    current_price=current_price,
                    is_bullish=is_bullish,
                    indicators=indicators,
                    signal_type=signal_type,
                    market_regime=market_regime
                )

                confidence = self.calculate_confidence(
                    prediction=prediction,
                    current_price=current_price,
                    indicators=indicators,
                    signal_type=signal_type,
                    market_regime=market_regime
                )
                
                min_confidence = self.min_confidence_long if signal_type == "Long-Term" else self.min_confidence_short
                if confidence < min_confidence:
                    logger.warning(f"Low confidence {signal_type} signal for {symbol} ({confidence:.1f}% < {min_confidence}%), skipping...")
                    continue

                signal = {
                    'symbol': symbol,
                    'timestamp': datetime.now(self.timezone),
                    'current_price': float(current_price),
                    'trade_signal': 'BUY' if is_bullish else 'SELL',
                    'entry_level': float(current_price),
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'status': 'Open',
                    'comments': f'Strategy {signal_type} on {self.timeframe} (Regime: {market_regime})',
                    'confidence': confidence,
                    'risk_reward_ratio': risk_reward,
                    'atr': float(indicators.get('atr', 0)),
                    'timeframe': self.timeframe,
                    'indicators': indicators,
                    'signal_type': signal_type,
                    'market_regime': market_regime
                }
                
                logger.info(f"$$$ Generated {signal['trade_signal']} ({signal_type}) signal for {symbol}: {signal}")
                signals.append(signal)
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating signals: {str(e)}")
            return []

    def is_bullish_signal(self, prediction: float, current_price: float, indicators: dict, market_regime: str) -> tuple:
        """Enhanced bullish signal detection with market regime consideration"""
        # Adjust conditions based on market regime
        regime_adjustments = {
            "Strong Trend": {"prediction_threshold": 1.002, "adx_threshold": 20},
            "Moderate Trend": {"prediction_threshold": 1.003, "adx_threshold": 15},
            "High Volatility": {"prediction_threshold": 1.005, "adx_threshold": 10},
            "Range-bound": {"prediction_threshold": 1.001, "adx_threshold": 10},
            "Extreme Conditions": {"prediction_threshold": 1.004, "adx_threshold": 10},
            "Neutral": {"prediction_threshold": 1.002, "adx_threshold": 15}
        }
        
        params = regime_adjustments.get(market_regime, regime_adjustments["Neutral"])
        
        conditions = {
            "long_term": {
                "prediction_above": prediction >= current_price * params["prediction_threshold"],
                "ema_alignment": indicators['ema50'] > indicators['ema200'],
                "adx_strong": indicators['adx'] > params["adx_threshold"],
                "macd_bullish": indicators['macd'] > indicators['macd_signal'],
                "price_not_overbought": current_price < indicators['bb_upper'] * 0.98
            },
            "short_term": {
                "prediction_above": prediction >= current_price * (params["prediction_threshold"] - 0.001),
                "ema_alignment": indicators['ema7'] > indicators['ema13'],
                "macd_bullish": indicators['macd'] > indicators['macd_signal'],
                "price_not_overbought": current_price < indicators['bb_upper'] * 0.98
            }
        }

        # Calculate scores with regime-adjusted weights
        weights = self._get_regime_weights(market_regime)
        
        long_term_score = sum(
            weights['long_term'].get(k, 0) for k, v in conditions['long_term'].items() if v
        )
        short_term_score = sum(
            weights['short_term'].get(k, 0) for k, v in conditions['short_term'].items() if v
        )

        is_long_term = long_term_score >= 0.8
        is_short_term = short_term_score >= 0.7

        return is_long_term, is_short_term

    def is_bearish_signal(self, prediction: float, current_price: float, indicators: dict, market_regime: str) -> tuple:
        """Enhanced bearish signal detection with market regime consideration"""
        # Similar structure to is_bullish_signal but with bearish conditions
        regime_adjustments = {
            "Strong Trend": {"prediction_threshold": 0.998, "adx_threshold": 20},
            "Moderate Trend": {"prediction_threshold": 0.997, "adx_threshold": 15},
            "High Volatility": {"prediction_threshold": 0.995, "adx_threshold": 10},
            "Range-bound": {"prediction_threshold": 0.999, "adx_threshold": 10},
            "Extreme Conditions": {"prediction_threshold": 0.996, "adx_threshold": 10},
            "Neutral": {"prediction_threshold": 0.998, "adx_threshold": 15}
        }
        
        params = regime_adjustments.get(market_regime, regime_adjustments["Neutral"])
        
        conditions = {
            "long_term": {
                "prediction_below": prediction <= current_price * params["prediction_threshold"],
                "ema_alignment": indicators['ema50'] < indicators['ema200'],
                "adx_strong": indicators['adx'] > params["adx_threshold"],
                "macd_bearish": indicators['macd'] < indicators['macd_signal'],
                "price_not_oversold": current_price > indicators['bb_lower'] * 1.02
            },
            "short_term": {
                "prediction_below": prediction <= current_price * (params["prediction_threshold"] + 0.001),
                "ema_alignment": indicators['ema7'] < indicators['ema13'],
                "macd_bearish": indicators['macd'] < indicators['macd_signal'],
                "price_not_oversold": current_price > indicators['bb_lower'] * 1.02
            }
        }

        weights = self._get_regime_weights(market_regime)
        
        long_term_score = sum(
            weights['long_term'].get(k, 0) for k, v in conditions['long_term'].items() if v
        )
        short_term_score = sum(
            weights['short_term'].get(k, 0) for k, v in conditions['short_term'].items() if v
        )

        is_long_term = long_term_score >= 0.8
        is_short_term = short_term_score >= 0.7

        return is_long_term, is_short_term

    def _get_regime_weights(self, market_regime: str) -> dict:
        """Get condition weights based on market regime"""
        base_weights = {
            'long_term': {
                'prediction_above': 0.20,
                'ema_alignment': 0.25,
                'adx_strong': 0.20,
                'macd_bullish': 0.25,
                'price_not_overbought': 0.10
            },
            'short_term': {
                'prediction_above': 0.25,
                'ema_alignment': 0.30,
                'macd_bullish': 0.35,
                'price_not_overbought': 0.10
            }
        }
        
        # Adjust weights based on market regime
        if market_regime == "Strong Trend":
            base_weights['long_term']['ema_alignment'] = 0.35
            base_weights['long_term']['adx_strong'] = 0.25
        elif market_regime == "High Volatility":
            base_weights['long_term']['price_not_overbought'] = 0.20
            base_weights['short_term']['price_not_overbought'] = 0.20
        
        return base_weights

    def predict_price(self, data: pd.DataFrame) -> float:
        try:
            close_prices = data['close'].astype(np.float64).values.reshape(-1, 1)
            if len(close_prices) < self.time_steps:
                logger.error(f"Insufficient data for prediction: {len(close_prices)} rows")
                return None

            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(close_prices)
            
            X = scaled_data[-self.time_steps:, 0].astype(np.float64)
            X = X.reshape(1, self.time_steps, 1)
            
            prediction_scaled = self.model.predict(X, verbose=0)[0][0]
            prediction = scaler.inverse_transform([[prediction_scaled]])[0][0]
            
            return float(prediction)
            
        except Exception as e:
            logger.error(f"Error in price prediction: {str(e)}")
            return None

    def calculate_confidence(self, prediction: float, current_price: float, 
                            indicators: dict, signal_type: str, market_regime: str) -> float:
        """Enhanced confidence calculation with market regime consideration"""
        try:
            # Base confidence components
            prediction_diff = abs(prediction - current_price) / current_price
            prediction_score = max(0, 1 - (prediction_diff / 0.01))
            
            # Trend alignment score
            if signal_type == "Long-Term":
                ema_condition = (indicators.get('ema50', 0) > indicators.get('ema200', 0)) if prediction > current_price else (indicators.get('ema50', 0) < indicators.get('ema200', 0))
                trend_score = 1.0 if ema_condition else 0.3
            else:
                ema_condition = (indicators.get('ema7', 0) > indicators.get('ema13', 0)) if prediction > current_price else (indicators.get('ema7', 0) < indicators.get('ema13', 0))
                trend_score = 1.0 if ema_condition else 0.5

            # Momentum score
            macd_diff = indicators.get('macd', 0) - indicators.get('macd_signal', 0)
            if (prediction > current_price and macd_diff > 0) or (prediction < current_price and macd_diff < 0):
                macd_score = min(1.0, abs(macd_diff) / (indicators.get('atr', 0.1) * 0.1))
            else:
                macd_score = 0.1
                
            # Volatility score
            volatility_score = max(0.5, 1 - (indicators.get('atr_pct', 0) / self.volatility_threshold))
            
            # Regime-based adjustments
            regime_adjustments = {
                "Strong Trend": {"trend_weight": 0.40, "volatility_weight": 0.05},
                "Moderate Trend": {"trend_weight": 0.35, "volatility_weight": 0.10},
                "High Volatility": {"trend_weight": 0.25, "volatility_weight": 0.25},
                "Range-bound": {"trend_weight": 0.20, "volatility_weight": 0.15},
                "Extreme Conditions": {"trend_weight": 0.30, "volatility_weight": 0.20},
                "Neutral": {"trend_weight": 0.30, "volatility_weight": 0.15}
            }
            
            params = regime_adjustments.get(market_regime, regime_adjustments["Neutral"])
            
            confidence = (
                0.20 * prediction_score +
                params["trend_weight"] * trend_score +
                0.30 * macd_score +
                0.10 * (indicators.get('adx', 0) / 25 if signal_type == "Long-Term" else 1 - (indicators.get('adx', 0) / 30)) +
                params["volatility_weight"] * volatility_score
            ) * 100
            
            return min(max(confidence, 0), 100)
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {str(e)}")
            return 50.0

    def calculate_sl_tp(self, current_price: float, is_bullish: bool, 
                        indicators: dict, signal_type: str, market_regime: str) -> tuple:
        """Enhanced SL/TP calculation with market regime consideration"""
        try:
            atr = indicators.get('atr', 0.0)
            price = current_price
            
            # Base SL/TP multipliers
            if signal_type == "Long-Term":
                sl_mult = 2.0
                tp_mult = 4.0
                min_rr = 2.0
            else:
                sl_mult = 1.5
                tp_mult = 2.5
                min_rr = 1.5
            
            # Adjust for market regime
            if market_regime == "High Volatility":
                sl_mult *= 1.2
                tp_mult *= 1.2
            elif market_regime == "Range-bound":
                sl_mult *= 0.8
                tp_mult *= 0.8
            
            if is_bullish:
                stop_loss = price - (sl_mult * atr)
                take_profit = price + (tp_mult * atr)
            else:
                stop_loss = price + (sl_mult * atr)
                take_profit = price - (tp_mult * atr)
            
            # Adjust for spread
            price_data = self.mt5_trader.get_current_price(self.symbol)
            if price_data:
                spread = price_data['ask'] - price_data['bid']
                if is_bullish:
                    stop_loss -= spread
                else:
                    stop_loss += spread
            
            # Ensure minimum risk/reward ratio
            if is_bullish:
                if (take_profit - price) < min_rr * (price - stop_loss):
                    take_profit = price + min_rr * (price - stop_loss)
            else:
                if (price - take_profit) < min_rr * (stop_loss - price):
                    take_profit = price - min_rr * (stop_loss - price)
            
            # Respect support/resistance levels
            if is_bullish:
                stop_loss = max(stop_loss, indicators['support1'] * 0.99)
                take_profit = min(take_profit, indicators['resistance1'] * 1.01)
            else:
                stop_loss = min(stop_loss, indicators['resistance1'] * 1.01)
                take_profit = max(take_profit, indicators['support1'] * 0.99)
            
            risk_reward = ((take_profit - price) / (price - stop_loss)) if is_bullish else ((price - take_profit) / (stop_loss - price))
            
            return float(stop_loss), float(take_profit), float(risk_reward)
            
        except Exception as e:
            logger.error(f"Error calculating SL/TP: {str(e)}")
            if is_bullish:
                return current_price * 0.995, current_price * 1.01, 2.0
            else:
                return current_price * 1.005, current_price * 0.99, 2.0

    def calculate_metrics(self, data: pd.DataFrame) -> dict:
        metrics = {
            'mae': 0.0,
            'mape': 0.0,
            'direction_accuracy': 0.0,
        }
        
        try:
            current_time = datetime.now(self.timezone)
            matched_pairs = []
            
            for pred_price, pred_time, expected_time in self.prediction_history[:]:
                time_diff = (current_time - expected_time).total_seconds() / 60.0
                if time_diff >= 0:
                    df_time_diff = abs(data['time'] - expected_time)
                    closest_idx = df_time_diff.idxmin()
                    if df_time_diff[closest_idx].total_seconds() / 60.0 <= 15:  # Within 15 minutes
                        actual_price = data['close'].iloc[closest_idx]
                        matched_pairs.append((pred_price, actual_price))
                        self.prediction_history.remove((pred_price, pred_time, expected_time))
            
            if len(matched_pairs) < 2:
                return metrics
                
            pred_prices, actual_prices = zip(*matched_pairs[-self.metrics_window:])
            pred_prices = np.array(pred_prices, dtype=np.float64)
            actual_prices = np.array(actual_prices, dtype=np.float64)
            
            errors = np.abs(pred_prices - actual_prices)
            metrics['mae'] = float(np.mean(errors))
            
            valid_mask = actual_prices != 0
            if valid_mask.any():
                metrics['mape'] = float(np.mean(errors[valid_mask] / actual_prices[valid_mask] * 100))
            
            direction_correct = np.sign(pred_prices[1:] - pred_prices[:-1]) == np.sign(actual_prices[1:] - actual_prices[:-1])
            metrics['direction_accuracy'] = float(np.mean(direction_correct) * 100) if len(direction_correct) > 0 else 0.0
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            return metrics
        
    def analyze_price_action(self, data: pd.DataFrame, regime: str) -> dict:
        """
        Enhanced price action analysis incorporating market regime and multiple timeframes.
        Returns a comprehensive dictionary of price behavior metrics and patterns.
        """
        try:
            # Get regime-specific configuration
            regime_config = config.config["PRICE_ACTION"].get(
                regime, 
                config.config["PRICE_ACTION"]["DEFAULT"]
            )
            lookback = regime_config["lookback_candles"]
            volatility_threshold = regime_config["volatility_threshold"]
            
            # Prepare data
            recent_data = data.tail(lookback * 2).copy()  # Get extra data for context
            if len(recent_data) < lookback:
                logger.warning(f"Insufficient data for price action analysis: {len(recent_data)} candles")
                return {}

            # Calculate core metrics
            close_prices = recent_data['close'].astype(np.float64).values
            high_prices = recent_data['high'].astype(np.float64).values
            low_prices = recent_data['low'].astype(np.float64).values
            open_prices = recent_data['open'].astype(np.float64).values
            current_price = close_prices[-1]

            # Momentum and volatility analysis
            net_change = (current_price - close_prices[0]) / close_prices[0] * 100
            atr = float(talib.ATR(high_prices, low_prices, close_prices, timeperiod=14)[-1])
            atr_pct = atr / current_price * 100
            volatility_level = "High" if atr_pct > volatility_threshold else "Moderate" if atr_pct > volatility_threshold/2 else "Low"

            # Pattern detection with regime-aware thresholds
            patterns = self._detect_candle_patterns(
                open_prices[-lookback:],
                high_prices[-lookback:],
                low_prices[-lookback:],
                close_prices[-lookback:],
                regime
            )

            # Trend analysis
            trend_strength, trend_direction = self._analyze_trend_structure(
                high_prices,
                low_prices,
                close_prices,
                regime
            )

            # Support/resistance analysis
            s_r_analysis = self._analyze_support_resistance(
                recent_data,
                current_price,
                regime_config["support_resistance_threshold"]
            )

            # Market session analysis
            session_info = self._get_market_session_info()

            # Compile comprehensive price behavior report
            price_behavior = {
                # Core metrics
                "current_price": current_price,
                "net_change_pct": net_change,
                "atr_pct": atr_pct,
                "volatility_level": volatility_level,
                
                # Trend analysis
                "trend_strength": trend_strength,
                "trend_direction": trend_direction,
                "higher_highs": trend_direction == "Up",
                "lower_lows": trend_direction == "Down",
                
                # Pattern analysis
                "candle_patterns": " | ".join(patterns) if patterns else "None",
                "pattern_count": len(patterns),
                
                # Support/resistance
                "near_support": s_r_analysis["near_support"],
                "near_resistance": s_r_analysis["near_resistance"],
                "support_level": s_r_analysis["support_level"],
                "resistance_level": s_r_analysis["resistance_level"],
                
                # Contextual information
                "market_regime": regime,
                "active_session": session_info["session"],
                "session_status": session_info["status"],
                "time": datetime.now(self.timezone).strftime("%H:%M | Kenya"),
                
                # Derived signals
                "momentum": "Bullish" if net_change > 0.1 else "Bearish" if net_change < -0.1 else "Neutral",
                "price_action_alert": self._generate_price_alert(
                    patterns,
                    s_r_analysis,
                    trend_strength,
                    net_change
                )
            }

            logger.debug(f"Price action analysis for {self.symbol}: { {k: round(v, 4) if isinstance(v, float) else v for k, v in price_behavior.items()} }")
            return price_behavior

        except Exception as e:
            logger.error(f"Error analyzing price action: {str(e)}", exc_info=True)
            return self.last_price_action if hasattr(self, 'last_price_action') else {}
    
    def _detect_candle_patterns(self, opens, highs, lows, closes, regime) -> list:
        """Helper method to detect candle patterns with regime-aware sensitivity"""
        patterns = []
        body_sensitivity = 0.1 if regime in ["High Volatility", "Strong Trend"] else 0.2
        
        for i in range(-1, -len(opens)-1, -1):
            body = abs(closes[i] - opens[i])
            candle_range = highs[i] - lows[i]
            upper_shadow = highs[i] - max(opens[i], closes[i])
            lower_shadow = min(opens[i], closes[i]) - lows[i]
            
            # Bullish patterns
            if closes[i] > opens[i] and body > upper_shadow * 2 and body > lower_shadow * 2:
                if i < -1 and closes[i] > opens[i+1] and closes[i] > highs[i+1]:
                    patterns.append("Bullish Engulfing")
                elif lower_shadow > body * 2 and upper_shadow < body * 0.5:
                    patterns.append("Hammer")
            
            # Bearish patterns  
            elif closes[i] < opens[i] and body > upper_shadow * 2 and body > lower_shadow * 2:
                if i < -1 and closes[i] < opens[i+1] and closes[i] < lows[i+1]:
                    patterns.append("Bearish Engulfing")
                elif upper_shadow > body * 2 and lower_shadow < body * 0.5:
                    patterns.append("Hanging Man")
            
            # Neutral patterns
            elif body < candle_range * body_sensitivity:
                patterns.append("Doji")
        
        return patterns[-3:]  # Return most recent 3 patterns

    def _analyze_trend_structure(self, highs, lows, closes, regime) -> tuple:
        """Analyze trend strength and direction with regime context"""
        # Use different sensitivity based on market regime
        min_trend_length = 3 if regime in ["Strong Trend", "High Volatility"] else 5
        
        # Check for higher highs/lower lows
        up_trend = all(highs[i] > highs[i-1] for i in range(1, min_trend_length))
        down_trend = all(lows[i] < lows[i-1] for i in range(1, min_trend_length))
        
        # Calculate trend strength using ADX if available
        if len(closes) >= 14:
            adx = talib.ADX(highs, lows, closes, timeperiod=14)[-1]
            strength = "Strong" if adx > 25 else "Moderate" if adx > 20 else "Weak"
        else:
            strength = "Moderate" if up_trend or down_trend else "Weak"
        
        direction = "Up" if up_trend else "Down" if down_trend else "Sideways"
        return strength, direction

    def _analyze_support_resistance(self, data, current_price, threshold_pct=0.5) -> dict:
        """Identify nearby support/resistance levels"""
        indicators = self.calculate_all_indicators(data)
        support = indicators.get('support1', current_price * 0.99)
        resistance = indicators.get('resistance1', current_price * 1.01)
        
        # Calculate distance to nearest levels
        support_dist_pct = abs(current_price - support) / current_price * 100
        resistance_dist_pct = abs(current_price - resistance) / current_price * 100
        
        return {
            "near_support": support_dist_pct < threshold_pct,
            "near_resistance": resistance_dist_pct < threshold_pct,
            "support_level": support,
            "resistance_level": resistance,
            "support_distance_pct": support_dist_pct,
            "resistance_distance_pct": resistance_dist_pct
        }

    def _get_market_session_info(self) -> dict:
        """Determine current market session status"""
        current_time = datetime.now(self.timezone)
        sessions = config.config["TRADING_SESSIONS"]
        
        for session_name, times in sessions.items():
            open_time = datetime.strptime(times["open"], "%H:%M").time()
            close_time = datetime.strptime(times["close"], "%H:%M").time()
            current_time_only = current_time.time()
            
            if open_time <= close_time:
                if open_time <= current_time_only <= close_time:
                    return {"session": session_name, "status": "Active"}
            else:
                if current_time_only >= open_time or current_time_only <= close_time:
                    return {"session": session_name, "status": "Active"}
        
        return {"session": "None", "status": "Closed"}

    def _generate_price_alert(self, patterns, s_r_analysis, trend_strength, net_change) -> str:
        """Generate actionable price action alerts"""
        alerts = []
        
        # Pattern-based alerts
        if "Bullish Engulfing" in patterns and s_r_analysis["near_support"]:
            alerts.append("Bullish Reversal Signal")
        elif "Bearish Engulfing" in patterns and s_r_analysis["near_resistance"]:
            alerts.append("Bearish Reversal Signal")
        
        # Trend strength alerts
        if trend_strength == "Strong" and abs(net_change) > 1.0:
            alerts.append("Strong Trend Continuation")
        
        # Support/resistance alerts
        if s_r_analysis["near_support"] and net_change < -0.5:
            alerts.append("Potential Support Bounce")
        elif s_r_analysis["near_resistance"] and net_change > 0.5:
            alerts.append("Potential Resistance Rejection")
        
        return " | ".join(alerts) if alerts else "No Alert"
        
        
        
        