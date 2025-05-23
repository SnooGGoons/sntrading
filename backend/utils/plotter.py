import plotly.graph_objects as go
import plotly.subplots as sp
import pandas as pd
import numpy as np
import os
import logging
import traceback
import webbrowser
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import pickle
from utils.mt5_trading import MT5Trader
import config
import talib
import time
from datetime import datetime, timedelta


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Plotter:
    def __init__(self):
        self.cache_dir = os.path.join(config.BASE_DIR, "cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.model = None
        self.time_steps = config.config["TIME_STEPS"]
        self.trader = MT5Trader()
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.load_model()
        
        # Color scheme configuration
        self.colors = {
            'background': '#1e2124',
            'text': '#e0e0e0',
            'grid': '#2a2e32',
            'up_candle': '#3dc26b',
            'down_candle': '#e74c3c',
            'volume': '#7f8c8d',
            'indicators': {
                'sma20': '#9b59b6',
                'ema50': '#f39c12',
                'ema200': '#3498db',
                'vwap': '#1abc9c',
                'upper_bb': '#ff69b4',
                'middle_bb': '#ffffff',
                'lower_bb': '#ff69b4',
                'prediction': '#ff4500',
                'macd': '#3498db',
                'signal': '#f39c12',
                'hist_positive': '#2ecc71',
                'hist_negative': '#e74c3c',
                'rsi': '#3498db',
                'rsi_high': '#e74c3c',
                'rsi_low': '#e74c3c',
                'rsi_mid': '#95a5a6',
                'volume_ma': '#f39c12',
                'atr': '#e67e22',
                'adx': '#9b59b6'
            }
        }

    def load_model(self):
        """Load the LSTM model with error handling."""
        try:
            model_path = config.config.get("MODEL_PATH", "")
            if not os.path.exists(model_path):
                logger.error(f"Model file not found at {model_path}")
                return
            self.model = load_model(model_path)
            logger.info("LSTM model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            logger.debug(traceback.format_exc())

    def ensure_mt5_connection(self):
        """Ensure MT5 connection is active with retries."""
        max_retries = 3
        retry_delay = 5
        for attempt in range(max_retries):
            if self.trader.check_connection():
                return True
            logger.warning(f"MT5 connection attempt {attempt + 1} failed, retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
        logger.error("Failed to establish MT5 connection after multiple attempts")
        return False

    def get_cache_path(self, symbol, timeframe):
        """Generate cache file path for given symbol and timeframe."""
        return os.path.join(self.cache_dir, f"{symbol}_{timeframe}_cache.pkl")

    def load_cached_data(self, symbol, timeframe):
        """Load cached data if recent and valid."""
        cache_path = self.get_cache_path(symbol, timeframe)
        try:
            if os.path.exists(cache_path):
                with open(cache_path, 'rb') as f:
                    cached = pickle.load(f)
                    cache_timeout = config.config["TIMEFRAME_MINUTES"].get(timeframe, 15) * 60
                    if (datetime.now() - cached['timestamp']).total_seconds() < cache_timeout:
                        data = cached['data']
                        if self.validate_data(data):
                            logger.info(f"Loaded cached data for {symbol} on {timeframe}")
                            return data
                        else:
                            logger.warning(f"Invalid cached data for {symbol} on {timeframe}")
            return None
        except Exception as e:
            logger.error(f"Error loading cached data: {str(e)}")
            return None

    def save_cached_data(self, symbol, timeframe, data):
        """Save data to cache."""
        cache_path = self.get_cache_path(symbol, timeframe)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump({
                    'timestamp': datetime.now(),
                    'data': data,
                    'symbol': symbol,
                    'timeframe': timeframe
                }, f)
            logger.info(f"Cached data saved for {symbol} on {timeframe}")
        except Exception as e:
            logger.error(f"Error saving cached data: {str(e)}")

    def validate_data(self, data):
        """Validate that the DataFrame has required columns and sufficient rows."""
        if not isinstance(data, pd.DataFrame):
            logger.error("Data is not a pandas DataFrame")
            return False
        required_columns = ['time', 'open', 'high', 'low', 'close', 'tick_volume']
        if not all(col in data.columns for col in required_columns):
            logger.error(f"Data missing required columns: {required_columns}")
            return False
        if len(data) < self.time_steps:
            logger.error(f"Data has insufficient rows: {len(data)} < {self.time_steps}")
            return False
        if data[['open', 'high', 'low', 'close']].isnull().any().any():
            logger.error("Data contains NaN values in price columns")
            return False
        return True

    def fetch_market_data(self, symbol, timeframe, bars=200):
        """Fetch market data with connection checks and retries."""
        if not self.ensure_mt5_connection():
            logger.error("Cannot fetch data - no connection to MT5")
            return None
        try:
            data = self.trader.fetch_data(symbol, timeframe, bars)
            if data is None or data.empty:
                logger.error(f"No data available for {symbol} on {timeframe}")
                return None
            if not self.validate_data(data):
                logger.error(f"Invalid data fetched for {symbol} on {timeframe}")
                return None
            return data
        except Exception as e:
            logger.error(f"Error fetching data for {symbol} on {timeframe}: {str(e)}")
            return None

    def calculate_indicators(self, data):
        """Calculate technical indicators with error handling."""
        try:
            close = data['close'].values.astype(np.float64)
            high = data['high'].values.astype(np.float64)
            low = data['low'].values.astype(np.float64)
            volume = data['tick_volume'].values.astype(np.float64)
            
            indicators = {
                'sma20': talib.SMA(close, timeperiod=20),
                'ema50': talib.EMA(close, timeperiod=50),
                'ema200': talib.EMA(close, timeperiod=200),
                'rsi': talib.RSI(close, timeperiod=14),
                'macd': talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)[0],
                'macd_signal': talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)[1],
                'macd_hist': talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)[2],
                'volume_ma': talib.SMA(volume, timeperiod=5),
                'atr': talib.ATR(high, low, close, timeperiod=14),
                'vwap': (data['close'] * data['tick_volume']).cumsum() / data['tick_volume'].cumsum(),
                'adx': talib.ADX(high, low, close, timeperiod=14),
                'obv': talib.OBV(data['close'], data['tick_volume']),
                'stoch_k': talib.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)[0],
                'stoch_d': talib.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)[1],
                'cci': talib.CCI(high, low, close, timeperiod=20)
            }
            
            upper_bb, middle_bb, lower_bb = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
            indicators.update({
                'upper_bb': upper_bb,
                'middle_bb': middle_bb,
                'lower_bb': lower_bb
            })
            
            # Calculate Fibonacci levels if we have enough data
            if len(close) >= 20:
                max_price = max(close[-20:])
                min_price = min(close[-20:])
                diff = max_price - min_price
                indicators.update({
                    'fib_0': max_price,
                    'fib_23': max_price - diff * 0.236,
                    'fib_38': max_price - diff * 0.382,
                    'fib_50': max_price - diff * 0.5,
                    'fib_61': max_price - diff * 0.618,
                    'fib_100': min_price
                })
            
            return indicators
        except Exception as e:
            logger.error(f"Error calculating indicators: {str(e)}")
            logger.debug(traceback.format_exc())
            return None

    def predict_prices(self, data):
        """Generate price predictions using LSTM model."""
        try:
            close_prices = data['close'].astype(np.float64).values.reshape(-1, 1)
            if len(close_prices) < self.time_steps:
                logger.error(f"Insufficient data for prediction: {len(close_prices)} rows")
                return None, None, None

            scaled_data = self.scaler.fit_transform(close_prices)
            X = scaled_data[-self.time_steps:, 0].astype(np.float64)
            X = X.reshape(1, self.time_steps, 1)
            
            # Predict next 5 candles with uncertainty estimation
            predictions = []
            prediction_uncertainty = []
            future_times = []
            timeframe_minutes = config.config["TIMEFRAME_MINUTES"].get(data['timeframe'].iloc[0], 15)
            last_time = data['time'].iloc[-1]
            
            current_X = X.copy()
            for i in range(5):
                pred_scaled = self.model.predict(current_X, verbose=0)[0][0]
                pred_price = self.scaler.inverse_transform([[pred_scaled]])[0][0]
                uncertainty = abs(pred_price - close_prices[-1]) * 0.05  # 5% of price change
                
                predictions.append(pred_price)
                prediction_uncertainty.append(uncertainty)
                future_times.append(last_time + timedelta(minutes=timeframe_minutes * (i + 1)))
                
                current_X = np.roll(current_X, -1, axis=1)
                current_X[0, -1, 0] = pred_scaled
            
            return predictions, prediction_uncertainty, future_times
        except Exception as e:
            logger.error(f"Error in price prediction: {str(e)}")
            return None, None, None

    def generate_plot(self, symbol, timeframe):
        """Generate technical analysis plot with interactive indicator controls and predictions."""
        try:
            if symbol not in config.config["SYMBOL_OPTIONS"]:
                logger.error(f"Invalid symbol: {symbol}")
                return False
            if timeframe not in config.config["TIMEFRAME_OPTIONS"]:
                logger.error(f"Invalid timeframe: {timeframe}")
                return False

            # Get data (try cache first)
            data = self.load_cached_data(symbol, timeframe)
            if data is None:
                logger.info(f"Fetching fresh data for {symbol} on {timeframe}")
                data = self.fetch_market_data(symbol, timeframe)
                if data is None:
                    logger.error("Failed to fetch market data")
                    return False
                self.save_cached_data(symbol, timeframe, data)

            data = data.tail(200).copy()
            data['time'] = pd.to_datetime(data['time'])
            data['timeframe'] = timeframe

            # Calculate indicators and predictions
            indicators = self.calculate_indicators(data)
            if indicators is None:
                logger.error("Failed to calculate indicators")
                return False

            predictions, uncertainty, future_times = self.predict_prices(data)
            if predictions is None:
                logger.warning("Failed to generate predictions, proceeding without them")

            # Create figure with buttons
            fig = self.create_figure(symbol, timeframe, data, indicators, predictions, uncertainty, future_times)
            if fig is None:
                return False

            return self.save_and_display_plot(fig, symbol, timeframe)
        except Exception as e:
            logger.error(f"Error in generate_plot: {str(e)}")
            logger.debug(traceback.format_exc())
            return False

    def create_figure(self, symbol, timeframe, data, indicators, predictions, uncertainty, future_times):
        """Create the Plotly figure with subplots and collapsible indicator controls."""
        try:
            # Create subplots with rows for price, volume, RSI, MACD, Stochastic
            fig = sp.make_subplots(
                rows=5, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.02,
                row_heights=[0.5, 0.15, 0.15, 0.1, 0.1],
                specs=[[{"secondary_y": True}], [{}], [{}], [{}], [{}]]
            )

            # Add all chart elements
            self.add_price_chart(fig, data, indicators, predictions, uncertainty, future_times, row=1)
            self.add_volume_chart(fig, data, indicators, row=2)
            self.add_rsi_chart(fig, data, indicators, row=3)
            self.add_macd_chart(fig, data, indicators, row=4)
            self.add_stochastic_chart(fig, data, indicators, row=5)

            # Define indicator groups for collapsible controls
            indicator_groups = {
                'Moving Averages': ['ema50', 'ema200', 'sma20'],
                'Bollinger Bands': ['bollinger'],
                'Fibonacci': ['fibonacci']
            }

            # Create collapsible buttons
            buttons = []
            visibility_state = {}  # Track visibility state for each group

            # Initialize visibility state
            for group in indicator_groups:
                visibility_state[group] = False  # Start collapsed

            # Create buttons for each group
            for group_name, indicators_in_group in indicator_groups.items():
                # Toggle button for group
                visible_list = []
                for trace in fig.data:
                    trace_type = trace.meta.get('trace_type', '') if hasattr(trace, 'meta') else ''
                    if trace.name == 'Price' or trace_type == 'predicted':
                        visible_list.append(True)  # Always show price and predictions
                    elif any(ind in trace_type for ind in indicators_in_group):
                        visible_list.append(visibility_state[group_name])
                    else:
                        visible_list.append(False)
                
                buttons.append(dict(
                    label=f"{group_name} {'▼' if visibility_state[group_name] else '▶'}",
                    method="update",
                    args=[{
                        "visible": visible_list,
                        "updatemenus[0].buttons": [
                            dict(
                                label=f"{g} {'▼' if (g == group_name and not visibility_state[g]) or (g != group_name and visibility_state[g]) else '▶'}",
                                method="update",
                                args=[{
                                    "visible": [
                                        True if trace.name == 'Price' or trace.meta.get('trace_type', '') == 'predicted'
                                        else any(ind in trace.meta.get('trace_type', '') for ind in indicator_groups[g]) if g == group_name and not visibility_state[g]
                                        else visibility_state[g] and any(ind in trace.meta.get('trace_type', '') for ind in indicator_groups[g])
                                        for trace in fig.data
                                    ]
                                }]
                            ) for g in indicator_groups
                        ]
                    }]
                ))

            # Add layout with collapsible controls
            fig.update_layout(
                height=1200,
                width=1400,
                template="plotly_dark",
                hovermode="x unified",
                margin=dict(l=100, r=50, t=100, b=50),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                plot_bgcolor=self.colors['background'],
                paper_bgcolor=self.colors['background'],
                font=dict(color=self.colors['text']),
                title=dict(
                    text=f"{symbol} - {timeframe} Technical Analysis",
                    x=0.5,
                    font=dict(size=20)
                ),
                updatemenus=[
                    dict(
                        type="buttons",
                        direction="down",
                        pad={"r": 10, "t": 10},
                        showactive=True,
                        x=0.02,
                        xanchor="left",
                        y=1.15,
                        yanchor="top",
                        buttons=buttons
                    )
                ]
            )

            # Update axis titles and styles
            fig.update_xaxes(
                title_text="Time",
                row=5,
                col=1,
                gridcolor=self.colors['grid'],
                showspikes=True,
                spikethickness=1,
                spikecolor="grey",
                spikemode="across"
            )
            
            fig.update_yaxes(
                title_text="Price",
                row=1,
                col=1,
                gridcolor=self.colors['grid']
            )
            
            fig.update_yaxes(
                title_text="Volume",
                row=2,
                col=1,
                gridcolor=self.colors['grid']
            )
            
            fig.update_yaxes(
                title_text="RSI",
                row=3,
                col=1,
                gridcolor=self.colors['grid'],
                range=[0, 100]
            )
            
            fig.update_yaxes(
                title_text="MACD",
                row=4,
                col=1,
                gridcolor=self.colors['grid']
            )
            
            fig.update_yaxes(
                title_text="Stochastic",
                row=5,
                col=1,
                gridcolor=self.colors['grid'],
                range=[0, 100]
            )

            # Add range slider
            fig.update_xaxes(
                rangeslider=dict(
                    visible=True,
                    thickness=0.05,
                    bgcolor=self.colors['grid']
                ),
                row=5,
                col=1
            )

            return fig
        except Exception as e:
            logger.error(f"Error creating figure: {str(e)}")
            return None

    def add_price_chart(self, fig, data, indicators, predictions, uncertainty, future_times, row):
        """Add price chart with indicators and predictions."""
        # Candlestick
        fig.add_trace(go.Candlestick(
            x=data['time'],
            open=data['open'],
            high=data['high'],
            low=data['low'],
            close=data['close'],
            name='Price',
            increasing_line_color=self.colors['up_candle'],
            decreasing_line_color=self.colors['down_candle'],
            meta={'trace_type': 'price'}
        ), row=row, col=1)

        # Moving Averages
        for name, values, color, trace_type in [
            ('SMA 20', indicators['sma20'], self.colors['indicators']['sma20'], 'sma20'),
            ('EMA 50', indicators['ema50'], self.colors['indicators']['ema50'], 'ema50'),
            ('EMA 200', indicators['ema200'], self.colors['indicators']['ema200'], 'ema200')
        ]:
            fig.add_trace(go.Scatter(
                x=data['time'],
                y=values,
                name=name,
                line=dict(color=color, width=1.5),
                opacity=0.8,
                visible=False,
                meta={'trace_type': trace_type}
            ), row=row, col=1)

        # Bollinger Bands
        for name, values, color in [
            ('BB Upper', indicators['upper_bb'], self.colors['indicators']['upper_bb']),
            ('BB Middle', indicators['middle_bb'], self.colors['indicators']['middle_bb']),
            ('BB Lower', indicators['lower_bb'], self.colors['indicators']['lower_bb'])
        ]:
            fig.add_trace(go.Scatter(
                x=data['time'],
                y=values,
                name=name,
                line=dict(color=color, width=1.5, dash='dash' if name != 'BB Middle' else 'solid'),
                visible=False,
                meta={'trace_type': 'bollinger'}
            ), row=row, col=1)

        # Fibonacci Levels (if available)
        if 'fib_0' in indicators:
            for level, value in [
                ('Fib 0%', indicators['fib_0']),
                ('Fib 23.6%', indicators['fib_23']),
                ('Fib 38.2%', indicators['fib_38']),
                ('Fib 50%', indicators['fib_50']),
                ('Fib 61.8%', indicators['fib_61']),
                ('Fib 100%', indicators['fib_100'])
            ]:
                fig.add_trace(go.Scatter(
                    x=[data['time'].iloc[0], data['time'].iloc[-1]],
                    y=[value, value],
                    name=level,
                    line=dict(color='#FFA07A', width=1, dash='dot'),
                    visible=False,
                    meta={'trace_type': 'fibonacci'}
                ), row=row, col=1)

        # Predicted Prices with uncertainty bands (always visible)
        if predictions and future_times:
            fig.add_trace(go.Scatter(
                x=future_times,
                y=predictions,
                name='Predicted Price',
                line=dict(color=self.colors['indicators']['prediction'], width=2),
                mode='lines+markers',
                meta={'trace_type': 'predicted'}
            ), row=row, col=1)
            
            fig.add_trace(go.Scatter(
                x=future_times,
                y=[p + u for p, u in zip(predictions, uncertainty)],
                name='Upper Bound',
                line=dict(color=self.colors['indicators']['prediction'], width=0.5, dash='dot'),
                fill=None,
                mode='lines',
                meta={'trace_type': 'predicted'}
            ), row=row, col=1)
            
            fig.add_trace(go.Scatter(
                x=future_times,
                y=[p - u for p, u in zip(predictions, uncertainty)],
                name='Lower Bound',
                line=dict(color=self.colors['indicators']['prediction'], width=0.5, dash='dot'),
                fill='tonexty',
                mode='lines',
                meta={'trace_type': 'predicted'}
            ), row=row, col=1)

    def add_volume_chart(self, fig, data, indicators, row):
        """Add volume chart."""
        fig.add_trace(go.Bar(
            x=data['time'],
            y=data['tick_volume'],
            name='Volume',
            marker_color=self.colors['volume'],
            opacity=0.7,
            meta={'trace_type': 'volume'}
        ), row=row, col=1)

    def add_rsi_chart(self, fig, data, indicators, row):
        """Add RSI chart."""
        fig.add_trace(go.Scatter(
            x=data['time'],
            y=indicators['rsi'],
            name='RSI',
            line=dict(color=self.colors['indicators']['rsi'], width=1.5),
            meta={'trace_type': 'rsi'}
        ), row=row, col=1)

        for level, color in [
            (70, self.colors['indicators']['rsi_high']),
            (30, self.colors['indicators']['rsi_low']),
            (50, self.colors['indicators']['rsi_mid'])
        ]:
            fig.add_hline(
                y=level,
                line_dash="dash",
                line_color=color,
                opacity=0.7,
                row=row, col=1
            )

    def add_macd_chart(self, fig, data, indicators, row):
        """Add MACD chart."""
        for name, values, color in [
            ('MACD', indicators['macd'], self.colors['indicators']['macd']),
            ('Signal', indicators['macd_signal'], self.colors['indicators']['signal'])
        ]:
            fig.add_trace(go.Scatter(
                x=data['time'],
                y=values,
                name=name,
                line=dict(color=color, width=1.5),
                meta={'trace_type': 'macd'}
            ), row=row, col=1)

        fig.add_trace(go.Bar(
            x=data['time'],
            y=indicators['macd_hist'],
            name='Histogram',
            marker_color=np.where(
                indicators['macd_hist'] >= 0,
                self.colors['indicators']['hist_positive'],
                self.colors['indicators']['hist_negative']
            ),
            opacity=0.6,
            meta={'trace_type': 'macd'}
        ), row=row, col=1)

        fig.add_hline(
            y=0,
            line_color=self.colors['text'],
            row=row, col=1
        )

    def add_stochastic_chart(self, fig, data, indicators, row):
        """Add Stochastic Oscillator chart."""
        for name, values, color in [
            ('%K', indicators['stoch_k'], '#3498db'),
            ('%D', indicators['stoch_d'], '#f39c12')
        ]:
            fig.add_trace(go.Scatter(
                x=data['time'],
                y=values,
                name=name,
                line=dict(color=color, width=1.5),
                meta={'trace_type': 'stochastic'}
            ), row=row, col=1)

        for level, color in [
            (80, '#e74c3c'),
            (20, '#e74c3c'),
            (50, '#95a5a6')
        ]:
            fig.add_hline(
                y=level,
                line_dash="dash",
                line_color=color,
                opacity=0.7,
                row=row, col=1
            )

    def save_and_display_plot(self, fig, symbol, timeframe):
        """Save plot to file and display it."""
        try:
            plot_dir = config.config.get("PLOT_DIR", os.path.join(config.BASE_DIR, "plots"))
            os.makedirs(plot_dir, exist_ok=True)
            plot_path = os.path.join(plot_dir, f"{symbol}_{timeframe}_analysis.html")
            
            fig.write_html(
                plot_path,
                auto_open=False,
                config={
                    'responsive': True,
                    'scrollZoom': True,
                    'displayModeBar': True,
                    'displaylogo': False,
                    'modeBarButtonsToAdd': ['hoverclosest', 'hovercompare']
                }
            )
            logger.info(f"Plot saved to {plot_path}")
            
            webbrowser.open(f"file://{os.path.abspath(plot_path)}")
            return True
        except Exception as e:
            logger.error(f"Failed to save plot: {str(e)}")
            return False