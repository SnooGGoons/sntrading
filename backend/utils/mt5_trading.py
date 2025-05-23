import MetaTrader5 as mt5
import pandas as pd
from typing import Optional, Dict
import logging
from datetime import datetime
import pytz
import time
import config
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MT5Trader:
    """Enhanced MT5 trading client with robust connection management and error handling."""
    
    def __init__(self):
        """Initialize trader with connection parameters and caches."""
        self.connected = False
        self.last_connection_time = None
        self.timezone = pytz.timezone('Africa/Nairobi')
        self.timeout = 30000  # 30 seconds in milliseconds
        self.max_retries = 5
        self.retry_delay = 5  # seconds
        self.symbol_info_cache = {}
        self.data_cache = {}
        self.last_trade_time = None
        self.trade_timeout = 10  # seconds between trades

    # --------------------------
    # Connection Management
    # --------------------------
    
    def connect(self) -> bool:
        """Establish MT5 connection with comprehensive error handling."""
        if self.connected:
            return True
            
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Connecting to MT5 (attempt {attempt + 1}/{self.max_retries})")
                
                if not mt5.initialize(
                    login=int(config.config['MT5_LOGIN']),
                    password=config.config['MT5_PASSWORD'],
                    server=config.config['MT5_SERVER'],
                    timeout=self.timeout,
                    portable=False
                ):
                    error = mt5.last_error()
                    logger.error(f"Connection failed: {error}")
                    
                    # Specific error handling
                    if "invalid account" in error.description.lower():
                        logger.critical("Invalid account credentials")
                        return False
                    if "no connection" in error.description.lower():
                        logger.error("Network connection issue")
                    
                    time.sleep(self.retry_delay)
                    continue
                
                self.connected = True
                self.last_connection_time = datetime.now()
                logger.info("Connected successfully to MT5 terminal")
                
                # Verify terminal status
                terminal_info = mt5.terminal_info()
                if not terminal_info or not terminal_info.connected:
                    logger.error("Terminal not properly connected")
                    self.disconnect()
                    continue
                    
                # Preload symbol info
                self._preload_symbol_info()
                return True
                
            except Exception as e:
                logger.error(f"Connection error: {str(e)}")
                time.sleep(self.retry_delay)
        
        logger.error("Failed to connect after maximum retries")
        return False

    def _preload_symbol_info(self):
        """Preload information for all configured symbols."""
        for symbol in config.config["SYMBOL_OPTIONS"]:
            try:
                if not mt5.symbol_select(symbol, True):
                    logger.error(f"Failed to select {symbol} in Market Watch. Ensure symbol is available on server.")
                    continue
                info = mt5.symbol_info(symbol)
                if info:
                    self.symbol_info_cache[symbol] = info._asdict()
                    #logger.info(f"Preloaded symbol info for {symbol}")
                else:
                    logger.warning(f"No symbol info retrieved for {symbol}")
            except Exception as e:
                logger.error(f"Error preloading {symbol}: {str(e)}")
                
    def disconnect(self) -> None:
        """Cleanly shutdown MT5 connection."""
        try:
            if self.connected:
                mt5.shutdown()
                self.connected = False
                self.symbol_info_cache.clear()
                self.data_cache.clear()
                logger.info("Disconnected from MT5 and cleared caches")
        except Exception as e:
            logger.error(f"Disconnection error: {str(e)}")

    def check_connection(self) -> bool:
        """Verify and maintain active connection with health checks."""
        if not self.connected:
            return self.connect()
        
        # Check connection health
        try:
            # Simple ping-like check
            if not mt5.terminal_info() or not mt5.account_info():
                logger.warning("Connection health check failed, reconnecting...")
                self.disconnect()
                return self.connect()
                
            self.last_connection_time = datetime.now()
            return True
            
        except Exception as e:
            logger.error(f"Connection check failed: {str(e)}")
            self.disconnect()
            return self.connect()

    # --------------------------
    # Trading Methods
    # --------------------------
    
    def place_trade(self, symbol: str, trade_type: str, volume: float, 
                  entry: float, sl: float, tp: float, comment: str = "") -> Dict:
        """Execute trade with comprehensive error handling and validation."""
        try:
            # Connection check with retry
            if not self.check_connection():
                return {"error": "Failed to establish MT5 connection", "retcode": None}
            
            # Trade rate limiting
            if self.last_trade_time and (datetime.now() - self.last_trade_time).seconds < self.trade_timeout:
                return {"error": "Trade rate limit exceeded", "retcode": None}
            
            # Validate inputs
            if not self.validate_symbol(symbol):
                return {"error": f"Invalid symbol: {symbol}", "retcode": None}
                
            if volume <= 0:
                return {"error": "Volume must be positive", "retcode": None}
                
            # Ensure symbol is selected in Market Watch
            if not mt5.symbol_select(symbol, True):
                return {"error": f"Failed to select {symbol} in Market Watch", "retcode": None}
            
            # Get current symbol info
            symbol_info = mt5.symbol_info(symbol)
            if not symbol_info:
                return {"error": f"Failed to get symbol info for {symbol}", "retcode": None}
            
            # Determine order type and price
            order_type = mt5.ORDER_TYPE_BUY if trade_type == "BUY" else mt5.ORDER_TYPE_SELL
            price = symbol_info.ask if trade_type == "BUY" else symbol_info.bid
            
            # Validate stop levels
            if trade_type == "BUY":
                if sl >= price and sl != 0:
                    return {"error": "Stop loss must be below entry for BUY", "retcode": None}
                if tp <= price and tp != 0:
                    return {"error": "Take profit must be above entry for BUY", "retcode": None}
            else:
                if sl <= price and sl != 0:
                    return {"error": "Stop loss must be above entry for SELL", "retcode": None}
                if tp >= price and tp != 0:
                    return {"error": "Take profit must be below entry for SELL", "retcode": None}
            
            # Prepare trade request
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": float(volume),
                "type": order_type,
                "price": price,
                "sl": float(sl) if sl != 0 else 0.0,
                "tp": float(tp) if tp != 0 else 0.0,
                "deviation": 10,
                "magic": 20230815,
                "comment": comment,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": self._determine_filling_mode(symbol_info.filling_mode),
            }
            
            # Execute trade with timeout
            start_time = time.time()
            result = None
            
            while time.time() - start_time < 10:  # 10 second timeout
                result = mt5.order_send(request)
                if result is not None:
                    break
                time.sleep(0.5)
            
            if result is None:
                return {"error": "No response from MT5 after multiple attempts", "retcode": None}
            
            # Process result
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                error_msg = self._get_error_message(result.retcode)
                return {"error": f"Trade failed: {error_msg}", "retcode": result.retcode}
            
            # Success
            self.last_trade_time = datetime.now()
            logger.info(f"Trade executed successfully - Ticket: {result.deal}, Price: {result.price}")
            
            return {
                "ticket": result.deal,
                "price": result.price,
                "volume": result.volume,
                "retcode": result.retcode,
                "order": result.order,
                "request_id": result.request_id
            }

        except Exception as e:
            logger.error(f"Trade execution error: {str(e)}", exc_info=True)
            return {"error": str(e), "retcode": None}

    # --------------------------
    # Market Data Methods
    # --------------------------
    def fetch_data(self, symbol: str, timeframe: str, min_candles: int = 200) -> Optional[pd.DataFrame]:
        if not self.validate_symbol_timeframe(symbol, timeframe):
            return None
    
        cache_key = f"{symbol}_{timeframe}"
        if cache_key in self.data_cache:
            cached = self.data_cache[cache_key]
            if (datetime.now() - cached['timestamp']).seconds < config.config["TIMEFRAME_MINUTES"].get(timeframe, 15)*60:
                if len(cached['data']) >= min_candles:
                    return cached['data'].copy()
    
        if not self.check_connection():
            return None
    
        try:
            tf_map = {
                'M1': mt5.TIMEFRAME_M1, 'M5': mt5.TIMEFRAME_M5,
                'M15': mt5.TIMEFRAME_M15, 'M30': mt5.TIMEFRAME_M30,
                'H1': mt5.TIMEFRAME_H1, 'H4': mt5.TIMEFRAME_H4,
                'D1': mt5.TIMEFRAME_D1
            }
        
            required_candles = max(min_candles, 200)
            count = self._calculate_candle_count(timeframe, required_candles)
        
            rates = None
            attempts = 0
            max_attempts = 3
            
            logger.info("-" * 75)
            logger.info(f"Fetching {count} candles for {symbol} on {timeframe}")
            while attempts < max_attempts and (rates is None or len(rates) < required_candles):
                try:
                    rates = mt5.copy_rates_from_pos(symbol, tf_map[timeframe], 0, count)
                    if rates is None or len(rates) < required_candles:
                        logger.warning(f"Attempt {attempts + 1}: Retrieved {len(rates) if rates else 0} candles, trying alternative method")
                        rates = mt5.copy_rates_from(symbol, tf_map[timeframe], datetime.now(), count)
                
                    attempts += 1
                    if rates is None or len(rates) < required_candles:
                        logger.warning(f"Attempt {attempts}: Retrieved {len(rates) if rates else 0} candles, retrying in 1 second")
                        time.sleep(1)
                except Exception as e:
                    logger.error(f"Attempt {attempts + 1} failed: {str(e)}")
                    attempts += 1
                    time.sleep(1)
                    continue
        
            if rates is None or len(rates) < required_candles:
                logger.error(f"Failed to fetch sufficient data for {symbol} ({timeframe}): {len(rates) if rates else 0}/{required_candles} candles")
                return None
        
            logger.info(f"Successfully fetched {len(rates)} candles for {symbol} ({timeframe})")
            df = self._process_dataframe(rates, symbol, timeframe, required_candles)
            if df is not None and len(df) >= 50:  # Ensure enough rows for indicators
                self.data_cache[cache_key] = {'timestamp': datetime.now(), 'data': df.copy()}
                return df
            else:
                logger.error(f"Processed DataFrame has insufficient rows: {len(df) if df is not None else 0}/50")
                return None
    
        except Exception as e:
            logger.error(f"Data fetch error for symbol {symbol}: {str(e)}", exc_info=True)
            return None        
    def get_current_price(self, symbol: str) -> Optional[Dict]:
        """Get current market prices with comprehensive error handling."""
        if not self.validate_symbol(symbol):
            return None
            
        if not self.check_connection():
            return None
            
        try:
            # Try multiple times to get tick data
            tick = None
            for _ in range(3):
                tick = mt5.symbol_info_tick(symbol)
                if tick is not None:
                    break
                time.sleep(0.5)
            
            if tick is None:
                logger.error(f"Failed to get tick data for {symbol}")
                return None
                
            spread_pips = (tick.ask - tick.bid) * (10 ** 4)
            return {
                "bid": tick.bid,
                "ask": tick.ask,
                "last": tick.last,
                "volume": tick.volume,
                "time": pd.to_datetime(tick.time, unit='s').tz_localize('UTC').astimezone(self.timezone),
                "spread_pips": spread_pips
            }
        except Exception as e:
            logger.error(f"Price fetch error for {symbol}: {str(e)}", exc_info=True)
            return None

    # --------------------------
    # Helper Methods
    # --------------------------
    def _calculate_candle_count(self, timeframe: str, min_candles: int) -> int:
        """Determine optimal candle count based on timeframe."""
        # Use configured minimum bars or fallback to reasonable defaults
        return max(
            config.config["MIN_DATA_BARS"].get(timeframe, 1000),
            min_candles * 3
        )
        
    def _process_dataframe(self, rates, symbol: str, timeframe: str, min_candles: int) -> Optional[pd.DataFrame]:
        """Convert and validate rate data to DataFrame."""
        try:
            df = pd.DataFrame(rates)
            if df.empty or len(df) < min_candles//2:
                logger.warning(f"Insufficient data for {symbol} ({timeframe}): {len(df)} candles")
                return None
                
            # Convert and clean data
            df['time'] = pd.to_datetime(df['time'], unit='s').dt.tz_localize('UTC').dt.tz_convert(self.timezone)
            df = df.dropna()
            df = df[df['high'] >= df['low']]  # Remove invalid candles
            
            # Add derived columns
            df['range'] = df['high'] - df['low']
            df['body'] = abs(df['close'] - df['open'])
            df['direction'] = np.where(df['close'] > df['open'], 1, -1)
            
            return df
        except Exception as e:
            logger.error(f"Data processing error: {str(e)}")
            return None

    def _determine_filling_mode(self, filling_mode: int) -> int:
        """Select appropriate filling mode based on symbol support."""
        if filling_mode == 1: return mt5.ORDER_FILLING_FOK
        elif filling_mode == 2: return mt5.ORDER_FILLING_RETURN
        elif filling_mode == 4: return mt5.ORDER_FILLING_IOC
        elif filling_mode in [3,5,6,7]: return mt5.ORDER_FILLING_RETURN
        return mt5.ORDER_FILLING_FOK

    def _get_error_message(self, retcode: int) -> str:
        """Convert MT5 error code to human-readable message."""
        errors = {
            10004: "Requote",
            10006: "Request rejected",
            10007: "Request canceled",
            10008: "Order placed",
            10013: "Invalid request",
            10014: "Invalid volume",
            10015: "Invalid price",
            10016: "Invalid stops",
            10017: "Trade disabled",
            10018: "Market closed",
            10019: "Insufficient funds",
            10020: "Prices changed",
            10021: "Prices unavailable",
            10022: "Order locked",
            10023: "Long positions only allowed",
            10024: "Too many requests",
            10025: "Order modify denied",
            10026: "Order close denied",
            10027: "Order delete denied",
            10028: "Order suspend denied",
            10029: "Order modify failed",
            10030: "Order close failed"
        }
        return errors.get(retcode, f"Unknown error (code: {retcode})")

    def validate_symbol(self, symbol: str) -> bool:
        """Check if symbol is valid."""
        if symbol not in config.config["SYMBOL_OPTIONS"]:
            logger.error(f"Invalid symbol: {symbol}")
            return False
        return True

    def validate_symbol_timeframe(self, symbol: str, timeframe: str) -> bool:
        """Validate symbol and timeframe combination."""
        if not self.validate_symbol(symbol):
            return False
        if timeframe not in config.config["TIMEFRAME_OPTIONS"]:
            logger.error(f"Invalid timeframe: {timeframe}")
            return False
        return True

    def __del__(self):
        """Ensure clean shutdown."""
        self.disconnect()