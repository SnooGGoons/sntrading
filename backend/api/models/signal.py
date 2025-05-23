import os
import logging
from logging.handlers import RotatingFileHandler
from typing import Dict, Any
# ====================== PATH CONFIGURATION ======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
PLOT_DIR = os.path.join(BASE_DIR, "plots")
DATA_DIR = os.path.join(BASE_DIR, "data")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# ====================== LOG INITIALIZATION ======================
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG if os.getenv("DEBUG", "False") == "True" else logging.INFO)
file_handler = RotatingFileHandler(
    os.path.join(LOGS_DIR, "trading.log"),
    maxBytes=10 * 1024 * 1024,
    backupCount=5
)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
console_handler.setLevel(logging.DEBUG if os.getenv("DEBUG", "False") == "True" else logging.INFO)
logger.addHandler(console_handler)

# ====================== ENVIRONMENT VARIABLES ======================
def get_env_var(name: str, default: str) -> str:
    value = os.getenv(name, default)
    return value

MT5_CREDS = {
    "LOGIN": get_env_var("MT5_LOGIN", "57210893"),
    "PASSWORD": get_env_var("MT5_PASSWORD", "Vision2030!"),
    "SERVER": get_env_var("MT5_SERVER", "HFMarketsKE-Demo2"),
    "TIMEOUT": int(get_env_var("MT5_TIMEOUT", "30"))
}

MYSQL_CONFIG = {
    "HOST": get_env_var("MYSQL_HOST", "localhost"),
    "USER": get_env_var("MYSQL_USER", "root"),
    "PASSWORD": get_env_var("MYSQL_PASSWORD", "Vision2030!"),
    "DATABASE": get_env_var("MYSQL_DATABASE", "trading_insights"),
    "PORT": int(get_env_var("MYSQL_PORT", "3306"))
}

# ====================== TRADING CONFIGURATION ======================
SYMBOL_OPTIONS = ["XAUUSD", "EURUSD", "GBPUSD", "USDJPY", "GBPJPY", "AUDUSD", "NZDUSD"]
TIMEFRAME_OPTIONS = ["M1", "M5", "M15", "M30", "H1", "H4", "D1"]

TIMEFRAME_MINUTES = {
    "M1": 1,
    "M5": 5,
    "M15": 15,
    "M30": 30,
    "H1": 60,
    "H4": 240,
    "D1": 1440
}

MIN_DATA_BARS={
    "M1": 5000,   # 3.5 days
    "M5": 2000,   # 7 days
    "M15": 1000,  # 10.5 days
    "M30": 500,   # 10.5 days
    "H1": 336,    # 14 days
    "H4": 168,    # 28 days
    "D1": 90      # 90 days
}

TRADING_SESSIONS_KE = {
    "Sydney":    {"open": "00:00", "close": "09:00"},
    "Tokyo":     {"open": "03:00", "close": "12:00"},
    "Asian":     {"open": "02:00", "close": "11:00"},
    "London":    {"open": "11:00", "close": "20:00"},
    "New York":  {"open": "16:00", "close": "01:00"},  # next day
}

SESSION_OVERLAPS_KE = {
    "Sydney-Tokyo Overlap": {"start": "03:00", "end": "09:00"},
    "Tokyo-London Overlap": {"start": "11:00", "end": "12:00"},
    "London-New York Overlap": {"start": "16:00", "end": "20:00"},
}


TRADING_DAYS = {0, 1, 2, 3, 4}  # Monday to Friday

# ====================== MODEL CONFIGURATION ======================
MODEL_CONFIG = {
    "TIME_STEPS": 60,
    "BATCH_SIZE": 32,
    "EPOCHS": 100,
    "LEARNING_RATE": 0.001,
    "DROPOUT_RATE": 0.2,
    "LOSS": "mse",
    "OPTIMIZER": "adam",
    "METRICS": ["mae"],
    "EARLY_STOPPING_PATIENCE": 10,
    "MIN_DELTA": 0.001
}

# ====================== SIGNAL CONFIGURATION ======================
SIGNAL_CONFIG = {
    "LONG_TERM_EXPIRY_HOURS": 24,
    "SHORT_TERM_EXPIRY_HOURS": 4,
    "MIN_CONFIDENCE_LONG": 80.0,
    "MIN_CONFIDENCE_SHORT": 70.0,
    "STATUS_OPTIONS": [
        "Open",
        "Closed - TP Hit",
        "Closed - SL Hit",
        "Closed - Expired",
        "Closed - Manual"
    ]
}


# ====================== PRICE ACTION CONFIGURATION ======================
PRICE_ACTION = {
    "Trending": {"lookback_candles": 50, "volatility_threshold": 0.015, "support_resistance_threshold": 0.5},
    "Range-bound": {"lookback_candles": 50, "volatility_threshold": 0.008, "support_resistance_threshold": 0.3},
    "Volatile": {"lookback_candles": 50, "volatility_threshold": 0.02, "support_resistance_threshold": 0.7},
    "Neutral": {"lookback_candles": 50, "volatility_threshold": 0.01, "support_resistance_threshold": 0.4},
    "DEFAULT": {"lookback_candles": 50, "volatility_threshold": 0.01, "support_resistance_threshold": 0.5}  # Add default
}

# ====================== REGIME ======================
REGIME_CONFIG = {
    "Trending": {
        "weights": {
            "prediction_accuracy": 0.20,
            "trend_strength": 0.40,
            "momentum": 0.25,
            "volatility_filters": 0.15
        },
        "min_confidence_long": 80.0,
        "min_confidence_short": 70.0,
        "position_size_factor": 1.0,
        "sl_tp_multipliers": {
            "Long-Term": {"sl": 2.0, "tp": 5.0},
            "Short-Term": {"sl": 1.5, "tp": 3.0}
        }
    },
    "Range-bound": {
        "weights": {
            "prediction_accuracy": 0.20,
            "trend_strength": 0.20,
            "momentum": 0.25,
            "volatility_filters": 0.35
        },
        "min_confidence_long": 75.0,
        "min_confidence_short": 65.0,
        "position_size_factor": 1.2,
        "sl_tp_multipliers": {
            "Long-Term": {"sl": 1.2, "tp": 2.0},
            "Short-Term": {"sl": 1.0, "tp": 1.5}
        }
    },
    "Volatile": {
        "weights": {
            "prediction_accuracy": 0.20,
            "trend_strength": 0.20,
            "momentum": 0.25,
            "volatility_filters": 0.35
        },
        "min_confidence_long": 85.0,
        "min_confidence_short": 75.0,
        "position_size_factor": 0.5,
        "sl_tp_multipliers": {
            "Long-Term": {"sl": 2.5, "tp": 3.0},
            "Short-Term": {"sl": 2.0, "tp": 2.5}
        }
    },
    "Neutral": {
        "weights": {
            "prediction_accuracy": 0.20,
            "trend_strength": 0.30,
            "momentum": 0.30,
            "volatility_filters": 0.20
        },
        "min_confidence_long": 80.0,
        "min_confidence_short": 70.0,
        "position_size_factor": 1.0,
        "sl_tp_multipliers": {
            "Long-Term": {"sl": 2.0, "tp": 4.0},
            "Short-Term": {"sl": 1.5, "tp": 2.5}
        }
    }
}

# ====================== MAIN CONFIGURATION ======================
config: Dict[str, Any] = {
    # Paths
    "MODEL_PATH": os.path.join(MODEL_DIR, "lstm_model.keras"),
    "SCALER_PATH": os.path.join(MODEL_DIR, "scaler.pkl"),
    "PLOT_DIR": PLOT_DIR,
    "DATA_DIR": DATA_DIR,
    
    # MetaTrader 5
    "MT5_LOGIN": MT5_CREDS["LOGIN"],
    "MT5_PASSWORD": MT5_CREDS["PASSWORD"],
    "MT5_SERVER": MT5_CREDS["SERVER"],
    
    # MySQL
    "MYSQL_HOST": MYSQL_CONFIG["HOST"],
    "MYSQL_USER": MYSQL_CONFIG["USER"],
    "MYSQL_PASSWORD": MYSQL_CONFIG["PASSWORD"],
    "MYSQL_DATABASE": MYSQL_CONFIG["DATABASE"],
    "MYSQL_PORT": MYSQL_CONFIG["PORT"],
    
    # GUI Settings
    "BUTTON_SIZE": (120, 60),
    "WINDOW_SIZE": (800, 600),
    "COMPACT_WINDOW_SIZE": (60, 60),
    "THEME": "dark",
    #PRICE ACTION
    "PRICE_ACTION": PRICE_ACTION,

    # Trading Parameters
    "DEFAULT_SYMBOL": "XAUUSD",
    "DEFAULT_TIMEFRAME": "M15",
    "SYMBOL_OPTIONS": SYMBOL_OPTIONS,
    "TIMEFRAME_OPTIONS": TIMEFRAME_OPTIONS,
    "TIMEFRAME_MINUTES": TIMEFRAME_MINUTES,
    "TRADING_SESSIONS": TRADING_SESSIONS_KE,
    "TRADING_DAYS": TRADING_DAYS,
    "MIN_DATA_BARS": MIN_DATA_BARS,
    
    # Signal Configuration
    "SIGNAL_CONFIG": SIGNAL_CONFIG,
    
    "REGIME_CONFIG": REGIME_CONFIG,
                
    # Model Parameters
    "VOLATILITY_THRESHOLD": 0.02,
    "MIN_CONFIDENCE": 75.0,
    "METRICS_WINDOW": 20,
    "TIME_STEPS": 100,
    
    # Auto-Trading
    "AUTO_TRADE_ENABLED": True,
    "AUTO_TRADE_VOLUME": 0.1,
    "AUTO_TRADE_RISK_PERCENT": 1.0,
    "AUTO_TRADE_MIN_CONFIDENCE_LONG": 80.0,
    "AUTO_TRADE_MIN_CONFIDENCE_SHORT": 70.0,
    "AUTO_TRADE_MAX_OPEN_TRADES": 3,
    
    # Report Settings
    "REPORT_TEMPLATE": "professional",
    "REPORT_AUTHOR": "Trading Assistant",
    "REPORT_COMPANY": "SnooG Trading Systems",
    
    # Risk Management
    "MAX_DAILY_LOSS_PERCENT": 2.0,
    "MAX_TRADE_RISK_PERCENT": 1.0,
    "MIN_RISK_REWARD_RATIO": 1.5,
    
    # Notification settings
    "NOTIFICATION_DURATION": 7000,
    
    # Performance Monitoring
    "PERF_MONITOR_INTERVAL": 300,
    "ALERT_THRESHOLDS": {
        "DRAWDOWN": 5.0,
        "WIN_RATE": 60.0,
        "SHARPE_RATIO": 1.0,
        "MAX_CONSECUTIVE_LOSSES": 3
    }    
}

# ====================== VALIDATION ======================
def validate_config():
    """Validate critical configuration values and log essential settings."""
    logger.info("-" * 75)
    logger.info(" SnooG Trading Strategy")
    logger.info("-" * 75)
    
    required_dirs = [MODEL_DIR, PLOT_DIR, DATA_DIR, REPORTS_DIR, LOGS_DIR]
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            logger.info(f"Created directory: {dir_path}")
    
    if config["MT5_PASSWORD"] == "Vision2030!":
        logger.warning("Using default MT5 password - consider changing in production")
    
    if not (0 < config["MYSQL_PORT"] <= 65535):
        raise ValueError(f"Invalid MySQL port: {config['MYSQL_PORT']}")
    
    if not (1000 <= config["NOTIFICATION_DURATION"] <= 30000):
        logger.warning(f"Invalid NOTIFICATION_DURATION: {config['NOTIFICATION_DURATION']}ms. Setting to 5000ms")
        config["NOTIFICATION_DURATION"] = 5000
    
    # Validate REGIME_CONFIG
    regimes = ["Trending", "Range-bound", "Volatile", "Neutral"]
    for regime in regimes:
        if regime not in config["REGIME_CONFIG"]:
            raise ValueError(f"Missing REGIME_CONFIG for {regime}")
        weights = config["REGIME_CONFIG"][regime]["weights"]
        if abs(sum(weights.values()) - 1.0) > 0.01:
            raise ValueError(f"Invalid weights sum for {regime}: {sum(weights.values())}")
        if not (0 < config["REGIME_CONFIG"][regime]["position_size_factor"] <= 2.0):
            raise ValueError(f"Invalid position_size_factor for {regime}")
        for signal_type in ["Long-Term", "Short-Term"]:
            if signal_type not in config["REGIME_CONFIG"][regime]["sl_tp_multipliers"]:
                raise ValueError(f"Missing sl_tp_multipliers for {signal_type} in {regime}")
    
    logger.info("Configuration loaded successfully")
    logger.info(f"MT5 Server: {config['MT5_SERVER']}")
    logger.info(f"MySQL Host: {config['MYSQL_HOST']}:{config['MYSQL_PORT']}")
    logger.info(f"Auto Trade Enabled: {config['AUTO_TRADE_ENABLED']}")
    logger.info(f"Default Symbol: {config['DEFAULT_SYMBOL']} ({config['DEFAULT_TIMEFRAME']})")
    logger.info(f"Available Symbols: {', '.join(config['SYMBOL_OPTIONS'])}")
    logger.info(f"Notification Duration: {config['NOTIFICATION_DURATION']}ms")
    logger.info(f"Regime Config: {list(config['REGIME_CONFIG'].keys())}")
    logger.info("Configuration validation passed")

validate_config()