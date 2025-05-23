from utils.forecaster import ForecastingThread
from utils.database import Database
from utils.config import config
from api.routes.signals import broadcast_signal
import logging
import asyncio

logger = logging.getLogger(__name__)

class ForecastingService:
    def __init__(self):
        self.db = Database()
        self.forecasters = {}

    def initialize_forecaster(self, symbol, timeframe):
        if (symbol, timeframe) not in self.forecasters:
            forecaster = ForecastingThread(symbol=symbol, timeframe=timeframe)
            forecaster.signal_generated.connect(self._handle_signal)
            forecaster.start()
            self.forecasters[(symbol, timeframe)] = forecaster
            logger.info(f"Initialized forecaster for {symbol} on {timeframe}")

    def _handle_signal(self, signal):
        # Broadcast signal to WebSocket clients
        asyncio.run(broadcast_signal(signal))
        logger.info(f"New signal for {signal['symbol']} on {signal['timeframe']}")

    def get_signals(self, symbol: str = None, timeframe: str = None):
        if symbol and timeframe:
            self.initialize_forecaster(symbol, timeframe)
        return self.db.get_open_signals(symbol, timeframe)

    def get_recent_signals(self, limit: int = 5):
        return self.db.get_recent_signals(limit)

    def get_metrics(self):
        return self.db.get_signal_stats()