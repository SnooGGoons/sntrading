from utils.mt5_trading import MT5Trader
import logging

logger = logging.getLogger(__name__)

class TradingService:
    def __init__(self):
        self.mt5_trader = MT5Trader()

    def execute_trade(self, symbol: str, trade_type: str, volume: float, entry: float, sl: float, tp: float, comment: str):
        return self.mt5_trader.place_trade(
            symbol=symbol,
            trade_type=trade_type,
            volume=volume,
            entry=entry,
            sl=sl,
            tp=tp,
            comment=comment
        )

    def get_current_price(self, symbol: str):
        return self.mt5_trader.get_current_price(symbol)