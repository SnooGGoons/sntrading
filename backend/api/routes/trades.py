from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from api.services.trading import TradingService
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

class TradeRequest(BaseModel):
    symbol: str
    trade_type: str
    volume: float
    entry: float
    sl: float
    tp: float
    signal_type: str

@router.post("/execute")
async def execute_trade(trade: TradeRequest):
    try:
        trading_service = TradingService()
        result = trading_service.execute_trade(
            symbol=trade.symbol,
            trade_type=trade.trade_type,
            volume=trade.volume,
            entry=trade.entry,
            sl=trade.sl,
            tp=trade.tp,
            comment=f"ManualTrade_{trade.trade_type}_{trade.signal_type}"
        )
        if result and result.get("retcode") == 10009:
            return {"status": "success", "ticket": result["ticket"], "message": f"Trade executed: Ticket {result['ticket']}"}
        else:
            error = result.get("error", "Unknown error") if result else "No response from MT5"
            raise HTTPException(status_code=400, detail=f"Trade failed: {error}")
    except Exception as e:
        logger.error(f"Error executing trade: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error executing trade: {str(e)}")