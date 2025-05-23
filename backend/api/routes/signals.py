from fastapi import APIRouter, HTTPException, WebSocket
from api.services.forecasting import ForecastingService
from api.models.signal import Signal
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)
router = APIRouter()
websocket_clients = set()

@router.get("/", response_model=List[Signal])
async def get_signals(symbol: Optional[str] = None, timeframe: Optional[str] = None):
    try:
        forecasting_service = ForecastingService()
        signals = forecasting_service.get_signals(symbol, timeframe)
        return [Signal(**signal) for signal in signals]
    except Exception as e:
        logger.error(f"Error fetching signals: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching signals: {str(e)}")

@router.get("/recent", response_model=List[Signal])
async def get_recent_signals(limit: int = 5):
    try:
        forecasting_service = ForecastingService()
        signals = forecasting_service.get_recent_signals(limit)
        return [Signal(**signal) for signal in signals]
    except Exception as e:
        logger.error(f"Error fetching recent signals: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching recent signals: {str(e)}")

# Broadcast new signals to connected WebSocket clients
async def broadcast_signal(signal: dict):
    signal_data = Signal(**signal).dict()
    for client in list(websocket_clients):
        try:
            await client.send_json(signal_data)
        except Exception as e:
            logger.error(f"Error broadcasting signal to client: {str(e)}")
            websocket_clients.remove(client)

@router.websocket("/ws/signals")
async def websocket_signals(websocket: WebSocket):
    await websocket.accept()
    websocket_clients.add(websocket)
    try:
        while True:
            await websocket.receive_text()
    except Exception:
        websocket_clients.remove(websocket)
    finally:
        await websocket.close()