from fastapi import APIRouter, HTTPException
from api.services.forecasting import ForecastingService
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/")
async def get_metrics():
    try:
        forecasting_service = ForecastingService()
        metrics = forecasting_service.get_metrics()
        return metrics
    except Exception as e:
        logger.error(f"Error fetching metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching metrics: {str(e)}")