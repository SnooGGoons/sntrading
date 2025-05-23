from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from api.services.plotting import PlottingService
import os
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/{symbol}/{timeframe}")
async def get_plot(symbol: str, timeframe: str):
    try:
        plotting_service = PlottingService()
        plot_path = plotting_service.generate_plot(symbol, timeframe)
        if not plot_path or not os.path.exists(plot_path):
            raise HTTPException(status_code=404, detail="Plot not found")
        return FileResponse(plot_path, media_type="text/html")
    except Exception as e:
        logger.error(f"Error generating plot: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating plot: {str(e)}")