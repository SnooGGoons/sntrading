from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from api.routes import signals, plots, metrics, trades
import logging

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Snoog Forex Trading Suite API",
    description="API for forex trading signals, plots, metrics, and trades",
    version="1.0.0"
)

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(signals.router, prefix="/api/signals", tags=["Signals"])
app.include_router(plots.router, prefix="/api/plots", tags=["Plots"])
app.include_router(metrics.router, prefix="/api/metrics", tags=["Metrics"])
app.include_router(trades.router, prefix="/api/trades", tags=["Trades"])

@app.get("/")
async def root():
    return {"message": "Snoog Forex Trading Suite API is running"}

@app.websocket("/ws/signals")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Keep connection alive; signals are pushed via services
            await websocket.receive_text()
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
    finally:
        await websocket.close()