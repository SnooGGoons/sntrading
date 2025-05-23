from utils.plotter import Plotter
from utils.config import config
import logging

logger = logging.getLogger(__name__)

class PlottingService:
    def __init__(self):
        self.plotter = Plotter()

    def generate_plot(self, symbol: str, timeframe: str):
        if symbol not in config["SYMBOL_OPTIONS"] or timeframe not in config["TIMEFRAME_OPTIONS"]:
            raise ValueError("Invalid symbol or timeframe")
        success = self.plotter.generate_plot(symbol, timeframe)
        if success:
            return f"{config['PLOT_DIR']}/{symbol}_{timeframe}_analysis.html"
        return None