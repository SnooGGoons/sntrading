import React, { useEffect, useRef } from 'react';
import { gsap } from 'gsap';

function ChartViewer({ symbol, timeframe, apiBaseUrl }) {
  const iframeRef = useRef(null);

  useEffect(() => {
    const fetchChart = async () => {
      try {
        const response = await fetch(`${apiBaseUrl}/api/plots/${symbol}/${timeframe}`);
        if (response.ok) {
          const blob = await response.blob();
          const url = URL.createObjectURL(blob);
          iframeRef.current.src = url;
          gsap.from(iframeRef.current, { opacity: 0, duration: 0.5 });
        }
      } catch (error) {
        console.error('Error fetching chart:', error);
      }
    };
    fetchChart();
  }, [symbol, timeframe, apiBaseUrl]);

  return (
    <div className="chart-viewer bg-gray-800 p-6 rounded-lg shadow-lg col-span-1 md:col-span-2 lg:col-span-3">
      <h2 className="text-xl font-semibold mb-4">{symbol} - {timeframe} Chart</h2>
      <iframe ref={iframeRef} title="Trading Chart" className="w-full h-[600px] rounded-lg" />
    </div>
  );
}

export default ChartViewer;