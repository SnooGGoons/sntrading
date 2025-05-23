import React, { useState, useEffect } from 'react';
import { gsap } from 'gsap';
import SignalCard from './components/SignalCard';
import ChartViewer from './components/ChartViewer';
import MetricsPanel from './components/MetricsPanel';
import TradePanel from './components/TradePanel';
import DataDownload from './components/DataDownload';
import ForecastPanel from './components/ForecastPanel';
import './App.css';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

function App() {
  const [signals, setSignals] = useState([]);
  const [metrics, setMetrics] = useState({});
  const [selectedSymbol, setSelectedSymbol] = useState('XAUUSD');
  const [selectedTimeframe, setSelectedTimeframe] = useState('M15');
  const [isExpanded, setIsExpanded] = useState(true);

  useEffect(() => {
    fetchSignals();
    fetchMetrics();
    setupWebSocket();
  }, [selectedSymbol, selectedTimeframe]);

  useEffect(() => {
    gsap.to('.dashboard', {
      opacity: isExpanded ? 1 : 0,
      height: isExpanded ? 'auto' : 0,
      duration: 0.5,
      ease: 'power2.inOut'
    });
  }, [isExpanded]);

  const fetchSignals = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/signals?symbol=${selectedSymbol}&timeframe=${selectedTimeframe}`);
      const data = await response.json();
      setSignals(data);
    } catch (error) {
      console.error('Error fetching signals:', error);
    }
  };

  const fetchMetrics = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/metrics`);
      const data = await response.json();
      setMetrics(data);
    } catch (error) {
      console.error('Error fetching metrics:', error);
    }
  };

  const setupWebSocket = () => {
    const ws = new WebSocket(`ws://${API_BASE_URL.replace('http://', '')}/ws/signals`);
    ws.onmessage = (event) => {
      const signal = JSON.parse(event.data);
      if (signal.symbol === selectedSymbol && signal.timeframe === selectedTimeframe) {
        setSignals((prev) => [signal, ...prev.slice(0, 4)]);
        gsap.fromTo('.signal-card', { scale: 0.9, opacity: 0 }, { scale: 1, opacity: 1, duration: 0.3 });
      }
    };
    ws.onclose = () => setTimeout(setupWebSocket, 5000);
    return () => ws.close();
  };

  const toggleDashboard = () => {
    setIsExpanded(!isExpanded);
  };

  return (
    <div className="min-h-screen bg-gray-900 text-gray-100">
      <header className="p-4 flex justify-between items-center">
        <h1 className="text-2xl font-bold">Snoog Forex Trading Suite</h1>
        <button
          onClick={toggleDashboard}
          className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-full"
        >
          {isExpanded ? 'Collapse' : 'Expand'}
        </button>
      </header>
      <div className="dashboard p-4 space-y-6">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          <SignalCard signals={signals} />
          <MetricsPanel metrics={metrics} />
          <TradePanel
            symbol={selectedSymbol}
            timeframe={selectedTimeframe}
            apiBaseUrl={API_BASE_URL}
          />
          <DataDownload
            symbol={selectedSymbol}
            setSymbol={setSelectedSymbol}
            timeframe={selectedTimeframe}
            setTimeframe={setSelectedTimeframe}
          />
          <ForecastPanel symbol={selectedSymbol} timeframe={selectedTimeframe} />
          <ChartViewer
            symbol={selectedSymbol}
            timeframe={selectedTimeframe}
            apiBaseUrl={API_BASE_URL}
          />
        </div>
      </div>
      <footer className="p-4 text-center text-gray-500">
        © 2025 Snoog Forex Trading Suite. All rights reserved. This is a simulation tool and should not be considered as financial advice.
      </footer>
    </div>
  );
}

export default App;