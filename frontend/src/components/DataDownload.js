import React, { useState } from 'react';

function DataDownload({ symbol, setSymbol, timeframe, setTimeframe }) {
  const [dateRange, setDateRange] = useState('Last Week');

  const handleDownload = () => {
    // Placeholder for data download API call
    alert(`Downloading data for ${symbol} - ${timeframe} - ${dateRange}`);
  };

  return (
    <div className="bg-gray-800 p-6 rounded-lg shadow-lg">
      <h2 className="text-xl font-semibold mb-4">Currency Data Download</h2>
      <div className="space-y-4">
        <div>
          <label className="block text-gray-400">Currency Pair</label>
          <select
            value={symbol}
            onChange={(e) => setSymbol(e.target.value)}
            className="w-full bg-gray-700 text-white p-2 rounded"
          >
            {['XAUUSD', 'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD'].map(s => (
              <option key={s} value={s}>{s}</option>
            ))}
          </select>
        </div>
        <div>
          <label className="block text-gray-400">Timeframe</label>
          <select
            value={timeframe}
            onChange={(e) => setTimeframe(e.target.value)}
            className="w-full bg-gray-700 text-white p-2 rounded"
          >
            {['M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1', 'W1', 'MN1'].map(t => (
              <option key={t} value={t}>{t}</option>
            ))}
          </select>
        </div>
        <div>
          <label className="block text-gray-400">Date Range</label>
          <select
            value={dateRange}
            onChange={(e) => setDateRange(e.target.value)}
            className="w-full bg-gray-700 text-white p-2 rounded"
          >
            {['Last Day', 'Last Week', 'Last Month', 'Last 3 Months', 'Last 6 Months', 'Last Year', 'Custom Range'].map(r => (
              <option key={r} value={r}>{r}</option>
            ))}
          </select>
        </div>
        <button
          onClick={handleDownload}
          className="w-full bg-blue-600 hover:bg-blue-700 text-white p-2 rounded"
        >
          Download Data
        </button>
      </div>
    </div>
  );
}

export default DataDownload;