import React, { useState } from 'react';

function ForecastPanel({ symbol, timeframe }) {
  const [model, setModel] = useState('LSTM');
  const [period, setPeriod] = useState('5 Periods');

  const runForecast = () => {
    // Placeholder for forecast API call
    alert(`Running ${model} forecast for ${symbol} - ${timeframe} - ${period}`);
  };

  return (
    <div className="bg-gray-800 p-6 rounded-lg shadow-lg">
      <h2 className="text-xl font-semibold mb-4">Forecasting</h2>
      <div className="space-y-4">
        <div>
          <label className="block text-gray-400">Forecasting Model</label>
          <select
            value={model}
            onChange={(e) => setModel(e.target.value)}
            className="w-full bg-gray-700 text-white p-2 rounded"
          >
            {['ARIMA', 'LSTM', 'SVR', 'Facebook Prophet', 'Ensemble Model'].map(m => (
              <option key={m} value={m}>{m}</option>
            ))}
          </select>
        </div>
        <div>
          <label className="block text-gray-400">Period</label>
          <select
            value={period}
            onChange={(e) => setPeriod(e.target.value)}
            className="w-full bg-gray-700 text-white p-2 rounded"
          >
            {['1 Period', '3 Periods', '5 Periods', '10 Periods', '24 Hours'].map(p => (
              <option key={p} value={p}>{p}</option>
            ))}
          </select>
        </div>
        <button
          onClick={runForecast}
          className="w-full bg-blue-600 hover:bg-blue-700 text-white p-2 rounded"
        >
          Run Forecast
        </button>
      </div>
    </div>
  );
}

export default ForecastPanel;