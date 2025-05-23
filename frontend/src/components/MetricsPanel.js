import React from 'react';

function MetricsPanel({ metrics }) {
  return (
    <div className="bg-gray-800 p-6 rounded-lg shadow-lg">
      <h2 className="text-xl font-semibold mb-4">Performance Metrics</h2>
      <div className="grid grid-cols-2 gap-4">
        <div>
          <p className="text-gray-400">Total Signals</p>
          <p className="text-2xl">{metrics.total_signals || 0}</p>
        </div>
        <div>
          <p className="text-gray-400">Win Rate</p>
          <p className="text-2xl">{(metrics.long_term_win_rate || 0).toFixed(1)}%</p>
        </div>
        <div>
          <p className="text-gray-400">Avg R:R</p>
          <p className="text-2xl">{(metrics.avg_rr || 0).toFixed(2)}</p>
        </div>
        <div>
          <p className="text-gray-400">Net Pips</p>
          <p className="text-2xl">{(metrics.total_pips || 0).toFixed(1)}</p>
        </div>
      </div>
    </div>
  );
}

export default MetricsPanel;