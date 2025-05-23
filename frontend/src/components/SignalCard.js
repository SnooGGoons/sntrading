import React, { useEffect } from 'react';
import { gsap } from 'gsap';

function SignalCard({ signals }) {
  useEffect(() => {
    gsap.from('.signal-item', {
      y: 20,
      opacity: 0,
      stagger: 0.2,
      duration: 0.5,
      ease: 'power2.out'
    });
  }, [signals]);

  return (
    <div className="signal-card bg-gray-800 p-6 rounded-lg shadow-lg">
      <h2 className="text-xl font-semibold mb-4">Trading Signals</h2>
      {signals.length === 0 ? (
        <p className="text-gray-400 pulse">Monitoring for trading opportunities...</p>
      ) : (
        signals.map((signal) => (
          <div
            key={signal.id}
            className={`signal-item p-4 mb-2 rounded-lg ${
              signal.trade_signal === 'BUY' ? 'bg-green-900' : 'bg-red-900'
            }`}
          >
            <p className="font-bold">{signal.symbol} - {signal.timeframe}</p>
            <p>Signal: {signal.trade_signal} ({signal.signal_type})</p>
            <p>Entry: {signal.entry_level.toFixed(4)}</p>
            <p>Stop Loss: {signal.stop_loss.toFixed(4)}</p>
            <p>Take Profit: {signal.take_profit.toFixed(4)}</p>
            <p>Confidence: {signal.confidence.toFixed(1)}%</p>
            <p>Time: {new Date(signal.timestamp).toLocaleString()}</p>
          </div>
        ))
      )}
    </div>
  );
}

export default SignalCard;