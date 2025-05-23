import React, { useState, useEffect } from 'react';
import { gsap } from 'gsap';

function TradePanel({ symbol, timeframe, apiBaseUrl }) {
  const [volume, setVolume] = useState('0.1');
  const [sl, setSl] = useState('0.0');
  const [tp, setTp] = useState('0.0');
  const [tradeStatus, setTradeStatus] = useState('Ready');
  const [latestSignal, setLatestSignal] = useState(null);

  useEffect(() => {
    const fetchLatestSignal = async () => {
      try {
        const response = await fetch(`${apiBaseUrl}/api/signals/recent?limit=1`);
        const data = await response.json();
        if (data.length > 0) {
          setLatestSignal(data[0]);
          setSl(data[0].stop_loss.toFixed(4));
          setTp(data[0].take_profit.toFixed(4));
        }
      } catch (error) {
        console.error('Error fetching latest signal:', error);
      }
    };
    fetchLatestSignal();
  }, [apiBaseUrl]);

  const executeTrade = async () => {
    if (!latestSignal) {
      alert('No signal available');
      return;
    }
    try {
      const priceData = await (await fetch(`${apiBaseUrl}/api/trades/price/${symbol}`)).json();
      const entry = latestSignal.trade_signal === 'BUY' ? priceData.ask : priceData.bid;
      const trade = {
        symbol,
        trade_type: latestSignal.trade_signal,
        volume: parseFloat(volume),
        entry,
        sl: parseFloat(sl),
        tp: parseFloat(tp),
        signal_type: latestSignal.signal_type
      };
      const response = await fetch(`${apiBaseUrl}/api/trades/execute`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(trade)
      });
      const result = await response.json();
      setTradeStatus(result.message);
      gsap.to('.trade-status', { color: '#27ae60', duration: 0.5 });
    } catch (error) {
      setTradeStatus('Trade failed');
      gsap.to('.trade-status', { color: '#c0392b', duration: 0.5 });
    }
  };

  return (
    <div className="bg-gray-800 p-6 rounded-lg shadow-lg">
      <h2 className="text-xl font-semibold mb-4">Execute Trade</h2>
      <div className="space-y-4">
        <div>
          <label className="block text-gray-400">Volume</label>
          <input
            type="number"
            value={volume}
            onChange={(e) => setVolume(e.target.value)}
            className="w-full bg-gray-700 text-white p-2 rounded"
          />
        </div>
        <div>
          <label className="block text-gray-400">Stop Loss</label>
          <input
            type="number"
            value={sl}
            onChange={(e) => setSl(e.target.value)}
            className="w-full bg-gray-700 text-white p-2 rounded"
          />
        </div>
        <div>
          <label className="block text-gray-400">Take Profit</label>
          <input
            type="number"
            value={tp}
            onChange={(e) => setTp(e.target.value)}
            className="w-full bg-gray-700 text-white p-2 rounded"
          />
        </div>
        <button
          onClick={executeTrade}
          className="w-full bg-blue-600 hover:bg-blue-700 text-white p-2 rounded"
        >
          Trade Now
        </button>
        <p className="trade-status text-center">{tradeStatus}</p>
      </div>
    </div>
  );
}

export default TradePanel;