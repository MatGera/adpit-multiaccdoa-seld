"use client";

import { useState } from "react";

export function TimelinePanel() {
  const [timeRange, setTimeRange] = useState<"1h" | "6h" | "24h" | "7d">("24h");

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-semibold">Event Timeline</h2>
        <div className="flex gap-1 bg-gray-800 rounded-lg p-1">
          {(["1h", "6h", "24h", "7d"] as const).map((range) => (
            <button
              key={range}
              onClick={() => setTimeRange(range)}
              className={`px-3 py-1 text-xs rounded-md transition-colors ${
                timeRange === range
                  ? "bg-brand-600 text-white"
                  : "text-gray-400 hover:text-white"
              }`}
            >
              {range}
            </button>
          ))}
        </div>
      </div>

      {/* Placeholder chart area */}
      <div className="bg-gray-900 rounded-lg border border-gray-800 p-6 h-64 flex items-center justify-center">
        <p className="text-gray-500 text-sm">
          Event frequency chart will render here using Recharts.
          <br />
          Selected range: {timeRange}
        </p>
      </div>

      {/* Event table */}
      <div className="bg-gray-900 rounded-lg border border-gray-800 overflow-hidden">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-gray-800">
              <th className="text-left p-3 text-gray-400 font-medium">Time</th>
              <th className="text-left p-3 text-gray-400 font-medium">Device</th>
              <th className="text-left p-3 text-gray-400 font-medium">Event</th>
              <th className="text-left p-3 text-gray-400 font-medium">Confidence</th>
              <th className="text-left p-3 text-gray-400 font-medium">Asset</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td colSpan={5} className="p-8 text-center text-gray-500">
                Connect to API to load historical events
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  );
}
