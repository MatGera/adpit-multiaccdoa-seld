"use client";

import { useEffect, useState } from "react";
import { useLiveAlerts } from "@/hooks/use-live-alerts";

interface Alert {
  id: string;
  device_id: string;
  class_name: string;
  confidence: number;
  severity: "info" | "warning" | "critical" | "emergency";
  timestamp: string;
  asset_name?: string;
  message?: string;
}

export function AlertPanel() {
  const { alerts } = useLiveAlerts();

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-semibold">Live Alerts</h2>
        <div className="flex items-center gap-2 text-sm text-gray-400">
          <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
          Streaming
        </div>
      </div>

      {/* Alert filters */}
      <div className="flex gap-2">
        {(["all", "info", "warning", "critical", "emergency"] as const).map(
          (level) => (
            <button
              key={level}
              className="px-3 py-1 text-xs rounded-full bg-gray-800 text-gray-400 hover:text-white hover:bg-gray-700 transition-colors capitalize"
            >
              {level}
            </button>
          )
        )}
      </div>

      {/* Alert list */}
      <div className="space-y-2">
        {alerts.length === 0 ? (
          <div className="text-center py-12 text-gray-500">
            <p className="text-lg">No alerts</p>
            <p className="text-sm mt-1">
              Alerts will appear here as acoustic events are detected
            </p>
          </div>
        ) : (
          alerts.map((alert) => <AlertCard key={alert.id} alert={alert} />)
        )}
      </div>
    </div>
  );
}

function AlertCard({ alert }: { alert: Alert }) {
  const severityColors: Record<string, string> = {
    info: "border-blue-500/30 bg-blue-500/5",
    warning: "border-amber-500/30 bg-amber-500/5",
    critical: "border-red-500/30 bg-red-500/5",
    emergency: "border-red-700/50 bg-red-700/10",
  };

  return (
    <div
      className={`rounded-lg border p-4 ${severityColors[alert.severity] ?? "border-gray-800"}`}
    >
      <div className="flex items-start justify-between">
        <div>
          <div className="flex items-center gap-2">
            <span className={`severity-badge severity-${alert.severity}`}>
              {alert.severity.toUpperCase()}
            </span>
            <span className="text-sm font-medium text-white">
              {alert.class_name}
            </span>
          </div>
          <p className="text-xs text-gray-400 mt-1">
            Device: {alert.device_id}
            {alert.asset_name && ` · Asset: ${alert.asset_name}`}
          </p>
        </div>
        <div className="text-right">
          <p className="text-xs text-gray-500">
            {new Date(alert.timestamp).toLocaleTimeString()}
          </p>
          <p className="text-xs text-gray-600 mt-0.5">
            {(alert.confidence * 100).toFixed(0)}% confidence
          </p>
        </div>
      </div>
      {alert.message && (
        <p className="text-sm text-gray-300 mt-2">{alert.message}</p>
      )}
    </div>
  );
}
