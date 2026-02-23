"use client";

import { useEffect, useRef, useState, useCallback } from "react";

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

const MAX_ALERTS = 100;

/**
 * Hook to receive real-time alerts via WebSocket.
 * Connects to the API Gateway WebSocket endpoint for live prediction streaming.
 */
export function useLiveAlerts() {
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [connected, setConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);

  const connect = useCallback(() => {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.host}/api/v1/predictions/live`;

    try {
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        setConnected(true);
        console.log("[WS] Connected to live prediction feed");
      };

      ws.onmessage = (event) => {
        try {
          const prediction = JSON.parse(event.data);

          // Transform prediction into alert format
          const alert: Alert = {
            id: `${prediction.device_id}-${prediction.frame_idx}-${Date.now()}`,
            device_id: prediction.device_id,
            class_name: prediction.class ?? prediction.class_name ?? "unknown",
            confidence: prediction.confidence ?? 0,
            severity: classifyServerity(prediction.confidence ?? 0),
            timestamp: prediction.timestamp ?? new Date().toISOString(),
            asset_name: prediction.asset_name,
          };

          setAlerts((prev) => [alert, ...prev].slice(0, MAX_ALERTS));
        } catch {
          console.warn("[WS] Failed to parse message", event.data);
        }
      };

      ws.onclose = () => {
        setConnected(false);
        console.log("[WS] Disconnected, reconnecting in 3s...");
        setTimeout(connect, 3000);
      };

      ws.onerror = (err) => {
        console.error("[WS] Error", err);
        ws.close();
      };
    } catch {
      setTimeout(connect, 5000);
    }
  }, []);

  useEffect(() => {
    connect();
    return () => {
      wsRef.current?.close();
    };
  }, [connect]);

  return { alerts, connected };
}

function classifyServerity(
  confidence: number
): "info" | "warning" | "critical" | "emergency" {
  if (confidence >= 0.9) return "emergency";
  if (confidence >= 0.75) return "critical";
  if (confidence >= 0.5) return "warning";
  return "info";
}
