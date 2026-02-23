"use client";

import { useQuery } from "@tanstack/react-query";

interface Device {
  device_id: string;
  name: string;
  status: string;
  hardware_type: string;
  model_version: string | null;
  cpu_temp?: number;
  gpu_temp?: number;
  inference_latency_ms?: number;
}

export function DeviceFleetPanel() {
  const { data, isLoading } = useQuery<{ devices: Device[] }>({
    queryKey: ["devices"],
    queryFn: () =>
      fetch("/api/v1/devices").then((r) => r.json()),
    retry: false,
  });

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-semibold">Device Fleet</h2>
        <button className="px-4 py-2 bg-brand-600 text-white text-sm rounded-lg hover:bg-brand-700 transition-colors">
          + Register Device
        </button>
      </div>

      {/* Device grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {isLoading ? (
          Array.from({ length: 3 }).map((_, i) => (
            <div
              key={i}
              className="bg-gray-900 rounded-lg border border-gray-800 p-4 animate-pulse h-48"
            />
          ))
        ) : data?.devices && data.devices.length > 0 ? (
          data.devices.map((device) => (
            <DeviceCard key={device.device_id} device={device} />
          ))
        ) : (
          <div className="col-span-full bg-gray-900 rounded-lg border border-gray-800 p-12 text-center">
            <p className="text-gray-500">
              No devices registered. Connect to the API Gateway to manage devices.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

function DeviceCard({ device }: { device: Device }) {
  const statusColors: Record<string, string> = {
    online: "bg-green-500",
    offline: "bg-gray-500",
    degraded: "bg-amber-500",
    maintenance: "bg-blue-500",
    provisioning: "bg-purple-500",
  };

  return (
    <div className="bg-gray-900 rounded-lg border border-gray-800 p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="font-medium text-white truncate">{device.name}</h3>
        <div className="flex items-center gap-1.5">
          <div
            className={`w-2 h-2 rounded-full ${statusColors[device.status] ?? "bg-gray-500"}`}
          />
          <span className="text-xs text-gray-400 capitalize">{device.status}</span>
        </div>
      </div>

      <div className="space-y-2 text-sm">
        <div className="flex justify-between">
          <span className="text-gray-500">ID</span>
          <span className="text-gray-300 font-mono text-xs">{device.device_id}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-gray-500">Type</span>
          <span className="text-gray-300">{device.hardware_type}</span>
        </div>
        {device.model_version && (
          <div className="flex justify-between">
            <span className="text-gray-500">Model</span>
            <span className="text-gray-300">{device.model_version}</span>
          </div>
        )}
        {device.cpu_temp !== undefined && (
          <div className="flex justify-between">
            <span className="text-gray-500">CPU Temp</span>
            <span className="text-gray-300">{device.cpu_temp}°C</span>
          </div>
        )}
        {device.inference_latency_ms !== undefined && (
          <div className="flex justify-between">
            <span className="text-gray-500">Latency</span>
            <span className="text-gray-300">{device.inference_latency_ms}ms</span>
          </div>
        )}
      </div>
    </div>
  );
}
