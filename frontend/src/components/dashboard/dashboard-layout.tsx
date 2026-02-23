"use client";

import { useState } from "react";
import { Sidebar } from "./sidebar";
import { BIMViewerPanel } from "../viewer/bim-viewer-panel";
import { AlertPanel } from "../alerts/alert-panel";
import { TimelinePanel } from "../timeline/timeline-panel";
import { DeviceFleetPanel } from "../devices/device-fleet-panel";

type Tab = "viewer" | "alerts" | "timeline" | "devices";

export function DashboardLayout() {
  const [activeTab, setActiveTab] = useState<Tab>("viewer");

  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar activeTab={activeTab} onTabChange={setActiveTab} />

      <main className="flex-1 flex flex-col overflow-hidden">
        {/* Header */}
        <header className="h-14 border-b border-gray-800 flex items-center px-6 shrink-0">
          <h1 className="text-lg font-semibold text-white">
            SELD Digital Twin
          </h1>
          <span className="ml-3 text-xs text-gray-500 bg-gray-800 px-2 py-0.5 rounded">
            OPERATOR DASHBOARD
          </span>
          <div className="ml-auto flex items-center gap-4">
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
              <span className="text-sm text-gray-400">System Online</span>
            </div>
          </div>
        </header>

        {/* Content */}
        <div className="flex-1 overflow-auto p-4">
          {activeTab === "viewer" && <BIMViewerPanel />}
          {activeTab === "alerts" && <AlertPanel />}
          {activeTab === "timeline" && <TimelinePanel />}
          {activeTab === "devices" && <DeviceFleetPanel />}
        </div>
      </main>
    </div>
  );
}
