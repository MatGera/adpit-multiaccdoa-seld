"use client";

import dynamic from "next/dynamic";
import { Suspense, useState } from "react";

// Dynamically import Three.js canvas to avoid SSR issues
const BIMScene = dynamic(() => import("./bim-scene").then((mod) => mod.BIMScene), {
  ssr: false,
  loading: () => (
    <div className="flex items-center justify-center h-full bg-gray-900 rounded-lg">
      <div className="text-gray-500">Loading 3D viewer...</div>
    </div>
  ),
});

export function BIMViewerPanel() {
  const [selectedAsset, setSelectedAsset] = useState<string | null>(null);

  return (
    <div className="h-full flex gap-4">
      {/* 3D Viewer */}
      <div className="flex-1 rounded-lg overflow-hidden border border-gray-800 bg-gray-900">
        <Suspense fallback={<div className="p-8 text-gray-500">Loading...</div>}>
          <BIMScene onAssetSelect={setSelectedAsset} />
        </Suspense>
      </div>

      {/* Asset info sidebar */}
      <div className="w-80 bg-gray-900 rounded-lg border border-gray-800 p-4 overflow-y-auto">
        <h3 className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-4">
          Asset Details
        </h3>

        {selectedAsset ? (
          <div className="space-y-4">
            <div>
              <label className="text-xs text-gray-500">Asset ID</label>
              <p className="text-sm text-white font-mono">{selectedAsset}</p>
            </div>
            <div>
              <label className="text-xs text-gray-500">Status</label>
              <p className="text-sm">
                <span className="severity-badge severity-info">Normal</span>
              </p>
            </div>
            <div>
              <label className="text-xs text-gray-500">Recent Events</label>
              <p className="text-sm text-gray-400">No recent acoustic events</p>
            </div>
          </div>
        ) : (
          <p className="text-sm text-gray-500">
            Select an asset in the 3D viewer to see details
          </p>
        )}
      </div>
    </div>
  );
}
