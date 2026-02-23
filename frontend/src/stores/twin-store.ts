import { create } from "zustand";

interface Prediction {
  device_id: string;
  timestamp: string;
  frame_idx: number;
  class_name: string;
  confidence: number;
  vector: [number, number, number];
}

interface TwinState {
  // Selected elements
  selectedAssetId: string | null;
  selectedDeviceId: string | null;

  // Live prediction data
  latestPredictions: Map<string, Prediction>;

  // BIM model state
  loadedModelId: string | null;

  // Actions
  selectAsset: (id: string | null) => void;
  selectDevice: (id: string | null) => void;
  updatePrediction: (deviceId: string, prediction: Prediction) => void;
  setLoadedModel: (modelId: string | null) => void;
}

export const useTwinStore = create<TwinState>((set) => ({
  selectedAssetId: null,
  selectedDeviceId: null,
  latestPredictions: new Map(),
  loadedModelId: null,

  selectAsset: (id) => set({ selectedAssetId: id }),
  selectDevice: (id) => set({ selectedDeviceId: id }),
  updatePrediction: (deviceId, prediction) =>
    set((state) => {
      const newMap = new Map(state.latestPredictions);
      newMap.set(deviceId, prediction);
      return { latestPredictions: newMap };
    }),
  setLoadedModel: (modelId) => set({ loadedModelId: modelId }),
}));
