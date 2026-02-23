import { z } from "zod";

// --- DOA Vector ---
export const DOAVectorSchema = z.tuple([z.number(), z.number(), z.number()]);
export type DOAVector = z.infer<typeof DOAVectorSchema>;

// --- Single Prediction ---
export const PredictionItemSchema = z.object({
  class: z.string(),
  confidence: z.number().min(0).max(1),
  vector: DOAVectorSchema,
});
export type PredictionItem = z.infer<typeof PredictionItemSchema>;

// --- Device Telemetry ---
export const TelemetrySchema = z.object({
  inference_ms: z.number(),
  cpu_temp: z.number(),
  gpu_temp: z.number(),
  mem_used_mb: z.number(),
});
export type Telemetry = z.infer<typeof TelemetrySchema>;

// --- Edge Prediction Payload (MQTT JSON) ---
export const EdgePredictionPayloadSchema = z.object({
  device_id: z.string(),
  timestamp: z.string().datetime(),
  frame_idx: z.number().int().nonnegative(),
  predictions: z.array(PredictionItemSchema),
  telemetry: TelemetrySchema.optional(),
});
export type EdgePredictionPayload = z.infer<typeof EdgePredictionPayloadSchema>;

// --- Historical Prediction Query ---
export const PredictionQuerySchema = z.object({
  device_id: z.string().optional(),
  class_name: z.string().optional(),
  start_time: z.string().datetime().optional(),
  end_time: z.string().datetime().optional(),
  min_confidence: z.number().min(0).max(1).optional(),
  limit: z.number().int().positive().max(10000).default(100),
  offset: z.number().int().nonnegative().default(0),
});
export type PredictionQuery = z.infer<typeof PredictionQuerySchema>;
