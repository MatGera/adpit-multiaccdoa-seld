import { z } from "zod";

// --- 3D Point ---
export const Point3DSchema = z.object({
  x: z.number(),
  y: z.number(),
  z: z.number(),
});
export type Point3D = z.infer<typeof Point3DSchema>;

// --- 4x4 Homogeneous Transformation Matrix (row-major) ---
export const TransformMatrixSchema = z.array(z.number()).length(16);
export type TransformMatrix = z.infer<typeof TransformMatrixSchema>;

// --- Raycast Request ---
export const RaycastRequestSchema = z.object({
  device_id: z.string(),
  direction: Point3DSchema,
  bim_model_id: z.string(),
  max_distance: z.number().positive().default(100.0),
});
export type RaycastRequest = z.infer<typeof RaycastRequestSchema>;

// --- Spatial Hit (single ray-BIM intersection) ---
export const SpatialHitSchema = z.object({
  asset_id: z.string(),
  asset_name: z.string(),
  ifc_type: z.string(),
  hit_point: Point3DSchema,
  distance: z.number(),
  confidence: z.number().min(0).max(1),
});
export type SpatialHit = z.infer<typeof SpatialHitSchema>;

// --- Raycast Response ---
export const RaycastResponseSchema = z.object({
  ray_origin: Point3DSchema,
  ray_direction: Point3DSchema,
  hits: z.array(SpatialHitSchema),
});
export type RaycastResponse = z.infer<typeof RaycastResponseSchema>;

// --- Triangulation Request ---
export const TriangulationRequestSchema = z.object({
  observations: z.array(
    z.object({
      device_id: z.string(),
      direction: Point3DSchema,
      confidence: z.number().min(0).max(1),
    })
  ).min(2),
  bim_model_id: z.string(),
});
export type TriangulationRequest = z.infer<typeof TriangulationRequestSchema>;

// --- Triangulation Result ---
export const TriangulationResultSchema = z.object({
  estimated_point: Point3DSchema,
  residual_error: z.number(),
  nearest_asset: SpatialHitSchema.nullable(),
  contributing_devices: z.array(z.string()),
});
export type TriangulationResult = z.infer<typeof TriangulationResultSchema>;

// --- Spatial Query (batch: N predictions -> ranked asset hits) ---
export const SpatialQueryRequestSchema = z.object({
  predictions: z.array(
    z.object({
      device_id: z.string(),
      class_name: z.string(),
      confidence: z.number(),
      vector: z.tuple([z.number(), z.number(), z.number()]),
    })
  ),
  bim_model_id: z.string(),
  top_k: z.number().int().positive().default(5),
});
export type SpatialQueryRequest = z.infer<typeof SpatialQueryRequestSchema>;

// --- Calibration Matrix ---
export const CalibrationMatrixSchema = z.object({
  device_id: z.string(),
  bim_model_id: z.string(),
  matrix: TransformMatrixSchema,
  updated_at: z.string().datetime(),
});
export type CalibrationMatrix = z.infer<typeof CalibrationMatrixSchema>;
