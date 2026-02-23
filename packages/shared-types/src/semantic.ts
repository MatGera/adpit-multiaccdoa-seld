import { z } from "zod";
import { SpatialHitSchema } from "./spatial.js";
import { PredictionItemSchema } from "./prediction.js";

// --- RAG Context (assembled before LLM call) ---
export const RAGContextSchema = z.object({
  seld_predictions: z.array(PredictionItemSchema),
  spatial_hit: SpatialHitSchema.nullable(),
  asset_metadata: z
    .object({
      asset_id: z.string(),
      asset_name: z.string(),
      asset_type: z.string(),
      model: z.string().optional(),
      manufacturer: z.string().optional(),
      install_date: z.string().optional(),
      last_maintenance: z.string().optional(),
      maintenance_history: z
        .array(
          z.object({
            date: z.string(),
            type: z.string(),
            description: z.string(),
          })
        )
        .optional(),
    })
    .nullable(),
  retrieved_chunks: z.array(
    z.object({
      content: z.string(),
      source: z.string(),
      page: z.number().optional(),
      score: z.number(),
    })
  ),
});
export type RAGContext = z.infer<typeof RAGContextSchema>;

// --- Semantic Query Request ---
export const SemanticQueryRequestSchema = z.object({
  device_id: z.string(),
  class_name: z.string(),
  confidence: z.number(),
  vector: z.tuple([z.number(), z.number(), z.number()]),
  bim_model_id: z.string(),
  additional_context: z.string().optional(),
});
export type SemanticQueryRequest = z.infer<typeof SemanticQueryRequestSchema>;

// --- LLM Response ---
export const LLMResponseSchema = z.object({
  response_text: z.string(),
  citations: z.array(
    z.object({
      source: z.string(),
      page: z.number().optional(),
      quote: z.string(),
    })
  ),
  confidence: z.number().min(0).max(1),
  asset_id: z.string().nullable(),
  recommended_actions: z.array(z.string()),
  severity: z.enum(["info", "warning", "critical", "emergency"]),
});
export type LLMResponse = z.infer<typeof LLMResponseSchema>;

// --- Document Ingestion Request ---
export const DocumentIngestionSchema = z.object({
  file_name: z.string(),
  file_type: z.enum(["pdf", "docx", "txt"]),
  asset_tags: z.array(z.string()).optional(),
  metadata: z.record(z.string()).optional(),
});
export type DocumentIngestion = z.infer<typeof DocumentIngestionSchema>;
