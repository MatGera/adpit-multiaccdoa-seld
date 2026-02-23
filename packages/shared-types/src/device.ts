import { z } from "zod";

// --- Device Status ---
export const DeviceStatusEnum = z.enum([
  "online",
  "offline",
  "degraded",
  "maintenance",
  "provisioning",
]);
export type DeviceStatus = z.infer<typeof DeviceStatusEnum>;

// --- Device Hardware Type ---
export const HardwareTypeEnum = z.enum([
  "industrial_capacitive",
  "infrastructure_piezoelectric",
]);
export type HardwareType = z.infer<typeof HardwareTypeEnum>;

// --- Device Configuration ---
export const DeviceConfigSchema = z.object({
  device_id: z.string(),
  name: z.string(),
  hardware_type: HardwareTypeEnum,
  num_channels: z.number().int().refine((n) => n === 4 || n === 8, {
    message: "num_channels must be 4 (tetrahedral) or 8 (spherical)",
  }),
  sample_rate: z.number().int().default(48000),
  frame_length_ms: z.number().int().default(100),
  confidence_threshold: z.number().min(0).max(1).default(0.5),
  model_version: z.string().optional(),
  location: z
    .object({
      building: z.string().optional(),
      floor: z.string().optional(),
      zone: z.string().optional(),
      coordinates: z
        .object({
          x: z.number(),
          y: z.number(),
          z: z.number(),
        })
        .optional(),
    })
    .optional(),
  mqtt_topic_prefix: z.string().default("dt/edge"),
  ota_enabled: z.boolean().default(true),
});
export type DeviceConfig = z.infer<typeof DeviceConfigSchema>;

// --- Device Registration Request ---
export const DeviceRegistrationSchema = z.object({
  device_id: z.string().min(1).max(64),
  name: z.string().min(1).max(255),
  hardware_type: HardwareTypeEnum,
  num_channels: z.number().int(),
  location: DeviceConfigSchema.shape.location,
});
export type DeviceRegistration = z.infer<typeof DeviceRegistrationSchema>;

// --- Device Health ---
export const DeviceHealthSchema = z.object({
  device_id: z.string(),
  status: DeviceStatusEnum,
  last_seen: z.string().datetime().nullable(),
  uptime_seconds: z.number().nonnegative(),
  cpu_temp: z.number(),
  gpu_temp: z.number(),
  mem_used_mb: z.number(),
  mem_total_mb: z.number(),
  inference_latency_ms: z.number(),
  predictions_per_minute: z.number(),
  model_version: z.string().nullable(),
  firmware_version: z.string().nullable(),
});
export type DeviceHealth = z.infer<typeof DeviceHealthSchema>;
