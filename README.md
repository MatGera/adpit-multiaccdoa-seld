# Semantic Acoustic Digital Twin

**Industrial cyber-physical system for condition monitoring and infrastructure security via acoustic anomaly detection.**

A complete Late Fusion architecture: edge devices (NVIDIA Jetson Orin) run SELD inference locally on MEMS microphone arrays, transmitting only lightweight JSON prediction payloads to the cloud — where a BIM-based spatial engine triangulates fault locations and a prescriptive LLM layer generates actionable maintenance reports.

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│  EDGE (Jetson Orin)                    CLOUD                                     │
│                                                                                  │
│  MEMS Array (FOA)                      EMQX Broker                               │
│       │                                     │                                    │
│  PDM→PCM→Features             ┌────────────▼────────────┐                       │
│  Log-Mel + SALSA + IV          │   Kafka Event Bus        │                      │
│       │                        └─────┬──────────┬────────┘                      │
│  ResNet-Conformer                    │          │                                │
│  Multi-ACCDOA                  ┌─────▼──┐  ┌───▼──────┐                        │
│  (ADPIT loss)                  │Ingest  │  │TimescaleDB│                        │
│       │                        │Service │  │+ pgvector │                        │
│  JSON payload ──MQTT──────────►└────────┘  └──────┬───┘                        │
│  {DOA, class, conf}                               │                              │
│  < 10 Kbps/node                           ┌───────▼────────┐                   │
│                                            │ Spatial Engine │                   │
│  Audio destroyed                          │  IFC + Raycast  │                   │
│  < 500 ms (GDPR)                          │  Triangulation  │                   │
│                                            └───────┬────────┘                   │
│                                                    │                             │
│                                            ┌───────▼────────┐                   │
│                                            │ Semantic Layer │                   │
│                                            │  RAG + LLM     │                   │
│                                            │  (Claude/Llama)│                   │
│                                            └───────┬────────┘                   │
│                                                    │                             │
│                                            ┌───────▼────────┐                   │
│                                            │   Next.js 15   │                   │
│                                            │ Three.js Viewer│                   │
│                                            │  BIM + DOA     │                   │
│                                            └────────────────┘                   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Repository Structure](#repository-structure)
3. [Prerequisites](#prerequisites)
4. [Getting Started](#getting-started)
5. [Services](#services)
   - [Edge Agent](#edge-agent)
   - [ML Training Pipeline](#ml-training-pipeline)
   - [API Gateway](#api-gateway)
   - [Ingestion Service](#ingestion-service)
   - [Device Service](#device-service)
   - [Spatial Engine](#spatial-engine)
   - [Semantic Layer](#semantic-layer)
   - [Vision Fusion](#vision-fusion)
   - [Frontend Dashboard](#frontend-dashboard)
6. [Infrastructure](#infrastructure)
7. [Database Schema](#database-schema)
8. [Configuration Reference](#configuration-reference)
9. [Development Workflow](#development-workflow)
10. [Testing](#testing)
11. [Deployment](#deployment)
12. [MQTT Topics & Payload Schema](#mqtt-topics--payload-schema)
13. [API Reference](#api-reference)
14. [Security & GDPR](#security--gdpr)
15. [Technical Decisions](#technical-decisions)
16. [Contributing](#contributing)
17. [License](#license)

---

## Architecture Overview

### Core Paradigm: Late Fusion

Raw audio **never leaves** the edge device. Each Jetson Orin node performs full acoustic inference locally, transmitting only a compact JSON payload containing:

- Sound event class (e.g. `"valve_leak"`)
- Confidence score (derived from the ACCDOA vector norm)
- 3D Cartesian DOA (Direction of Arrival) vector in the device's local frame

This design guarantees:
- **Privacy by design** (GDPR compliant — no remote audio surveillance of workers)
- **Bandwidth efficiency** (`< 10 Kbps` per node vs. `> 1 Mbps` for raw audio)
- **Edge resilience** (inference continues if cloud connectivity drops)

### AI Core: Multi-ACCDOA + ADPIT

The SELD model outputs in **Multi-ACCDOA** format — a tensor of shape `(C_classes × T_tracks × 3)` where each class maintains 3 independent tracks with Cartesian DOA vectors. The vector norm encodes activity confidence, solving the *polyphony collapse* problem of single-ACCDOA.

Training uses **ADPIT** (Auxiliary Duplicating Permutation Invariant Training):
- PIT (Permutation Invariant Training) over active tracks resolves track assignment ambiguity
- An auxiliary suppression loss forces unassigned tracks toward zero, eliminating false-positive ghost vectors

### Spatial Resolution

A calibrated **roto-translation matrix** `M ∈ R^(4×4)` per device maps local DOA vectors into absolute BIM world coordinates. Rays from multiple independent nodes are triangulated via SVD-based least-squares closest-point, unambiguously pinpointing the faulty asset in 3D space.

---

## Repository Structure

```
adpit-multiaccdoa-seld/
│
├── LICENSE                          # Apache 2.0
├── README.md
├── .gitignore
│
├── pnpm-workspace.yaml              # TypeScript monorepo workspaces
├── package.json                     # Root: scripts, Turborepo
├── turbo.json                       # Turborepo task graph
├── pyproject.toml                   # Python workspace (uv)
│
├── docker-compose.yml               # Full local dev stack
├── docker-compose.edge.yml          # Edge simulation (NanoMQ + mock sensor)
│
├── .github/
│   └── workflows/
│       ├── ci-edge.yml              # Edge: lint, test, aarch64 Docker build
│       ├── ci-cloud.yml             # Cloud services: lint, test, Docker push
│       ├── ci-frontend.yml          # Next.js: build, Playwright tests
│       ├── ci-ml-training.yml       # Training: model smoke tests, ONNX export
│       └── cd-deploy.yml            # Deploy to staging/production via Helm
│
├── .cursor/rules/
│   ├── python-style.mdc             # Python conventions for Cursor AI
│   ├── typescript-style.mdc         # TypeScript conventions
│   └── project-structure.mdc        # Monorepo navigation guide
│
├── packages/                        # Shared TypeScript packages
│   ├── shared-types/                # Zod schemas + TS types (all services)
│   │   └── src/
│   │       ├── prediction.ts        # EdgePredictionPayload, PredictionItem
│   │       ├── device.ts            # DeviceConfig, DeviceHealth
│   │       ├── spatial.ts           # SpatialHit, TriangulationResult
│   │       └── semantic.ts          # RAGContext, LLMResponse
│   └── proto/                       # Protobuf / gRPC definitions
│       └── src/
│           ├── prediction.proto     # PredictionService
│           ├── device.proto         # DeviceRegistryService
│           ├── spatial.proto        # SpatialEngineService
│           └── semantic.proto       # SemanticLayerService
│
├── libs/
│   └── seld-common/                 # Shared Python library
│       └── src/seld_common/
│           ├── schemas.py           # Pydantic models (mirrors shared-types)
│           ├── audio_utils.py       # FOA encoding, mel, intensity vectors
│           ├── mqtt_client.py       # Async MQTT 5.0 client (gmqtt)
│           └── db.py                # SQLAlchemy async session factory
│
├── edge/                            # Edge agent (Jetson Orin)
│   ├── Dockerfile                   # Production: L4T + JetPack 6.x
│   ├── Dockerfile.dev               # Development: Python 3.12 slim
│   ├── nanomq.conf                  # NanoMQ broker + QUIC bridge config
│   └── src/edge_agent/
│       ├── audio/                   # PDM capture, CIC decimation, FOA encoder
│       ├── features/                # Log-Mel, SALSA, Intensity Vectors
│       ├── inference/               # TensorRT engine, Multi-ACCDOA decoder
│       ├── transport/               # MQTT publisher, HTTPS client
│       └── device/                  # Health monitor, OTA updater
│
├── training/                        # ML Training (cloud GPU)
│   ├── Dockerfile                   # CUDA 12.x + PyTorch 2.x
│   └── src/seld_training/
│       ├── model/                   # ResNet-Conformer, SE blocks, ACCDOA head
│       ├── loss/                    # ADPIT loss (PIT + auxiliary suppression)
│       ├── data/                    # STARSS23 dataset, 4-stage augmentation
│       ├── evaluation/              # SELD metrics (F1, LE, LR, SELD-error)
│       └── export/                  # ONNX (opset 17) → TensorRT FP16/INT8
│
├── cloud/
│   ├── api-gateway/                 # FastAPI REST + WebSocket gateway
│   ├── ingestion-service/           # Kafka consumer → TimescaleDB
│   └── device-service/              # gRPC device registry + mTLS certs
│
├── spatial/                         # Spatial Engine (IFC → raycasting)
│   └── src/spatial_engine/
│       ├── ifc_parser.py            # IfcOpenShell + trimesh mesh extraction
│       ├── roto_translation.py      # 4×4 homogeneous transform utilities
│       ├── raycaster.py             # BVH ray-mesh intersection
│       ├── triangulation.py         # SVD least-squares multi-sensor solver
│       └── grpc_servicer.py         # gRPC SpatialService implementation
│
├── semantic/                        # Semantic Layer (RAG + LLM)
│   └── src/semantic_layer/
│       ├── embeddings.py            # sentence-transformers embedding service
│       ├── document_ingestion.py    # PDF/DOCX → chunk → embed → pgvector
│       ├── vector_search.py         # pgvector cosine + BM25 + RRF fusion
│       ├── llm_orchestrator.py      # Anthropic Claude / Ollama streaming
│       ├── prompt_templates.py      # Jinja2 prescriptive templates
│       └── guardrails.py            # PII filter, off-topic detection
│
├── vision/                          # Vision Fusion (CCTV + mobile mapping)
│   └── src/vision_fusion/
│       ├── cctv_pipeline.py         # YOLO11 + BoT-SORT multi-camera
│       ├── tracker.py               # Track lifecycle management
│       ├── homography.py            # Pixel → BIM ground-plane mapping
│       ├── state_cache.py           # Redis TTL-decay ghost map
│       └── mobile_mapping.py        # LiDAR point cloud → BIM-Lite
│
├── frontend/                        # Operator Dashboard (Next.js 15)
│   └── src/
│       ├── app/                     # Next.js App Router pages
│       ├── components/
│       │   ├── viewer/              # Three.js BIM canvas + DOA arrows
│       │   ├── alerts/              # Live alert panel + LLM streaming
│       │   ├── timeline/            # Historical event replay
│       │   └── devices/             # Fleet management grid
│       ├── hooks/                   # useLiveAlerts (WebSocket)
│       └── stores/                  # Zustand twin state
│
├── infra/                           # Pulumi IaC (TypeScript)
│   └── src/k8s/
│       ├── postgres.ts              # PG 16 StatefulSet + PVC
│       ├── emqx.ts                  # EMQX MQTT broker Deployment
│       ├── kafka.ts                 # Redpanda StatefulSet
│       ├── services.ts              # All application Deployments
│       ├── monitoring.ts            # Prometheus, Grafana, Loki, Jaeger
│       └── ingress.ts               # nginx Ingress rules
│
├── migrations/                      # Alembic database migrations
│   └── versions/
│       ├── 001_init_extensions.py   # pgvector, TimescaleDB, pg_trgm
│       ├── 002_device_registry.py   # devices, device_certificates tables
│       ├── 003_prediction_hypertable.py # TimescaleDB hypertable + aggregates
│       ├── 004_bim_models.py        # bim_models, bim_assets tables
│       ├── 005_calibration_matrices.py # Per-device 4×4 transform matrices
│       ├── 006_document_embeddings.py  # pgvector HNSW + tsvector FTS index
│       └── 007_spatial_hits.py      # Spatial hits hypertable + LLM log
│
├── scripts/
│   ├── setup-dev.sh                 # One-command dev environment setup
│   ├── seed-db.sh                   # Load test devices + BIM models
│   ├── generate-protos.sh           # Compile .proto → Python + TypeScript
│   └── build-tensorrt.sh            # Build TRT engine on Jetson
│
└── tests/
    ├── unit/                        # Pure unit tests (no services needed)
    │   ├── test_schemas.py          # Pydantic model validation
    │   ├── test_roto_translation.py # Geometric transform math
    │   ├── test_triangulation.py    # Multi-sensor DOA triangulation
    │   ├── test_guardrails.py       # LLM safety guardrails
    │   └── test_homography.py       # Camera-to-BIM pixel mapping
    ├── integration/                 # Requires running services
    │   ├── test_api_gateway.py      # API endpoint contract tests
    │   └── test_e2e_smoke.py        # Full pipeline smoke test
    ├── load/
    │   └── locustfile.py            # 1000+ concurrent node simulation
    └── security/
        └── test_auth.py             # CORS, SQL injection, auth validation
```

---

## Prerequisites

### Local Development

| Tool | Minimum version | Purpose |
|------|----------------|---------|
| Docker + Docker Compose | 24.x | Run all infrastructure services |
| Node.js | 22 LTS | TypeScript services, frontend |
| pnpm | 9.x | TypeScript package manager |
| Python | 3.12+ | All Python services |
| [uv](https://docs.astral.sh/uv/) | 0.5+ | Python package manager (fast) |

**Install uv:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Install pnpm:**
```bash
npm install -g pnpm@9
```

### Edge Device (Jetson Orin)

| Component | Version |
|-----------|---------|
| JetPack | 6.x |
| TensorRT | 10.x (bundled with JetPack) |
| CUDA | 12.x (Orin compute capability 8.7) |
| NanoMQ | 0.21+ |
| Python | 3.12 |

### Optional (Training)

| Tool | Purpose |
|------|---------|
| CUDA GPU (A100/H100 recommended) | SELD model training |
| Weights & Biases account | Experiment tracking |
| Anthropic API key | Claude LLM for prescriptive layer |

---

## Getting Started

### 1. Clone and initialize

```bash
git clone https://github.com/your-org/adpit-multiaccdoa-seld.git
cd adpit-multiaccdoa-seld
```

### 2. Run the setup script

This installs all dependencies, starts infrastructure, and runs database migrations:

```bash
bash scripts/setup-dev.sh
```

Or manually:

```bash
# Install TypeScript dependencies
pnpm install

# Install Python dependencies for all services
for svc in libs/seld-common edge training cloud/api-gateway \
            cloud/ingestion-service cloud/device-service \
            spatial semantic vision; do
    (cd $svc && uv sync --dev)
done

# Start infrastructure (PostgreSQL, Redis, EMQX, Redpanda)
docker compose up -d postgres redis emqx redpanda

# Wait for PostgreSQL to be ready
until docker compose exec postgres pg_isready -U seld; do sleep 1; done

# Run database migrations
cd migrations && uv run alembic upgrade head && cd ..

# Seed test data
bash scripts/seed-db.sh
```

### 3. Start all cloud services

```bash
docker compose up -d
```

### 4. Start the frontend

```bash
cd frontend
pnpm dev
```

Dashboard available at **http://localhost:3000**

### 5. Simulate an edge device

```bash
docker compose -f docker-compose.edge.yml up
```

This starts a mock edge agent that publishes synthetic predictions to NanoMQ, which bridges to EMQX.

---

## Services

### Edge Agent

**Path:** `edge/`  
**Runtime:** Jetson Orin (JetPack 6.x), Python 3.12  
**Container:** `seld-edge-agent` (aarch64)

The edge agent orchestrates a real-time audio processing pipeline with a strict **< 500 ms total latency budget** per frame, ensuring audio buffers are destroyed well within that window.

#### Pipeline

```
PDM Capture (ALSA)
    ↓ 100 ms audio frame
CIC Decimation Filter (PDM → PCM, R=64, N=4 stages)
    ↓
FOA Encoder (raw 4-ch → B-format W,X,Y,Z)
    ↓
Feature Extraction:
  • Log-Mel spectrogram (4 ch × T × 128 bins)
  • Intensity Vectors    (3 ch × T × F)
  • SALSA               (3 ch × T × F) — normalized inter-channel phase differences
    ↓ shape: (1, 10, T, 128)
TensorRT FP16 Inference (< 50 ms target on Orin)
    ↓ shape: (C=13, T_tracks=3, 3)
Multi-ACCDOA Decoder (threshold = 0.5 on vector norm)
    ↓
MQTT Publish → NanoMQ local broker
    ↓
BUFFER DESTRUCTION (numpy.fill(0) + gc.collect())
```

#### TensorRT Engine

The model must be built **on the Jetson device** (engines are not portable across GPU architectures):

```bash
# On Jetson Orin — convert ONNX to TensorRT FP16 engine
bash scripts/build-tensorrt.sh models/seld_model.onnx models/seld_fp16.engine fp16

# Optional INT8 with calibration dataset
bash scripts/build-tensorrt.sh models/seld_model.onnx models/seld_int8.engine int8 2048 data/calibration/
```

#### Key environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DEVICE_ID` | `ARRAY_01` | Unique device identifier |
| `AUDIO_DEVICE` | `hw:1,0` | ALSA device string (use `mock` for dev) |
| `SAMPLE_RATE` | `48000` | Audio sample rate (Hz) |
| `NUM_CHANNELS` | `4` | Microphone channels (4=tetrahedral, 8=spherical) |
| `TENSORRT_ENGINE_PATH` | `/models/seld_fp16.engine` | TRT engine file path |
| `CONFIDENCE_THRESHOLD` | `0.5` | Minimum ACCDOA vector norm to emit prediction |
| `MQTT_BROKER_URL` | `mqtt://localhost:1883` | Local NanoMQ broker URL |
| `MOCK_AUDIO` | `false` | Use synthetic audio for testing |

---

### ML Training Pipeline

**Path:** `training/`  
**Runtime:** Cloud GPU (A100/H100), Python 3.12 + PyTorch 2.x  
**Container:** `seld-training` (CUDA 12.x)

#### Model Architecture: ResNet-Conformer

```
Input: (B, C_feat=10, T=100, F=128)
          │
    ┌─────▼──────────────────────────────┐
    │ Modified ResNet-18 Backbone         │
    │  • conv1: 7×7, 64ch (multi-channel) │
    │  • layer1-4: 64→128→256→512 ch      │
    │  • SE blocks after each stage        │
    └──────────────┬──────────────────────┘
                   │  (B, 512, T', 1) after freq pool
    ┌──────────────▼──────────────────────┐
    │ Linear projection: 512 → d_model=512 │
    └──────────────┬──────────────────────┘
                   │  (B, T', 512)
    ┌──────────────▼──────────────────────┐
    │ Conformer Encoder × 8 layers         │
    │  Each layer:                         │
    │    FFN(½) → MHSA(h=8) →             │
    │    DepthwiseConv(k=51) → FFN(½)     │
    │    → LayerNorm                       │
    └──────────────┬──────────────────────┘
                   │  (B, T', 512)
    ┌──────────────▼──────────────────────┐
    │ Multi-ACCDOA Head                    │
    │  Linear: 512 → C×T×3                │
    │  Reshape: (B, T', C=13, T=3, 3)     │
    │  No activation (regression output)  │
    └─────────────────────────────────────┘
```

#### ADPIT Loss

For each class `c`, ADPIT:
1. Enumerates all `T!` permutations of track assignments
2. Selects the minimum-cost permutation (PIT)
3. Adds auxiliary suppression loss on zero-target tracks: `λ · Σ ||pred_t||` when `target_t = 0`

This forces the model to zero out inactive tracks rather than hallucinating ghost detections.

#### Training

```bash
cd training

# Training with default config (STARSS23 dataset)
uv run python -m seld_training.train

# Override config options
uv run python -m seld_training.train \
    model.conformer_layers=8 \
    training.batch_size=64 \
    training.max_epochs=200 \
    wandb.project=my-seld-experiment

# W&B sweep for hyperparameter search
wandb sweep --project seld-digital-twin sweep.yaml
```

#### Export Pipeline

```bash
# 1. PyTorch → ONNX (opset 17)
uv run python -c "
from seld_training.export.onnx_export import export_to_onnx
from seld_training.model.resnet_conformer import ResNetConformer
model = ResNetConformer.load_from_checkpoint('checkpoints/best.ckpt')
export_to_onnx(model, 'models/seld_model.onnx', opset=17)
"

# 2. ONNX → TensorRT (on Jetson Orin)
bash scripts/build-tensorrt.sh models/seld_model.onnx models/seld_fp16.engine fp16
```

#### 4-Stage Data Augmentation

| Stage | Technique | Effect |
|-------|-----------|--------|
| ACS | Audio Channel Swapping | Expands DOA representations via rotation |
| MCS | Multi-Channel Simulation (SpatialScaper) | Synthesizes new spatial positions for isolated events |
| TDM | Time-Domain Mixing | Creates polyphonic scenes from mono events |
| TFM | Time-Frequency Masking (SpecAugment) | Increases spectral diversity, prevents overfitting |

---

### API Gateway

**Path:** `cloud/api-gateway/`  
**Runtime:** Python 3.12, FastAPI 0.115+  
**Port:** `8000`

The REST/WebSocket gateway is the single entry point for:
- Frontend dashboard (REST + WebSocket)
- External integrations (REST)
- Proxies internal gRPC calls to Spatial Engine, Semantic Layer, Device Service

```
GET/POST /api/v1/predictions      # Historical query + ingest webhook
WS       /api/v1/predictions/live # Real-time WebSocket stream
GET/POST /api/v1/devices          # Device registry CRUD
PATCH    /api/v1/devices/{id}     # Update device config
POST     /api/v1/spatial/query    # DOA → BIM asset hits
POST     /api/v1/spatial/triangulate  # Multi-sensor triangulation
POST     /api/v1/bim/upload       # Upload IFC model
GET      /api/v1/bim/{id}/glb     # Download GLB for viewer
POST     /api/v1/semantic/ask     # Prescriptive LLM (SSE stream)
POST     /api/v1/semantic/ingest  # Upload document for RAG
POST     /api/v1/vision/homography # Set camera-to-BIM matrix
GET      /api/v1/vision/tracks    # Current object tracks
GET      /health                  # Health check
```

#### WebSocket Protocol

Clients connect to `ws://host/api/v1/predictions/live`. Messages are JSON objects matching `EdgePredictionPayload`. Clients can send a JSON filter object to limit which devices are streamed:

```json
{"device_id": "ARRAY_04", "min_confidence": 0.7}
```

---

### Ingestion Service

**Path:** `cloud/ingestion-service/`  
**Runtime:** Python 3.12, aiokafka  

Consumes the `seld.predictions.raw` Kafka topic and batch-inserts into the TimescaleDB `predictions` hypertable. Also exposes an EMQX webhook endpoint for direct MQTT-to-HTTP ingestion.

**Data flow:**

```
NanoMQ (edge)
    ──MQTT 5.0 over QUIC──►
                            EMQX (cloud broker)
                                │
                        rule engine action
                                │
                        Kafka topic: seld.predictions.raw
                                │
                    Ingestion Service (aiokafka consumer)
                                │ batch_size=100 / timeout=500ms
                    TimescaleDB hypertable: predictions
                                │
                    WebSocket broadcast → Frontend
```

---

### Device Service

**Path:** `cloud/device-service/`  
**Runtime:** Python 3.12, grpcio  
**Port:** `50053` (gRPC)

Manages the device registry and provisions mTLS client certificates for edge devices. The certificate manager uses `cryptography` (ECDSA P-256) to issue X.509 certificates signed by an internal CA.

**mTLS flow:**
1. New Jetson device boots and calls `IssueCertificate(device_id, csr_pem)` via gRPC
2. Device Service signs the CSR with the internal CA
3. Device stores the certificate and uses it for all subsequent MQTT and HTTPS connections
4. EMQX and API Gateway validate the certificate against the internal CA

---

### Spatial Engine

**Path:** `spatial/`  
**Runtime:** Python 3.12, IfcOpenShell, trimesh, SciPy  
**Port:** `50051` (gRPC)

Converts DOA unit vectors from device-local frames to absolute BIM coordinates and intersects them with the building geometry.

#### IFC Parsing

Uses **IfcOpenShell** to:
1. Load `.ifc` files (IFC2X3, IFC4, IFC4X3 schemas)
2. Extract triangulated meshes via `ifcopenshell.geom.create_shape()` with world-space coordinates
3. Export to GLB via trimesh for the Three.js viewer

All meshes are combined into a single BVH-accelerated trimesh scene. The `trimesh.ray.intersects_location()` method provides O(log n) ray-mesh intersection.

#### Calibration

Each device has a 4×4 homogeneous transformation matrix stored in the `calibration_matrices` table:

```
v_world = M · [v_local; 0]   (direction, homogeneous with w=0)
o_world = M · [0, 0, 0, 1]ᵀ  (origin, device position in world coords)
```

#### Multi-Sensor Triangulation

Given N observations (origin + direction per device), finds the 3D point minimizing the sum of squared angular residuals via `scipy.optimize.least_squares`:

```python
residual_i(point) = weight_i × (1 - cos(∠(point - origin_i, direction_i)))
```

---

### Semantic Layer

**Path:** `semantic/`  
**Runtime:** Python 3.12, sentence-transformers, Anthropic SDK  
**Port:** `50052` (gRPC)

#### RAG Pipeline

```
Query: "valve_leak, confidence=0.94, near PMP-884-A"
    │
    ▼
Hybrid Search:
  ┌─ pgvector HNSW cosine similarity (weight=0.7)
  └─ PostgreSQL tsvector BM25 (weight=0.3)
  └─► RRF fusion (Reciprocal Rank Fusion, k=60)
    │
    ▼
Top-K chunks (technical manuals, P&ID schematics, maintenance logs)
    │
    ▼
Jinja2 prompt assembly (SELD data + spatial context + retrieved chunks)
    │
    ▼
Claude Sonnet / Llama 3.3 70B (streaming response)
    │
    ▼
Guardrails (PII filter, severity check, citation extraction)
    │
    ▼
SSE stream → Frontend
```

#### Document Ingestion

```bash
# Via API
curl -X POST http://localhost:8000/api/v1/semantic/ingest \
  -F "file=@manuals/pump_ksb_krt.pdf" \
  -F "asset_tags=PMP-884-A,pump,centrifugal"

# Python
from semantic_layer.document_ingestion import DocumentIngestion
ingestion = DocumentIngestion(settings)
await ingestion.ingest_pdf("manuals/pump_ksb_krt.pdf", asset_tags=["pump"])
```

#### Switching LLM Backend

| Backend | Config | Notes |
|---------|--------|-------|
| Anthropic Claude | `LLM_PROVIDER=anthropic` + `ANTHROPIC_API_KEY=...` | Cloud API, requires internet |
| Ollama (local) | `LLM_PROVIDER=ollama` + `OLLAMA_HOST=http://gpu-node:11434` | Air-gapped, needs GPU node |

---

### Vision Fusion

**Path:** `vision/`  
**Runtime:** Python 3.12, Ultralytics, OpenCV  

#### CCTV Pipeline

```
RTSP Stream(s)
    │
YOLO11 (object detection, TensorRT on edge AI box)
    │ classes: person, vehicle, machinery
BoT-SORT tracker (primary) / ByteTrack (fallback)
    │ track_id + bbox + class
Homography: pixel → BIM ground-plane coordinate
    │ H ∈ R^(3×3)
Redis state cache (TTL=60s)
    │ key: vision:track:{camera_id}:{track_id}
API Gateway → Frontend WebSocket
```

**Tracker selection** (via `TRACKER_TYPE` env var):
- `botsort` — BoT-SORT: camera motion compensation + ReID network. Best for crowded scenes where tracks cross.
- `bytetrack` — ByteTrack: lightweight, no ReID. Best for sparse, fast-moving objects.

#### Mobile Mapping

Upload a LiDAR scan from any mobile mapping platform (iPhone Pro, Leica BLK2GO, etc.):

```bash
curl -X POST http://localhost:8000/api/v1/vision/pointcloud \
  -F "file=@scan_factory_floor.las"
```

The pipeline: downsample → statistical outlier removal → 3D semantic segmentation → bounding box extraction → BIM-Lite model generation. This enables zero-cost recalibration when machinery is moved.

---

### Frontend Dashboard

**Path:** `frontend/`  
**Runtime:** Node.js 22, Next.js 15, React 19  
**Port:** `3000`

#### 3D Viewer

Built on **Three.js r171** via `@react-three/fiber`. Features:
- Loads BIM models in GLB format (converted from IFC at upload time)
- Renders live DOA vectors as `ArrowHelper` objects, color-coded by confidence
- Animates triangulation intersection markers as pulsing spheres
- Vehicle tracking overlay for tunnel scenarios (from Vision Fusion)
- Asset selection: click a BIM element to see details + recent events

#### Alert Panel

Receives predictions via WebSocket and renders them as alert cards with severity color-coding (`info` / `warning` / `critical` / `emergency`). LLM prescriptive responses stream token-by-token via SSE.

#### Development

```bash
cd frontend
pnpm dev        # http://localhost:3000 with HMR (Turbopack)
pnpm build      # Production build
pnpm typecheck  # tsc --noEmit
```

---

## Infrastructure

### Kubernetes (Production)

All services are deployed as Kubernetes workloads via **Pulumi TypeScript**:

```bash
cd infra
pulumi stack init staging
pulumi up --stack staging       # Deploy to staging
pulumi up --stack production    # Deploy to production
```

**Stacks:**

| Stack | Namespace | Replicas |
|-------|-----------|---------|
| dev | seld-dev | 1 |
| staging | seld-staging | 1-2 |
| production | seld-production | 3+ |

### Database High Availability

The production PostgreSQL setup uses **CloudNativePG** operator or **Patroni** for:
- Streaming replication (sync replica for zero data loss)
- Automatic failover
- Continuous WAL archiving via `WAL-G` to object storage

Extensions installed per cluster:
- `pgvector 0.8+` — vector similarity search (HNSW index)
- `TimescaleDB 2.x` — time-series hypertables + continuous aggregates
- `pg_trgm` — trigram similarity for fuzzy text search
- `btree_gin` — composite GIN indexes

### Monitoring Stack

| Component | Purpose | URL (local dev) |
|-----------|---------|----------------|
| Prometheus | Metrics collection | `:9090` |
| Grafana | Dashboards | `:3000` (monitor namespace) |
| Loki | Log aggregation | `:3100` |
| Jaeger | Distributed tracing | `:16686` |

Custom Grafana dashboards include:
- **Edge Fleet Health** — per-device CPU/GPU temp, inference latency, battery level
- **Prediction Rate** — events/minute per class per zone, using TimescaleDB continuous aggregates
- **Spatial Engine Latency** — raycasting p50/p95/p99 latency
- **LLM Usage** — token consumption, response latency, model distribution

---

## Database Schema

### Core tables

```sql
-- Device registry
devices (
  device_id VARCHAR(64) PK,
  name VARCHAR(255),
  hardware_type ENUM('industrial_capacitive', 'infrastructure_piezoelectric'),
  num_channels INT,
  status ENUM('online', 'offline', 'degraded', 'maintenance', 'provisioning'),
  location JSONB,
  last_seen TIMESTAMPTZ,
  ...
)

-- TimescaleDB hypertable (partitioned by day)
predictions (
  time TIMESTAMPTZ NOT NULL,   -- partition key
  device_id VARCHAR(64),
  class_name VARCHAR(128),
  confidence FLOAT,
  vector_x, vector_y, vector_z FLOAT,
  inference_ms FLOAT
)
-- Continuous aggregate for dashboards:
CREATE MATERIALIZED VIEW prediction_rates_1m ...

-- Calibration matrices (4×4 transform per device per BIM model)
calibration_matrices (
  device_id VARCHAR(64) FK,
  bim_model_id VARCHAR(64) FK,
  matrix FLOAT[16],     -- row-major 4×4 homogeneous matrix
  origin_x, origin_y, origin_z FLOAT,
  ...
)

-- Vector embeddings for RAG (pgvector)
document_chunks (
  content TEXT,
  source VARCHAR(512),
  embedding vector(1536),    -- pgvector column
  content_tsv tsvector,       -- full-text search (generated)
  asset_tags TEXT[],
  ...
)
-- HNSW index: CREATE INDEX ... USING hnsw (embedding vector_cosine_ops)
-- GIN index:  CREATE INDEX ... USING GIN (content_tsv)

-- TimescaleDB hypertable for spatial event log
spatial_hits (
  time TIMESTAMPTZ NOT NULL,
  device_id, bim_model_id, class_name,
  asset_id, asset_name, ifc_type,
  hit_x, hit_y, hit_z FLOAT,
  triangulation_residual FLOAT,
  ...
)
```

### Migrations

```bash
# Apply all migrations
cd migrations && uv run alembic upgrade head

# Create new migration
uv run alembic revision --autogenerate -m "add_feature_x"

# Rollback one step
uv run alembic downgrade -1
```

---

## Configuration Reference

All services use `pydantic-settings` with environment variable loading. Create a `.env` file in the service directory or export variables before running.

### Edge Agent (`.env`)

```bash
DEVICE_ID=ARRAY_04
AUDIO_DEVICE=hw:1,0          # or 'mock' for dev
SAMPLE_RATE=48000
FRAME_LENGTH_MS=100
NUM_CHANNELS=4
CONFIDENCE_THRESHOLD=0.5
TENSORRT_ENGINE_PATH=/models/seld_fp16.engine
MQTT_BROKER_URL=mqtt://localhost:1883
CLOUD_API_URL=https://api.yourdomain.com
OTA_CHECK_INTERVAL_S=300
MOCK_AUDIO=false
```

### Cloud Services (`.env`)

```bash
DATABASE_URL=postgresql+asyncpg://seld:password@localhost:5432/seld_db
REDIS_URL=redis://localhost:6379/0
KAFKA_BOOTSTRAP_SERVERS=localhost:19092
EMQX_HOST=localhost
SPATIAL_GRPC_HOST=localhost:50051
SEMANTIC_GRPC_HOST=localhost:50052
DEVICE_GRPC_HOST=localhost:50053
JWT_SECRET=your-secret-here
```

### Semantic Layer (`.env`)

```bash
# Anthropic (cloud)
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-sonnet-4-20250514

# OR Ollama (self-hosted / air-gapped)
LLM_PROVIDER=ollama
OLLAMA_HOST=http://gpu-node:11434
OLLAMA_MODEL=llama3.3:70b

# Embeddings
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

---

## Development Workflow

### Adding a New Sound Event Class

1. **Annotate training data** using Label Studio or the custom annotation tool
2. **Update class list** in `training/src/seld_training/config/default.yaml`:
   ```yaml
   model:
     num_classes: 14   # was 13
   ```
3. **Retrain** the model with ADPIT loss (handles new class automatically)
4. **Export** to ONNX → TensorRT and OTA push to edge devices
5. **Update** `edge/src/edge_agent/config.py` `class_names` list
6. **Ingest** relevant documentation into the semantic layer for the new fault class

### Adding a New Edge Device

```bash
# 1. Register in device service
curl -X POST http://localhost:8000/api/v1/devices \
  -H "Content-Type: application/json" \
  -d '{
    "device_id": "ARRAY_05",
    "name": "Compressor Hall South",
    "hardware_type": "industrial_capacitive",
    "num_channels": 4,
    "location": {"building": "Plant-A", "zone": "Compressor-Hall"}
  }'

# 2. Calibrate: measure device position in BIM coordinates,
#    compute the 4×4 roto-translation matrix, then:
curl -X POST http://localhost:8000/api/v1/spatial/calibration \
  -d '{"device_id": "ARRAY_05", "bim_model_id": "bim-plant-a", "matrix": [...]}'

# 3. Flash Jetson with the edge agent Docker image and set env vars
```

### Generating Protobuf Stubs

```bash
bash scripts/generate-protos.sh
# Outputs:
#   packages/proto/generated/python/  → Python gRPC stubs
#   packages/proto/generated/typescript/ → TypeScript gRPC stubs
```

---

## Testing

### Unit Tests (no external services)

```bash
# All Python unit tests
cd tests && uv run pytest unit/ -v

# Specific service
cd spatial && uv run pytest tests/ -v

# TypeScript unit tests
pnpm --filter frontend test
```

### Integration Tests (requires docker-compose)

```bash
docker compose up -d
sleep 10  # wait for services

cd tests && uv run pytest integration/ -v -m "not e2e"
```

### End-to-End Smoke Test

```bash
cd tests && uv run pytest integration/test_e2e_smoke.py -v -m e2e
```

### Load Test (Locust)

```bash
# Simulate 1000 concurrent edge nodes + 100 dashboard users
cd tests/load
uv run locust -f locustfile.py \
  --host http://localhost:8000 \
  --users 1100 \
  --spawn-rate 50 \
  --run-time 5m \
  --headless
```

### SELD Model Regression Test

Before any model OTA push, verify performance does not degrade:

```bash
cd training
uv run python -c "
from seld_training.evaluation.regression_test import check_seld_regression
import json

with open('evaluation/current_metrics.json') as f:
    metrics = json.load(f)

passed, msg = check_seld_regression(metrics, 'evaluation/baseline_metrics.json')
print(msg)
assert passed, 'SELD regression detected!'
"
```

### Security Tests

```bash
cd tests && uv run pytest security/ -v -m security
```

---

## Deployment

### CI/CD Pipeline

| Pipeline | Trigger | Actions |
|----------|---------|---------|
| `ci-edge.yml` | Push to `edge/` | Lint → Test → Build aarch64 Docker image |
| `ci-cloud.yml` | Push to `cloud/` etc. | Lint → Test → Build + push Docker images |
| `ci-frontend.yml` | Push to `frontend/` | Type check → Build → Playwright tests |
| `ci-ml-training.yml` | Push to `training/` | Lint → Model forward pass → ONNX export smoke |
| `cd-deploy.yml` | Push to `main` / manual | Helm upgrade to staging or production |

### Manual Production Deployment

```bash
# Build and push all service images
TAG=$(git rev-parse --short HEAD)
docker buildx bake --push TAG=$TAG

# Deploy to Kubernetes
cd infra
pulumi up --stack production --yes

# Verify rollout
kubectl rollout status deployment/api-gateway -n seld-production
kubectl rollout status deployment/spatial-engine -n seld-production
```

### OTA Model Update to Edge Devices

1. Train and export the new TensorRT engine on a Jetson Orin (or cross-compile)
2. Upload to the model registry (W&B Artifacts or S3)
3. API Gateway broadcasts the OTA command to all target devices via MQTT:
   ```
   Topic: dt/cloud/ota/{device_id}
   Payload: {"version": "v1.2.0", "url": "https://...", "sha256": "abc123..."}
   ```
4. Edge agent downloads, verifies SHA256, atomically swaps the engine file, and restarts inference

---

## MQTT Topics & Payload Schema

### Edge → Cloud

```
Topic: dt/edge/{device_id}/predictions
QoS: 1

Payload:
{
  "device_id": "ARRAY_04",
  "timestamp": "2026-02-21T17:59:28Z",
  "frame_idx": 4201,
  "predictions": [
    {
      "class": "valve_leak",
      "confidence": 0.94,
      "vector": [0.65, 0.70, -0.29]
    }
  ],
  "telemetry": {
    "inference_ms": 12.4,
    "cpu_temp": 52.0,
    "gpu_temp": 61.0,
    "mem_used_mb": 3072
  }
}
```

```
Topic: dt/edge/{device_id}/telemetry
QoS: 0  (fire-and-forget, high frequency)
```

### Cloud → Edge

```
Topic: dt/cloud/config/{device_id}
Payload: {DeviceConfig JSON}

Topic: dt/cloud/ota/{device_id}
Payload: {"version": "...", "url": "...", "sha256": "..."}
```

### NanoMQ Bridge Configuration

The edge NanoMQ broker is configured to forward prediction and telemetry topics to the cloud EMQX broker via **MQTT 5.0 over QUIC** transport (`mqtt-quic://`), providing:
- 0-RTT connection resumption (< 1 ms reconnect after network interruption)
- Multiplexed streams (no head-of-line blocking)
- 20–40% lower latency on lossy industrial networks vs. TCP

---

## API Reference

Full OpenAPI documentation is available at **http://localhost:8000/docs** (Swagger UI) or **http://localhost:8000/redoc** when the API Gateway is running.

### Key endpoints

#### `POST /api/v1/spatial/query`

```json
Request:
{
  "device_id": "ARRAY_04",
  "direction": {"x": 0.65, "y": 0.70, "z": -0.29},
  "bim_model_id": "bim-plant-a",
  "max_distance": 50.0
}

Response:
{
  "ray_origin": {"x": 10.5, "y": 3.2, "z": 2.0},
  "ray_direction": {"x": 0.614, "y": 0.661, "z": -0.274},
  "hits": [
    {
      "asset_id": "1AbCD$...",
      "asset_name": "Pump PMP-884-A",
      "ifc_type": "IfcFlowMovingDevice",
      "hit_point": {"x": 13.1, "y": 6.5, "z": 1.2},
      "distance": 4.7,
      "confidence": 0.94
    }
  ]
}
```

#### `POST /api/v1/semantic/ask` (SSE stream)

```json
Request:
{
  "device_id": "ARRAY_04",
  "class_name": "valve_leak",
  "confidence": 0.94,
  "vector": [0.65, 0.70, -0.29],
  "bim_model_id": "bim-plant-a"
}

Response: text/event-stream
data: {"token": "Acoustic", "is_final": false}
data: {"token": " anomaly", "is_final": false}
...
data: {"token": "", "is_final": true, "severity": "CRITICAL", "actions": [...]}
```

---

## Security & GDPR

### Audio Data Privacy

- Audio buffers are **destroyed within 500 ms** of capture using `numpy.fill(0)` followed by explicit `del` and `gc.collect()`
- **No raw audio ever leaves the edge device** (Late Fusion paradigm)
- GDPR Article 25 (Privacy by Design) compliant by architecture

### Transport Security

| Connection | Protocol | Authentication |
|-----------|---------|---------------|
| Edge ↔ NanoMQ | MQTT 5.0 (localhost) | None (same host) |
| NanoMQ ↔ EMQX | MQTT 5.0 over QUIC (TLS 1.3) | mTLS (X.509 client cert) |
| Edge ↔ API Gateway | HTTPS (TLS 1.3) | mTLS (X.509 client cert) |
| Dashboard ↔ API Gateway | HTTPS/WSS | OAuth2/OIDC (Keycloak) |
| Service ↔ Service | gRPC (TLS optional) | mTLS in production |

### mTLS Certificate Lifecycle

1. Device boots → calls `DeviceRegistryService.IssueCertificate()` via gRPC
2. Device Service generates ECDSA P-256 keypair and CSR
3. Signs with internal CA (stored in K8s Secret, rotated annually)
4. Certificate valid for 365 days, auto-renewed 30 days before expiry
5. Revoked certificates stored in `device_certificates` table (`revoked=true`)

### LLM Guardrails

The semantic layer guardrails enforce:
- **PII scrubbing**: SSN, email, credit card patterns removed from all inputs/outputs
- **Severity check**: Prescriptive responses must include a severity level
- **Length limits**: Inputs capped at 50K characters
- **Domain guardrail**: Off-topic responses logged and flagged for review

---

## Technical Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Edge MQTT broker | **NanoMQ** | 2 MB footprint, native POSIX/ARM, MQTT 5.0 + QUIC. No Docker daemon needed on Jetson. |
| Cloud MQTT broker | **EMQX** | Built-in rule engine routes to Kafka. Handles 100M+ concurrent connections. |
| Event streaming | **Redpanda** | Kafka-compatible, simpler to operate (no ZooKeeper), 10x lower p99 latency. |
| Vector database | **pgvector + pgvectorscale** | Single PostgreSQL instance for relational + vector + time-series. Avoids separate Pinecone/Qdrant service. StreamingDiskANN index handles >10M vectors. |
| Time-series | **TimescaleDB** | PostgreSQL extension (same cluster as pgvector). Hypertables + continuous aggregates for prediction dashboards. |
| BIM parsing | **IfcOpenShell** | Python, cross-platform, IFC2X3/4/4X3, LGPL. xBIM excluded (C#/.NET only). |
| Spatial math | **trimesh + SciPy** | BVH-accelerated ray-mesh intersection; least-squares optimizer for triangulation. Rust (nalgebra + parry3d) available for hot-path if sub-ms latency required. |
| Object detection | **YOLO11** (Ultralytics) | 22% fewer params than YOLOv8, 2.4ms on Jetson, native BoT-SORT/ByteTrack integration. |
| Edge inference | **TensorRT** | 5-10x faster than ONNX Runtime on Jetson GPU. FP16 guaranteed on Orin (compute capability 8.7). |
| IaC | **Pulumi TypeScript** | Type-safe, testable, consistent with frontend stack. Eliminates YAML-heavy Terraform for complex conditional logic. |
| Hybrid search | **pgvector HNSW + tsvector + RRF** | Avoids a dedicated search engine (Elasticsearch/Typesense). RRF fusion of vector and BM25 scores consistently outperforms either alone. |

---

## Contributing

### Code Style

Python services use **ruff** for linting and formatting:
```bash
uv run ruff check src/
uv run ruff format src/
```

TypeScript services use **ESLint** + **Prettier**:
```bash
pnpm lint
pnpm format
```

### Branch Strategy

- `main` — stable, protected, auto-deploys to staging on push
- `develop` — integration branch
- `feature/*` — feature branches, PR to develop
- `hotfix/*` — production hotfixes, PR to main + develop

### Pull Request Requirements

- All CI checks must pass
- SELD model changes must include regression test results
- Database schema changes must include a forward-only Alembic migration
- New environment variables must be documented in `Configuration Reference`

---

## License

Apache License 2.0 — see [LICENSE](LICENSE) for the full text.

---

*Built on the shoulders of DCASE Challenge Task 3 research (2022–2025), STARSS23 dataset (Sony-TAu Realistic Spatial Soundscapes), and the open-source SELD community.*
