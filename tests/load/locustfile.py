"""Load tests using Locust.

Run with: locust -f tests/load/locustfile.py --host http://localhost:8000
"""

from __future__ import annotations

import json
import random
import time

from locust import HttpUser, task, between


class PredictionQueryUser(HttpUser):
    """Simulates operators querying the prediction API."""

    wait_time = between(0.5, 2.0)

    @task(5)
    def query_predictions(self):
        """Query recent predictions."""
        self.client.get(
            "/api/v1/predictions",
            params={
                "limit": random.choice([10, 50, 100]),
                "min_confidence": random.uniform(0.3, 0.8),
            },
        )

    @task(3)
    def list_devices(self):
        """List all devices."""
        self.client.get("/api/v1/devices")

    @task(2)
    def spatial_query(self):
        """Perform a spatial query."""
        self.client.post(
            "/api/v1/spatial/query",
            json={
                "device_id": f"device-{random.randint(1, 10):03d}",
                "direction": {
                    "x": random.uniform(-1, 1),
                    "y": random.uniform(-1, 1),
                    "z": random.uniform(0, 1),
                },
                "bim_model_id": "test-model",
            },
        )

    @task(1)
    def health_check(self):
        """Hit health endpoint."""
        self.client.get("/health")


class EdgeDeviceSimulator(HttpUser):
    """Simulates edge devices sending predictions at high frequency."""

    wait_time = between(0.05, 0.2)  # 5-20 predictions/second

    def on_start(self):
        self.device_id = f"sim-device-{random.randint(1, 100):03d}"
        self.frame_idx = 0

    @task
    def send_prediction(self):
        """Simulate an edge device sending a prediction payload."""
        classes = [
            "machinery_impact", "alarm", "engine_idle",
            "footsteps", "door_slam", "glass_break",
        ]

        payload = {
            "device_id": self.device_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            "frame_idx": self.frame_idx,
            "predictions": [
                {
                    "class": random.choice(classes),
                    "confidence": random.uniform(0.4, 0.99),
                    "vector": [
                        random.uniform(-1, 1),
                        random.uniform(-1, 1),
                        random.uniform(0, 1),
                    ],
                }
                for _ in range(random.randint(1, 3))
            ],
        }

        self.client.post(
            "/api/v1/predictions",
            json=payload,
            name="/api/v1/predictions [POST]",
        )
        self.frame_idx += 1
