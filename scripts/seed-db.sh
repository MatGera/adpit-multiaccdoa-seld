#!/usr/bin/env bash
set -euo pipefail

echo "=== Seeding development database ==="

DB_URL="${DATABASE_URL:-postgresql://seld:seld_dev_password@localhost:5432/seld_db}"

psql "$DB_URL" <<'SQL'
-- Seed test devices
INSERT INTO devices (device_id, name, hardware_type, num_channels, location, status)
VALUES
    ('ARRAY_01', 'Pump Room Sensor 1', 'industrial_capacitive', 4,
     '{"building": "Plant-A", "floor": "B1", "zone": "Pump-Room", "coordinates": {"x": 10.5, "y": 3.2, "z": 2.0}}',
     'online'),
    ('ARRAY_02', 'Pump Room Sensor 2', 'industrial_capacitive', 4,
     '{"building": "Plant-A", "floor": "B1", "zone": "Pump-Room", "coordinates": {"x": 15.0, "y": 3.2, "z": 2.0}}',
     'online'),
    ('ARRAY_03', 'Compressor Hall Sensor', 'industrial_capacitive', 8,
     '{"building": "Plant-A", "floor": "G", "zone": "Compressor-Hall", "coordinates": {"x": 25.0, "y": 8.0, "z": 3.5}}',
     'online'),
    ('ARRAY_04', 'Bridge Pier Sensor', 'infrastructure_piezoelectric', 4,
     '{"building": "Bridge-X1", "zone": "Pier-3", "coordinates": {"x": 0.0, "y": 0.0, "z": 5.0}}',
     'degraded')
ON CONFLICT (device_id) DO NOTHING;

-- Seed a test BIM model entry
INSERT INTO bim_models (id, name, description, ifc_file_path, status)
VALUES
    ('bim-plant-a', 'Plant A — Industrial Facility', 'Main production facility BIM model',
     '/data/bim/plant_a.ifc', 'ready')
ON CONFLICT (id) DO NOTHING;

-- Seed calibration matrices (identity matrices for dev)
INSERT INTO calibration_matrices (device_id, bim_model_id, matrix, origin_x, origin_y, origin_z)
VALUES
    ('ARRAY_01', 'bim-plant-a',
     '{1,0,0,10.5, 0,1,0,3.2, 0,0,1,2.0, 0,0,0,1}',
     10.5, 3.2, 2.0),
    ('ARRAY_02', 'bim-plant-a',
     '{1,0,0,15.0, 0,1,0,3.2, 0,0,1,2.0, 0,0,0,1}',
     15.0, 3.2, 2.0)
ON CONFLICT ON CONSTRAINT uq_calibration_device_model DO NOTHING;

SQL

echo "=== Database seeded successfully ==="
