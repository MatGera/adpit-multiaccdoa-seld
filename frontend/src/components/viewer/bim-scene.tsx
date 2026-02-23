"use client";

import { Canvas } from "@react-three/fiber";
import { OrbitControls, Grid, Environment, Html } from "@react-three/drei";
import { Suspense, useRef } from "react";
import * as THREE from "three";

interface BIMSceneProps {
  onAssetSelect?: (assetId: string | null) => void;
}

function LoadedModel({ onAssetSelect }: BIMSceneProps) {
  const groupRef = useRef<THREE.Group>(null);

  // Placeholder geometry — will be replaced by IFC/GLB loader
  return (
    <group ref={groupRef}>
      {/* Floor */}
      <mesh position={[0, -0.05, 0]} receiveShadow>
        <boxGeometry args={[20, 0.1, 20]} />
        <meshStandardMaterial color="#1a1a2e" />
      </mesh>

      {/* Sample building structure */}
      <mesh position={[0, 2, 0]} castShadow onClick={() => onAssetSelect?.("wall-001")}>
        <boxGeometry args={[10, 4, 0.3]} />
        <meshStandardMaterial color="#2d3748" transparent opacity={0.6} />
      </mesh>

      <mesh position={[5, 2, -5]} castShadow onClick={() => onAssetSelect?.("column-001")}>
        <cylinderGeometry args={[0.3, 0.3, 4, 16]} />
        <meshStandardMaterial color="#4a5568" />
      </mesh>

      <mesh position={[-5, 2, -5]} castShadow onClick={() => onAssetSelect?.("column-002")}>
        <cylinderGeometry args={[0.3, 0.3, 4, 16]} />
        <meshStandardMaterial color="#4a5568" />
      </mesh>

      {/* Microphone array indicator */}
      <mesh position={[0, 3, -2]}>
        <sphereGeometry args={[0.15, 16, 16]} />
        <meshStandardMaterial color="#3b82f6" emissive="#3b82f6" emissiveIntensity={0.5} />
      </mesh>

      {/* DOA arrow — placeholder */}
      <arrowHelper
        args={[
          new THREE.Vector3(0.7, -0.3, 0.5).normalize(),
          new THREE.Vector3(0, 3, -2),
          3,
          0xef4444,
          0.3,
          0.15,
        ]}
      />
    </group>
  );
}

export function BIMScene({ onAssetSelect }: BIMSceneProps) {
  return (
    <Canvas
      camera={{ position: [12, 8, 12], fov: 50 }}
      shadows
      style={{ width: "100%", height: "100%" }}
      gl={{ antialias: true }}
    >
      <color attach="background" args={["#0a0a14"]} />

      <ambientLight intensity={0.3} />
      <directionalLight position={[10, 15, 10]} intensity={1} castShadow />
      <pointLight position={[0, 3, -2]} color="#3b82f6" intensity={0.5} distance={8} />

      <Suspense
        fallback={
          <Html center>
            <div className="text-white bg-gray-800 px-4 py-2 rounded">
              Loading model...
            </div>
          </Html>
        }
      >
        <LoadedModel onAssetSelect={onAssetSelect} />
      </Suspense>

      <Grid
        infiniteGrid
        cellSize={1}
        cellThickness={0.5}
        sectionSize={5}
        sectionThickness={1}
        cellColor="#1e293b"
        sectionColor="#334155"
        fadeDistance={50}
        position={[0, 0, 0]}
      />

      <OrbitControls
        enableDamping
        dampingFactor={0.05}
        minDistance={3}
        maxDistance={100}
        maxPolarAngle={Math.PI / 2}
      />
    </Canvas>
  );
}
