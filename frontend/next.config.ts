import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  transpilePackages: ["@seld/shared-types"],
  webpack: (config) => {
    // web-ifc requires WASM support
    config.experiments = {
      ...config.experiments,
      asyncWebAssembly: true,
    };

    return config;
  },
};

export default nextConfig;
