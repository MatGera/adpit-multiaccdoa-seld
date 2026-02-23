import * as k8s from "@pulumi/kubernetes";

export function createNamespace(name: string, env: string) {
  return new k8s.core.v1.Namespace(name, {
    metadata: {
      name,
      labels: {
        app: "seld-digital-twin",
        environment: env,
      },
    },
  });
}
