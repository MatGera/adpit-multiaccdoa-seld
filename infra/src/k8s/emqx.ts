import * as k8s from "@pulumi/kubernetes";
import * as pulumi from "@pulumi/pulumi";

interface EMQXOptions {
  replicaCount: number;
}

export function deployEMQX(namespace: string, opts: EMQXOptions) {
  const labels = { app: "emqx", component: "mqtt-broker" };

  const deployment = new k8s.apps.v1.Deployment("emqx", {
    metadata: { namespace, labels },
    spec: {
      replicas: opts.replicaCount,
      selector: { matchLabels: labels },
      template: {
        metadata: { labels },
        spec: {
          containers: [
            {
              name: "emqx",
              image: "emqx/emqx:5.8",
              ports: [
                { name: "mqtt", containerPort: 1883 },
                { name: "mqtts", containerPort: 8883 },
                { name: "ws", containerPort: 8083 },
                { name: "dashboard", containerPort: 18083 },
              ],
              env: [
                { name: "EMQX_NAME", value: "emqx" },
                { name: "EMQX_LOADED_PLUGINS", value: "emqx_dashboard" },
              ],
              resources: {
                requests: { cpu: "250m", memory: "512Mi" },
                limits: { cpu: "1", memory: "1Gi" },
              },
            },
          ],
        },
      },
    },
  });

  const service = new k8s.core.v1.Service("emqx-svc", {
    metadata: { namespace, name: "emqx", labels },
    spec: {
      selector: labels,
      ports: [
        { name: "mqtt", port: 1883, targetPort: 1883 },
        { name: "mqtts", port: 8883, targetPort: 8883 },
        { name: "ws", port: 8083, targetPort: 8083 },
        { name: "dashboard", port: 18083, targetPort: 18083 },
      ],
    },
  });

  return {
    host: pulumi.interpolate`emqx.${namespace}.svc.cluster.local`,
    dashboardUrl: pulumi.interpolate`http://emqx.${namespace}.svc.cluster.local:18083`,
    service,
  };
}
