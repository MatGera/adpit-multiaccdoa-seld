import * as k8s from "@pulumi/kubernetes";
import * as pulumi from "@pulumi/pulumi";

interface KafkaOptions {
  replicaCount: number;
}

export function deployKafka(namespace: string, opts: KafkaOptions) {
  const labels = { app: "redpanda", component: "streaming" };

  // Using Redpanda as Kafka-compatible broker (simpler to operate)
  const statefulSet = new k8s.apps.v1.StatefulSet("redpanda", {
    metadata: { namespace, labels },
    spec: {
      serviceName: "redpanda",
      replicas: opts.replicaCount,
      selector: { matchLabels: labels },
      template: {
        metadata: { labels },
        spec: {
          containers: [
            {
              name: "redpanda",
              image: "redpandadata/redpanda:v24.3.1",
              command: [
                "rpk", "redpanda", "start",
                "--kafka-addr", "0.0.0.0:9092",
                "--advertise-kafka-addr", `redpanda.${namespace}.svc.cluster.local:9092`,
                "--smp", "1",
                "--memory", "1G",
                "--reserve-memory", "0M",
                "--overprovisioned",
              ],
              ports: [
                { name: "kafka", containerPort: 9092 },
                { name: "admin", containerPort: 9644 },
              ],
              resources: {
                requests: { cpu: "250m", memory: "512Mi" },
                limits: { cpu: "1", memory: "1536Mi" },
              },
            },
          ],
        },
      },
    },
  });

  const service = new k8s.core.v1.Service("redpanda-svc", {
    metadata: { namespace, name: "redpanda", labels },
    spec: {
      selector: labels,
      ports: [
        { name: "kafka", port: 9092, targetPort: 9092 },
        { name: "admin", port: 9644, targetPort: 9644 },
      ],
      clusterIP: "None",
    },
  });

  return {
    bootstrapServers: pulumi.interpolate`redpanda.${namespace}.svc.cluster.local:9092`,
    service,
  };
}
