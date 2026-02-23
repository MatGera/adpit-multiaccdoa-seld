import * as k8s from "@pulumi/kubernetes";

export function deployMonitoring(namespace: string) {
  const labels = { app: "monitoring" };

  // Prometheus
  const prometheus = new k8s.apps.v1.Deployment("prometheus", {
    metadata: { namespace, labels: { ...labels, component: "prometheus" } },
    spec: {
      replicas: 1,
      selector: { matchLabels: { component: "prometheus" } },
      template: {
        metadata: { labels: { component: "prometheus" } },
        spec: {
          containers: [
            {
              name: "prometheus",
              image: "prom/prometheus:v2.55.1",
              ports: [{ containerPort: 9090 }],
              resources: {
                requests: { cpu: "100m", memory: "256Mi" },
                limits: { cpu: "500m", memory: "1Gi" },
              },
            },
          ],
        },
      },
    },
  });

  // Grafana
  const grafana = new k8s.apps.v1.Deployment("grafana", {
    metadata: { namespace, labels: { ...labels, component: "grafana" } },
    spec: {
      replicas: 1,
      selector: { matchLabels: { component: "grafana" } },
      template: {
        metadata: { labels: { component: "grafana" } },
        spec: {
          containers: [
            {
              name: "grafana",
              image: "grafana/grafana:11.4.0",
              ports: [{ containerPort: 3000 }],
              env: [
                { name: "GF_SECURITY_ADMIN_PASSWORD", value: "admin" },
              ],
              resources: {
                requests: { cpu: "100m", memory: "128Mi" },
                limits: { cpu: "250m", memory: "512Mi" },
              },
            },
          ],
        },
      },
    },
  });

  // Loki
  const loki = new k8s.apps.v1.Deployment("loki", {
    metadata: { namespace, labels: { ...labels, component: "loki" } },
    spec: {
      replicas: 1,
      selector: { matchLabels: { component: "loki" } },
      template: {
        metadata: { labels: { component: "loki" } },
        spec: {
          containers: [
            {
              name: "loki",
              image: "grafana/loki:3.3.2",
              ports: [{ containerPort: 3100 }],
              resources: {
                requests: { cpu: "100m", memory: "128Mi" },
                limits: { cpu: "250m", memory: "512Mi" },
              },
            },
          ],
        },
      },
    },
  });

  // Jaeger
  const jaeger = new k8s.apps.v1.Deployment("jaeger", {
    metadata: { namespace, labels: { ...labels, component: "jaeger" } },
    spec: {
      replicas: 1,
      selector: { matchLabels: { component: "jaeger" } },
      template: {
        metadata: { labels: { component: "jaeger" } },
        spec: {
          containers: [
            {
              name: "jaeger",
              image: "jaegertracing/all-in-one:1.64",
              ports: [
                { name: "ui", containerPort: 16686 },
                { name: "otlp-grpc", containerPort: 4317 },
              ],
              resources: {
                requests: { cpu: "100m", memory: "128Mi" },
                limits: { cpu: "250m", memory: "512Mi" },
              },
            },
          ],
        },
      },
    },
  });

  // Services
  for (const [name, port] of [
    ["prometheus", 9090],
    ["grafana", 3000],
    ["loki", 3100],
    ["jaeger", 16686],
  ] as const) {
    new k8s.core.v1.Service(`${name}-svc`, {
      metadata: { namespace, name, labels: { ...labels, component: name } },
      spec: {
        selector: { component: name },
        ports: [{ port, targetPort: port }],
      },
    });
  }

  return { prometheus, grafana, loki, jaeger };
}
