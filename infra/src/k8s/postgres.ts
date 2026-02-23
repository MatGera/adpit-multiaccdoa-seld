import * as k8s from "@pulumi/kubernetes";
import * as pulumi from "@pulumi/pulumi";

interface PostgresOptions {
  replicaCount: number;
}

export function deployPostgres(namespace: string, opts: PostgresOptions) {
  const labels = { app: "postgres", component: "database" };

  // PVC for persistent storage
  const pvc = new k8s.core.v1.PersistentVolumeClaim("postgres-pvc", {
    metadata: { namespace, labels },
    spec: {
      accessModes: ["ReadWriteOnce"],
      resources: { requests: { storage: "50Gi" } },
    },
  });

  // ConfigMap for init scripts
  const initConfig = new k8s.core.v1.ConfigMap("postgres-init", {
    metadata: { namespace, labels },
    data: {
      "init-extensions.sql": `
        CREATE EXTENSION IF NOT EXISTS vector;
        CREATE EXTENSION IF NOT EXISTS timescaledb;
      `,
    },
  });

  // Secret for credentials
  const secret = new k8s.core.v1.Secret("postgres-secret", {
    metadata: { namespace, labels },
    stringData: {
      POSTGRES_USER: "seld",
      POSTGRES_PASSWORD: "seld_dev_password",
      POSTGRES_DB: "seld_db",
    },
  });

  // StatefulSet
  const statefulSet = new k8s.apps.v1.StatefulSet("postgres", {
    metadata: { namespace, labels },
    spec: {
      serviceName: "postgres",
      replicas: 1,
      selector: { matchLabels: labels },
      template: {
        metadata: { labels },
        spec: {
          containers: [
            {
              name: "postgres",
              image: "timescale/timescaledb-ha:pg16-ts2.17",
              ports: [{ containerPort: 5432 }],
              envFrom: [{ secretRef: { name: secret.metadata.name } }],
              volumeMounts: [
                { name: "data", mountPath: "/var/lib/postgresql/data" },
                { name: "init", mountPath: "/docker-entrypoint-initdb.d" },
              ],
              resources: {
                requests: { cpu: "500m", memory: "1Gi" },
                limits: { cpu: "2", memory: "4Gi" },
              },
            },
          ],
          volumes: [
            { name: "data", persistentVolumeClaim: { claimName: pvc.metadata.name } },
            { name: "init", configMap: { name: initConfig.metadata.name } },
          ],
        },
      },
    },
  });

  // Service
  const service = new k8s.core.v1.Service("postgres-svc", {
    metadata: { namespace, name: "postgres", labels },
    spec: {
      selector: labels,
      ports: [{ port: 5432, targetPort: 5432 }],
      clusterIP: "None",
    },
  });

  return {
    host: pulumi.interpolate`postgres.${namespace}.svc.cluster.local`,
    service,
  };
}
