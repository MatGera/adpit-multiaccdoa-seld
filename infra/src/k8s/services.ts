import * as k8s from "@pulumi/kubernetes";
import * as pulumi from "@pulumi/pulumi";

interface ServicesOptions {
  replicaCount: number;
  postgresHost: pulumi.Output<string>;
  kafkaBootstrap: pulumi.Output<string>;
  emqxHost: pulumi.Output<string>;
}

function createServiceDeployment(
  name: string,
  namespace: string,
  image: string,
  port: number,
  replicas: number,
  env: { name: string; value: pulumi.Input<string> }[] = []
) {
  const labels = { app: name };

  const deployment = new k8s.apps.v1.Deployment(name, {
    metadata: { namespace, labels },
    spec: {
      replicas,
      selector: { matchLabels: labels },
      template: {
        metadata: { labels },
        spec: {
          containers: [
            {
              name,
              image,
              ports: [{ containerPort: port }],
              env,
              resources: {
                requests: { cpu: "100m", memory: "256Mi" },
                limits: { cpu: "500m", memory: "512Mi" },
              },
              livenessProbe: {
                httpGet: { path: "/health", port },
                initialDelaySeconds: 10,
                periodSeconds: 30,
              },
            },
          ],
        },
      },
    },
  });

  const service = new k8s.core.v1.Service(`${name}-svc`, {
    metadata: { namespace, name, labels },
    spec: {
      selector: labels,
      ports: [{ port, targetPort: port }],
    },
  });

  return { deployment, service };
}

export function deployServices(namespace: string, opts: ServicesOptions) {
  const commonEnv = [
    { name: "DATABASE_URL", value: pulumi.interpolate`postgresql+asyncpg://seld:seld_dev_password@${opts.postgresHost}:5432/seld_db` },
    { name: "KAFKA_BOOTSTRAP_SERVERS", value: opts.kafkaBootstrap },
    { name: "EMQX_HOST", value: opts.emqxHost },
  ];

  const apiGateway = createServiceDeployment(
    "api-gateway", namespace,
    "seld/api-gateway:latest", 8000, opts.replicaCount,
    [
      ...commonEnv,
      { name: "CORS_ORIGINS", value: '["*"]' },
    ]
  );

  const ingestion = createServiceDeployment(
    "ingestion-service", namespace,
    "seld/ingestion-service:latest", 8001, opts.replicaCount,
    commonEnv
  );

  const deviceService = createServiceDeployment(
    "device-service", namespace,
    "seld/device-service:latest", 50053, opts.replicaCount,
    commonEnv
  );

  const spatialEngine = createServiceDeployment(
    "spatial-engine", namespace,
    "seld/spatial-engine:latest", 50051, opts.replicaCount,
    commonEnv
  );

  const semanticLayer = createServiceDeployment(
    "semantic-layer", namespace,
    "seld/semantic-layer:latest", 50052, opts.replicaCount,
    commonEnv
  );

  const visionFusion = createServiceDeployment(
    "vision-fusion", namespace,
    "seld/vision-fusion:latest", 8004, opts.replicaCount,
    commonEnv
  );

  const frontend = createServiceDeployment(
    "frontend", namespace,
    "seld/frontend:latest", 3000, opts.replicaCount,
    []
  );

  return {
    apiGateway: apiGateway.service,
    ingestion: ingestion.service,
    deviceService: deviceService.service,
    spatialEngine: spatialEngine.service,
    semanticLayer: semanticLayer.service,
    visionFusion: visionFusion.service,
    frontend: frontend.service,
  };
}
