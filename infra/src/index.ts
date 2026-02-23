import * as pulumi from "@pulumi/pulumi";
import * as k8s from "@pulumi/kubernetes";

import { createNamespace } from "./k8s/namespace";
import { deployPostgres } from "./k8s/postgres";
import { deployEMQX } from "./k8s/emqx";
import { deployKafka } from "./k8s/kafka";
import { deployServices } from "./k8s/services";
import { deployMonitoring } from "./k8s/monitoring";
import { deployIngress } from "./k8s/ingress";

const config = new pulumi.Config();
const env = config.require("environment");
const namespace = config.require("k8sNamespace");
const replicaCount = config.getNumber("replicaCount") ?? 1;

// Create namespace
const ns = createNamespace(namespace, env);

// Deploy infrastructure services
const postgres = deployPostgres(namespace, { replicaCount });
const emqx = deployEMQX(namespace, { replicaCount });
const kafka = deployKafka(namespace, { replicaCount });

// Deploy application services
const services = deployServices(namespace, {
  replicaCount,
  postgresHost: postgres.host,
  kafkaBootstrap: kafka.bootstrapServers,
  emqxHost: emqx.host,
});

// Deploy monitoring stack
const monitoring = deployMonitoring(namespace);

// Deploy ingress
const ingress = deployIngress(namespace, {
  apiGatewayService: services.apiGateway,
  frontendService: services.frontend,
});

// Exports
export const kubeNamespace = namespace;
export const apiGatewayUrl = ingress.apiUrl;
export const frontendUrl = ingress.frontendUrl;
export const postgresHost = postgres.host;
export const emqxDashboard = emqx.dashboardUrl;
