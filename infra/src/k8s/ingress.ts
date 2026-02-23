import * as k8s from "@pulumi/kubernetes";
import * as pulumi from "@pulumi/pulumi";

interface IngressOptions {
  apiGatewayService: k8s.core.v1.Service;
  frontendService: k8s.core.v1.Service;
}

export function deployIngress(namespace: string, opts: IngressOptions) {
  const ingress = new k8s.networking.v1.Ingress("seld-ingress", {
    metadata: {
      namespace,
      name: "seld-ingress",
      annotations: {
        "nginx.ingress.kubernetes.io/proxy-body-size": "100m",
        "nginx.ingress.kubernetes.io/proxy-read-timeout": "300",
        "nginx.ingress.kubernetes.io/websocket-services": "api-gateway",
      },
    },
    spec: {
      ingressClassName: "nginx",
      rules: [
        {
          host: "seld.local",
          http: {
            paths: [
              {
                path: "/api",
                pathType: "Prefix",
                backend: {
                  service: {
                    name: "api-gateway",
                    port: { number: 8000 },
                  },
                },
              },
              {
                path: "/",
                pathType: "Prefix",
                backend: {
                  service: {
                    name: "frontend",
                    port: { number: 3000 },
                  },
                },
              },
            ],
          },
        },
      ],
    },
  });

  return {
    apiUrl: pulumi.interpolate`http://seld.local/api`,
    frontendUrl: pulumi.interpolate`http://seld.local`,
    ingress,
  };
}
