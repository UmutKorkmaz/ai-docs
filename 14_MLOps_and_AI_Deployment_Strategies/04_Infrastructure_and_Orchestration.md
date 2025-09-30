# Module 4: Infrastructure and Orchestration

## Navigation
- **← Previous**: [03_Advanced_Deployment_Strategies.md](03_Advanced_Deployment_Strategies.md)
- **→ Next**: [05_Monitoring_and_Observability.md](05_Monitoring_and_Observability.md)
- **↑ Up**: [README.md](README.md)

## Overview

Infrastructure and orchestration are critical components of modern MLOps, providing the foundation for scalable, reliable, and efficient machine learning systems. This module covers containerization, Kubernetes deployment, service mesh integration, and advanced orchestration patterns.

## Container-Based Deployment

### Advanced Docker Configuration

#### Multi-Stage Docker Build
```dockerfile
# Multi-stage Dockerfile for optimized ML model service
FROM python:3.9-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    cmake \
    libboost-all-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /build

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/

# Install package in development mode
RUN pip install -e ./src/

# Production stage
FROM python:3.9-slim as production

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /bin/bash appuser

# Set working directory
WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --from=builder /build/src /app/src

# Copy models and config
COPY models/ ./models/
COPY config/ ./config/

# Create non-root user
RUN chown -R appuser:appuser /app
USER appuser

# Set environment variables
ENV PYTHONPATH=/app
ENV MODEL_PATH=/app/models/latest_model.pkl
ENV PORT=8080
ENV LOG_LEVEL=INFO

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run application
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8080"]
```

#### GPU-Optimized Dockerfile
```dockerfile
# GPU-optimized Dockerfile for ML inference
FROM nvidia/cuda:11.8-devel-ubuntu20.04 as base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3-pip \
    python3-dev \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3 /usr/bin/python

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY models/ ./models/
COPY config/ ./config/

# Set environment variables
ENV PYTHONPATH=/app
ENV MODEL_PATH=/app/models/latest_model.pkl
ENV PORT=8080
ENV CUDA_VISIBLE_DEVICES=0

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run application
CMD ["python3", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8080"]
```

### Advanced Docker Compose Configuration

#### Production-Ready Docker Compose
```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  # Main model service
  model-service:
    build:
      context: .
      dockerfile: Dockerfile
    image: ml-model-service:1.0.0
    ports:
      - "8080:8080"
      - "9090:9090"
    environment:
      - MODEL_PATH=/app/models/latest_model.pkl
      - LOG_LEVEL=INFO
      - REDIS_HOST=redis
      - REDIS_PORT=6379
      - POSTGRES_HOST=postgres
      - POSTGRES_PORT=5432
      - POSTGRES_USER=mlops
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
      - POSTGRES_DB=mlops_db
      - PROMETHEUS_URL=http://prometheus:9090
      - GRAFANA_URL=http://grafana:3000
    volumes:
      - ./models:/app/models:ro
      - ./logs:/app/logs
      - ./config:/app/config:ro
    depends_on:
      redis:
        condition: service_healthy
      postgres:
        condition: service_healthy
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
        reservations:
          memory: 1G
          cpus: '0.5'
      restart_policy:
        condition: on-failure
        delay: 5s
        max_attempts: 3
        window: 120s
    networks:
      - ml-network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  # Redis cache
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
      - ./config/redis.conf:/etc/redis/redis.conf:ro
    command: redis-server /etc/redis/redis.conf
    networks:
      - ml-network
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 10s

  # PostgreSQL database
  postgres:
    image: postgres:14
    environment:
      - POSTGRES_USER=mlops
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
      - POSTGRES_DB=mlops_db
      - PGDATA=/var/lib/postgresql/data/pgdata
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./scripts/init.sql:/docker-entrypoint-initdb.d/init.sql:ro
      - ./config/postgresql.conf:/etc/postgresql/postgresql.conf:ro
    networks:
      - ml-network
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U mlops -d mlops_db"]
      interval: 30s
      timeout: 10s
      retries: 5
      start_period: 30s

  # Prometheus monitoring
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9091:9090"
    volumes:
      - prometheus_data:/prometheus
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - ./monitoring/alerts.yml:/etc/prometheus/alerts.yml:ro
      - ./monitoring/rules:/etc/prometheus/rules:ro
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--storage.tsdb.retention.time=200h'
      - '--web.enable-lifecycle'
      - '--web.enable-admin-api'
      - '--alertmanager.timeout=10s'
    networks:
      - ml-network
    healthcheck:
      test: ["CMD", "wget", "--no-verbose", "--tries=1", "--spider", "http://localhost:9090/-/healthy"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 30s

  # Grafana dashboards
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD}
      - GF_USERS_ALLOW_SIGN_UP=false
      - GF_INSTALL_PLUGINS=grafana-clock-panel,grafana-simple-json-datasource
      - GF_SMTP_ENABLED=true
      - GF_SMTP_HOST=smtp.gmail.com:587
      - GF_SMTP_USER=${EMAIL_USER}
      - GF_SMTP_PASSWORD=${EMAIL_PASSWORD}
      - GF_SMTP_FROM_ADDRESS=${EMAIL_FROM}
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards:ro
      - ./monitoring/grafana/datasources:/etc/grafana/provisioning/datasources:ro
    depends_on:
      - prometheus
    networks:
      - ml-network
    healthcheck:
      test: ["CMD", "wget", "--no-verbose", "--tries=1", "--spider", "http://localhost:3000/api/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 30s

  # Alert manager
  alertmanager:
    image: prom/alertmanager:latest
    ports:
      - "9093:9093"
    volumes:
      - ./monitoring/alertmanager.yml:/etc/alertmanager/alertmanager.yml:ro
      - alertmanager_data:/alertmanager
    command:
      - '--config.file=/etc/alertmanager/alertmanager.yml'
      - '--storage.path=/alertmanager'
      - '--web.external-url=http://localhost:9093'
      - '--web.route-prefix=/'
    networks:
      - ml-network
    healthcheck:
      test: ["CMD", "wget", "--no-verbose", "--tries=1", "--spider", "http://localhost:9093/-/healthy"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 30s

  # Nginx reverse proxy
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/ssl:/etc/nginx/ssl:ro
      - ./nginx/conf.d:/etc/nginx/conf.d:ro
      - nginx_logs:/var/log/nginx
    depends_on:
      - model-service
    networks:
      - ml-network
    healthcheck:
      test: ["CMD", "wget", "--no-verbose", "--tries=1", "--spider", "http://localhost/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 10s

  # Fluentd logging
  fluentd:
    build:
      context: .
      dockerfile: Dockerfile.fluentd
    volumes:
      - ./logs:/fluentd/log
      - ./fluentd/conf:/fluentd/etc
    networks:
      - ml-network
    depends_on:
      - elasticsearch

  # Elasticsearch for logs
  elasticsearch:
    image: elasticsearch:7.17.0
    environment:
      - discovery.type=single-node
      - "ES_JAVA_OPTS=-Xms512m -Xmx512m"
    volumes:
      - elasticsearch_data:/usr/share/elasticsearch/data
    ports:
      - "9200:9200"
    networks:
      - ml-network

  # Kibana for log visualization
  kibana:
    image: kibana:7.17.0
    environment:
      - ELASTICSEARCH_HOSTS=http://elasticsearch:9200
    ports:
      - "5601:5601"
    depends_on:
      - elasticsearch
    networks:
      - ml-network

volumes:
  redis_data:
    driver: local
  postgres_data:
    driver: local
  prometheus_data:
    driver: local
  grafana_data:
    driver: local
  alertmanager_data:
    driver: local
  nginx_logs:
    driver: local
  elasticsearch_data:
    driver: local

networks:
  ml-network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
```

## Kubernetes Deployment

### Complete Kubernetes Configuration

#### Namespace and Resource Management
```yaml
# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: ml-platform
  labels:
    name: ml-platform
    env: production
  annotations:
    description: "Machine Learning Platform Namespace"

---
# resource-quota.yaml
apiVersion: v1
kind: ResourceQuota
metadata:
  name: ml-platform-quota
  namespace: ml-platform
spec:
  hard:
    requests.cpu: "10"
    requests.memory: 20Gi
    limits.cpu: "20"
    limits.memory: 40Gi
    persistentvolumeclaims: "10"
    requests.storage: "100Gi"
    count/deployments.apps: "10"
    count/replicasets.apps: "20"
    count/pods: "50"
    count/services: "20"
    count/secrets: "30"

---
# limit-range.yaml
apiVersion: v1
kind: LimitRange
metadata:
  name: ml-platform-limits
  namespace: ml-platform
spec:
  limits:
  - default:
      memory: "1Gi"
      cpu: "500m"
    defaultRequest:
      memory: "512Mi"
      cpu: "250m"
    type: Container
```

#### Configuration Management
```yaml
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: ml-model-config
  namespace: ml-platform
  labels:
    app: ml-model
    version: v1
data:
  MODEL_VERSION: "1.0.0"
  LOG_LEVEL: "INFO"
  MAX_BATCH_SIZE: "32"
  INFERENCE_TIMEOUT: "100"
  CACHE_TTL: "3600"
  ENABLE_METRICS: "true"
  METRICS_PORT: "9090"
  HEALTH_CHECK_INTERVAL: "30"
  REDIS_HOST: "redis-service"
  REDIS_PORT: "6379"
  POSTGRES_HOST: "postgres-service"
  POSTGRES_PORT: "5432"
  POSTGRES_DB: "mlops_db"
  PROMETHEUS_URL: "http://prometheus-service:9090"
  GRAFANA_URL: "http://grafana-service:3000"

---
# app-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: ml-app-config
  namespace: ml-platform
data:
  config.yaml: |
    model:
      name: "churn-prediction"
      version: "1.0.0"
      path: "/app/models/latest_model.pkl"

    inference:
      timeout: 100
      max_batch_size: 32
      cache_enabled: true
      cache_ttl: 3600

    monitoring:
      enabled: true
      metrics_port: 9090
      health_check_interval: 30

    database:
      host: "postgres-service"
      port: 5432
      name: "mlops_db"
      pool_size: 10

    redis:
      host: "redis-service"
      port: 6379
      db: 0
```

#### Secrets Management
```yaml
# secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: ml-model-secrets
  namespace: ml-platform
type: Opaque
data:
  DATABASE_URL: cG9zdGdyZXNxbDovL21sb3BzOnlvdXJfcGFzc3dvcmRAcG9zdGdyZXMtc2VydmljZTo1NDMyL21sb3BzX2Ri
  REDIS_PASSWORD: eW91cl9yZWRpc19wYXNzd29yZA==
  API_KEY: eW91cl9hcGlfa2V5
  JWT_SECRET: eW91cl9qd3Rfc2VjcmV0
  SLACK_WEBHOOK_URL: aHR0cHM6L2hvb2tzLnNsYWNrLmNvbS9zZXJ2aWNlcy9UMDhERjVGMjIwL0IwMTZSSUc5MjgvdjJYV3dGWEFwWjJjNk5vYlRkY0I=

---
# tls-secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: ml-tls-secret
  namespace: ml-platform
type: kubernetes.io/tls
data:
  tls.crt: LS0tLS1CRUdJTiBDRVJUSUZJQ0FURS0tLS0t...
  tls.key: LS0tLS1CRUdJTiBQUklWQVRFIEtFWS0tLS0t...
```

#### Deployment Configuration
```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-model-service
  namespace: ml-platform
  labels:
    app: ml-model
    version: v1
    component: model-service
  annotations:
    deployment.kubernetes.io/revision: "1"
    kubernetes.io/change-cause: "Initial deployment"
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 1
      maxSurge: 1
  selector:
    matchLabels:
      app: ml-model
      version: v1
  template:
    metadata:
      labels:
        app: ml-model
        version: v1
        component: model-service
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "9090"
        prometheus.io/path: "/metrics"
        prometheus.io/scheme: "http"
        sidecar.istio.io/inject: "true"
    spec:
      serviceAccountName: ml-service-account
      securityContext:
        fsGroup: 1000
        runAsUser: 1000
        runAsGroup: 1000
      containers:
      - name: model-server
        image: your-registry/ml-model:1.0.0
        imagePullPolicy: IfNotPresent
        ports:
        - containerPort: 8080
          name: http
          protocol: TCP
        - containerPort: 9090
          name: metrics
          protocol: TCP
        envFrom:
        - configMapRef:
            name: ml-model-config
        - secretRef:
            name: ml-model-secrets
        env:
        - name: MODEL_PATH
          value: "/app/models/latest_model.pkl"
        - name: POD_NAME
          valueFrom:
            fieldRef:
              fieldPath: metadata.name
        - name: POD_NAMESPACE
          valueFrom:
            fieldRef:
              fieldPath: metadata.namespace
        - name: POD_IP
          valueFrom:
            fieldRef:
              fieldPath: status.podIP
        - name: NODE_NAME
          valueFrom:
            fieldRef:
              fieldPath: spec.nodeName
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
            nvidia.com/gpu: "1"  # Uncomment if using GPU
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
            scheme: HTTP
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
          successThreshold: 1
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
            scheme: HTTP
          initialDelaySeconds: 5
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 3
          successThreshold: 1
        startupProbe:
          httpGet:
            path: /health
            port: 8080
            scheme: HTTP
          initialDelaySeconds: 10
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 30
          successThreshold: 1
        volumeMounts:
        - name: model-storage
          mountPath: /app/models
          readOnly: true
        - name: config-volume
          mountPath: /app/config
          readOnly: true
        - name: logs-volume
          mountPath: /app/logs
        - name: tmp-volume
          mountPath: /tmp
        securityContext:
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: false
          runAsNonRoot: true
          capabilities:
            drop:
            - ALL
          seccompProfile:
            type: RuntimeDefault
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: model-pvc
      - name: config-volume
        configMap:
          name: ml-app-config
      - name: logs-volume
        emptyDir: {}
      - name: tmp-volume
        emptyDir: {}
      nodeSelector:
        node.kubernetes.io/instance-type: "Standard_DS3_v2"
        accelerator: "nvidia-tesla-t4"  # Uncomment if using GPU
      tolerations:
      - key: "nvidia.com/gpu"
        operator: "Exists"
        effect: "NoSchedule"
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
            - matchExpressions:
              - key: "kubernetes.io/arch"
                operator: In
                values:
                - "amd64"
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 1
            preference:
              matchExpressions:
              - key: "node.kubernetes.io/instance-type"
                operator: In
                values:
                - "Standard_DS3_v2"
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchExpressions:
                - key: app
                  operator: In
                  values:
                  - ml-model
              topologyKey: "kubernetes.io/hostname"
      terminationGracePeriodSeconds: 30
      dnsPolicy: ClusterFirst
      restartPolicy: Always
      schedulerName: default-scheduler
```

#### Service Configuration
```yaml
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: ml-model-service
  namespace: ml-platform
  labels:
    app: ml-model
    service: ml-model-service
  annotations:
    service.beta.kubernetes.io/aws-load-balancer-type: nlb
    service.beta.kubernetes.io/aws-load-balancer-cross-zone-load-balancing-enabled: "true"
    service.beta.kubernetes.io/aws-load-balancer-backend-protocol: http
    service.beta.kubernetes.io/aws-load-balancer-ssl-ports: "443"
    service.beta.kubernetes.io/aws-load-balancer-ssl-negotiation-policy: ELBSecurityPolicy-TLS-1-2-2017-01
    service.beta.kubernetes.io/aws-load-balancer-ssl-cert: arn:aws:acm:us-east-1:123456789012:certificate/12345678-1234-1234-1234-123456789012
spec:
  selector:
    app: ml-model
    version: v1
  ports:
  - name: http
    port: 80
    targetPort: 8080
    protocol: TCP
    appProtocol: http
  - name: https
    port: 443
    targetPort: 8080
    protocol: TCP
    appProtocol: https
  - name: metrics
    port: 9090
    targetPort: 9090
    protocol: TCP
  type: LoadBalancer
  externalTrafficPolicy: Local
  sessionAffinity: None
  sessionAffinityConfig:
    clientIP:
      timeoutSeconds: 10800

---
# headless-service.yaml
apiVersion: v1
kind: Service
metadata:
  name: ml-model-service-headless
  namespace: ml-platform
  labels:
    app: ml-model
    service: ml-model-service-headless
spec:
  clusterIP: None
  selector:
    app: ml-model
    version: v1
  ports:
  - name: http
    port: 8080
    targetPort: 8080
    protocol: TCP
  publishNotReadyAddresses: true
```

#### Autoscaling Configuration
```yaml
# hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ml-model-hpa
  namespace: ml-platform
  labels:
    app: ml-model
    component: autoscaler
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ml-model-service
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  - type: Pods
    pods:
      metric:
        name: inference_latency
        selector:
          matchLabels:
            app: ml-model
      target:
        type: AverageValue
        averageValue: "100"
  - type: External
    external:
      metric:
        name: requests_per_second
        selector:
          matchLabels:
            app: ml-model
      target:
        type: AverageValue
        averageValue: "1000"
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 30
      policies:
      - type: Percent
        value: 100
        periodSeconds: 60
      - type: Pods
        value: 2
        periodSeconds: 60
      selectPolicy: Max
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
      selectPolicy: Min

---
# vpa.yaml
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: ml-model-vpa
  namespace: ml-platform
spec:
  targetRef:
    apiVersion: "apps/v1"
    kind:       "Deployment"
    name:       "ml-model-service"
  updatePolicy:
    updateMode: "Auto"
  resourcePolicy:
    containerPolicies:
    - containerName: "model-server"
      minAllowed:
        cpu: "250m"
        memory: "512Mi"
      maxAllowed:
        cpu: "2000m"
        memory: "4Gi"
      controlledResources: ["cpu", "memory"]
```

#### Network Policy
```yaml
# network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: ml-model-network-policy
  namespace: ml-platform
spec:
  podSelector:
    matchLabels:
      app: ml-model
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    - namespaceSelector:
        matchLabels:
          name: monitoring
    - podSelector:
        matchLabels:
          app: redis
    - podSelector:
        matchLabels:
          app: postgres
    ports:
    - protocol: TCP
      port: 8080
    - protocol: TCP
      port: 9090
  egress:
  - to: []
    ports:
    - protocol: TCP
      port: 53
    - protocol: UDP
      port: 53
  - to:
    - podSelector:
        matchLabels:
          app: redis
    ports:
    - protocol: TCP
      port: 6379
  - to:
    - podSelector:
        matchLabels:
          app: postgres
    ports:
    - protocol: TCP
      port: 5432
  - to:
    - namespaceSelector:
        matchLabels:
          name: monitoring
    ports:
    - protocol: TCP
      port: 9090
```

## Helm Chart for ML Platform

### Chart Structure
```yaml
# Chart.yaml
apiVersion: v2
name: ml-platform
description: A Helm chart for ML model deployment platform
type: application
version: 1.0.0
appVersion: "1.0.0"
keywords:
  - machine-learning
  - mlops
  - model-serving
  - inference
home: https://github.com/your-org/ml-platform
sources:
  - https://github.com/your-org/ml-platform
maintainers:
  - name: ML Team
    email: ml-team@yourcompany.com
dependencies:
  - name: redis
    version: "^17.0.0"
    repository: https://charts.bitnami.com/bitnami
  - name: postgresql
    version: "^12.0.0"
    repository: https://charts.bitnami.com/bitnami
  - name: prometheus
    version: "^19.0.0"
    repository: https://prometheus-community.github.io/helm-charts
  - name: grafana
    version: "^6.0.0"
    repository: https://grafana.github.io/helm-charts
```

### Values Configuration
```yaml
# values.yaml
# Global configuration
global:
  imageRegistry: your-registry.com
  imagePullSecrets:
    - name: registry-credentials
  storageClass: "gp2"

# Model service configuration
modelService:
  enabled: true
  replicaCount: 3

  image:
    repository: ml-model-service
    pullPolicy: IfNotPresent
    tag: "1.0.0"

  service:
    type: LoadBalancer
    port: 80
    targetPort: 8080
    metricsPort: 9090
    annotations:
      service.beta.kubernetes.io/aws-load-balancer-type: nlb

  resources:
    requests:
      memory: "1Gi"
      cpu: "500m"
    limits:
      memory: "2Gi"
      cpu: "1000m"

  autoscaling:
    enabled: true
    minReplicas: 2
    maxReplicas: 10
    targetCPUUtilizationPercentage: 70
    targetMemoryUtilizationPercentage: 80

  config:
    modelVersion: "1.0.0"
    logLevel: "INFO"
    maxBatchSize: 32
    inferenceTimeout: "100"
    cacheTTL: "3600"
    enableMetrics: "true"

  security:
    runAsUser: 1000
    runAsGroup: 1000
    fsGroup: 1000

  nodeSelector:
    node.kubernetes.io/instance-type: "Standard_DS3_v2"

  tolerations: []

  affinity: {}

# Redis configuration
redis:
  enabled: true
  auth:
    enabled: true
    password: ""
  master:
    persistence:
      enabled: true
      size: 8Gi
  metrics:
    enabled: true

# PostgreSQL configuration
postgresql:
  enabled: true
  auth:
    postgresPassword: ""
    database: "mlops_db"
    username: "mlops"
  primary:
    persistence:
      enabled: true
      size: 20Gi
  metrics:
    enabled: true

# Monitoring configuration
prometheus:
  enabled: true
  server:
    persistentVolume:
      enabled: true
      size: 10Gi
    configMapOverrideName: "prometheus-config"
  alertmanager:
    enabled: true
    config:
      global:
        smtp_smarthost: "localhost:587"
        smtp_from: "alerts@yourcompany.com"

grafana:
  enabled: true
  adminPassword: ""
  persistence:
    enabled: true
    size: 10Gi
  datasources:
    datasources.yaml:
      apiVersion: 1
      datasources:
      - name: Prometheus
        type: prometheus
        url: http://ml-platform-prometheus-server:9090
        access: proxy
        isDefault: true

  dashboardProviders:
    dashboardproviders.yaml:
      apiVersion: 1
      providers:
      - name: 'default'
        orgId: 1
        folder: ''
        type: file
        disableDeletion: false
        editable: true
        options:
          path: /var/lib/grafana/dashboards/default

# Ingress configuration
ingress:
  enabled: true
  className: "nginx"
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
  hosts:
    - host: ml-platform.yourcompany.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: ml-platform-tls
      hosts:
        - ml-platform.yourcompany.com

# PVC configuration
persistence:
  enabled: true
  storageClass: "gp2"
  modelStorage:
    size: 50Gi
  logsStorage:
    size: 20Gi
```

### Template Files

#### Deployment Template
```yaml
# templates/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ include "ml-platform.fullname" . }}-model-service
  labels:
    {{- include "ml-platform.labels" . | nindent 4 }}
    app.kubernetes.io/component: model-service
spec:
  replicas: {{ .Values.modelService.replicaCount }}
  selector:
    matchLabels:
      {{- include "ml-platform.selectorLabels" . | nindent 6 }}
      app.kubernetes.io/component: model-service
  template:
    metadata:
      labels:
        {{- include "ml-platform.selectorLabels" . | nindent 8 }}
        app.kubernetes.io/component: model-service
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: {{ .Values.modelService.service.metricsPort | quote }}
    spec:
      {{- with .Values.imagePullSecrets }}
      imagePullSecrets:
        {{- toYaml . | nindent 8 }}
      {{- end }}
      serviceAccountName: {{ include "ml-platform.serviceAccountName" . }}
      securityContext:
        {{- toYaml .Values.modelService.security | nindent 8 }}
      containers:
        - name: model-server
          securityContext:
            {{- toYaml .Values.modelService.security | nindent 12 }}
          image: "{{ .Values.global.imageRegistry }}/{{ .Values.modelService.image.repository }}:{{ .Values.modelService.image.tag | default .Chart.AppVersion }}"
          imagePullPolicy: {{ .Values.modelService.image.pullPolicy }}
          ports:
            - name: http
              containerPort: 8080
              protocol: TCP
            - name: metrics
              containerPort: {{ .Values.modelService.service.metricsPort }}
              protocol: TCP
          env:
            - name: MODEL_PATH
              value: "/app/models/latest_model.pkl"
            {{- range $key, $value := .Values.modelService.config }}
            - name: {{ $key | upper }}
              value: {{ $value | quote }}
            {{- end }}
          resources:
            {{- toYaml .Values.modelService.resources | nindent 12 }}
          livenessProbe:
            httpGet:
              path: /health
              port: http
            initialDelaySeconds: 30
            periodSeconds: 10
          readinessProbe:
            httpGet:
              path: /ready
              port: http
            initialDelaySeconds: 5
            periodSeconds: 5
          volumeMounts:
            - name: model-storage
              mountPath: /app/models
              readOnly: true
      volumes:
        - name: model-storage
          persistentVolumeClaim:
            claimName: {{ include "ml-platform.fullname" . }}-model-pvc
      {{- with .Values.modelService.nodeSelector }}
      nodeSelector:
        {{- toYaml . | nindent 8 }}
      {{- end }}
      {{- with .Values.modelService.affinity }}
      affinity:
        {{- toYaml . | nindent 8 }}
      {{- end }}
      {{- with .Values.modelService.tolerations }}
      tolerations:
        {{- toYaml . | nindent 8 }}
      {{- end }}
```

#### Service Template
```yaml
# templates/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: {{ include "ml-platform.fullname" . }}-model-service
  labels:
    {{- include "ml-platform.labels" . | nindent 4 }}
    app.kubernetes.io/component: model-service
  {{- with .Values.modelService.service.annotations }}
  annotations:
    {{- toYaml . | nindent 4 }}
  {{- end }}
spec:
  type: {{ .Values.modelService.service.type }}
  ports:
    - port: {{ .Values.modelService.service.port }}
      targetPort: http
      protocol: TCP
      name: http
    - port: {{ .Values.modelService.service.metricsPort }}
      targetPort: metrics
      protocol: TCP
      name: metrics
  selector:
    {{- include "ml-platform.selectorLabels" . | nindent 4 }}
    app.kubernetes.io/component: model-service
```

## Service Mesh Integration

### Istio Configuration

#### Virtual Service
```yaml
# istio/virtual-service.yaml
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: ml-model-virtual-service
  namespace: ml-platform
spec:
  hosts:
    - ml-platform.yourcompany.com
  gateways:
    - ml-platform-gateway
  http:
  - match:
    - uri:
        prefix: /api/
    route:
    - destination:
        host: ml-model-service
        subset: v1
      weight: 100
    retries:
      attempts: 3
      perTryTimeout: 2s
      retryOn: gateway-error,connect-failure,refused-stream
    timeout: 30s
  - match:
    - uri:
        prefix: /health
    route:
    - destination:
        host: ml-model-service
        subset: v1
      weight: 100
    retries:
      attempts: 3
      perTryTimeout: 1s
      retryOn: gateway-error,connect-failure,refused-stream
    timeout: 5s
```

#### Destination Rule
```yaml
# istio/destination-rule.yaml
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: ml-model-destination-rule
  namespace: ml-platform
spec:
  host: ml-model-service
  trafficPolicy:
    connectionPool:
      tcp:
        maxConnections: 100
        connectTimeout: 30s
        tcpKeepalive:
          time: 7200s
          interval: 75s
      http:
        http1MaxPendingRequests: 100
        http2MaxRequests: 1000
        maxRetries: 3
        idleTimeout: 90s
        h2UpgradePolicy: UPGRADE
    outlierDetection:
      consecutiveGatewayErrors: 5
      consecutive5xxErrors: 5
      interval: 30s
      baseEjectionTime: 30s
      maxEjectionPercent: 50
      minHealthPercent: 50
      splitExternalLocalOriginErrors: true
  subsets:
  - name: v1
    labels:
      version: v1
    trafficPolicy:
      connectionPool:
        tcp:
          maxConnections: 50
        http:
          http1MaxPendingRequests: 50
          maxRequestsPerConnection: 1
          maxRetries: 2
```

## Key Takeaways

### Infrastructure Components
1. **Containerization**: Docker and multi-stage builds for optimization
2. **Orchestration**: Kubernetes deployment and management
3. **Service Mesh**: Istio for advanced networking and traffic management
4. **Configuration**: Helm charts for reusable deployments
5. **Security**: Network policies, RBAC, and secrets management

### Best Practices
- **Immutability**: Use immutable containers and infrastructure
- **Scalability**: Auto-scaling and resource management
- **Monitoring**: Comprehensive observability and metrics
- **Security**: Multi-layered security approach
- **GitOps**: Infrastructure as Code with version control

### Common Challenges
- **Complexity**: Managing complex distributed systems
- **Resource Management**: Efficient resource utilization
- **Networking**: Service discovery and communication
- **Configuration**: Managing configuration across environments
- **Compliance**: Meeting regulatory requirements

---

## Next Steps

Continue to [Module 5: Monitoring and Observability](05_Monitoring_and_Observability.md) to learn about comprehensive monitoring and observability strategies for ML systems.

## Quick Reference

### Key Concepts
- **Containerization**: Docker containers for application packaging
- **Orchestration**: Kubernetes for container orchestration
- **Service Mesh**: Istio for advanced networking
- **Helm Charts**: Reusable Kubernetes packages
- **Infrastructure as Code**: Version-controlled infrastructure

### Essential Tools
- **Docker**: Container platform
- **Kubernetes**: Container orchestration
- **Helm**: Kubernetes package manager
- **Istio**: Service mesh
- **Prometheus**: Monitoring system

### Common Patterns
- **Multi-Stage Build**: Optimized container builds
- **Rolling Deployment**: Zero-downtime deployments
- **Auto-Scaling**: Dynamic resource allocation
- **Service Mesh**: Advanced traffic management
- **GitOps**: Infrastructure version control