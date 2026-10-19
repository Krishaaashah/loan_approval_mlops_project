#!/bin/bash
set -euo pipefail

exec > >(tee /var/log/loan-approval-bootstrap.log | logger -t loan-approval-bootstrap -s 2>/dev/console) 2>&1

REGION="ap-south-1"
IMAGE="049681811107.dkr.ecr.ap-south-1.amazonaws.com/loan-approval-mlops:release-local-20261019-v2"

dnf install -y docker
systemctl enable --now docker

aws ecr get-login-password --region "$REGION" \
  | docker login --username AWS --password-stdin "049681811107.dkr.ecr.$REGION.amazonaws.com"
docker pull "$IMAGE"
docker rm -f loan-approval-api || true
docker run --detach \
  --name loan-approval-api \
  --restart unless-stopped \
  --publish 80:8005 \
  --env MLOPS_UPLOAD_PREDICTIONS=false \
  --env MLOPS_ALLOWED_ORIGINS="*" \
  --log-driver=awslogs \
  --log-opt awslogs-region="$REGION" \
  --log-opt awslogs-group=/mlops/loan-approval \
  --log-opt awslogs-create-group=false \
  --log-opt awslogs-stream=loan-api \
  "$IMAGE"

touch /var/lib/loan-approval-deployed
