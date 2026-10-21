# Deployment Runbook

This runbook describes the release flow for the loan approval API in AWS.
The current target region is `ap-south-1`.

## Configured Resources

- Deployment target: one free-tier-sized `t3.micro` in `ap-south-1a`, instance `i-02fd037d73f8a7b68`, currently running at `43.205.216.66`.
- DVC remote: `s3://loan-approval-mlops-049681811107-ap-south-1/dvcstore`.
- ECR repository: `049681811107.dkr.ecr.ap-south-1.amazonaws.com/loan-approval-mlops`.
- CloudWatch log group: `/mlops/loan-approval`, retained for 7 days.
- Security group: `sg-0b199416088c16b23`; ports 22 and 80 are restricted to the current client IP.
- Current image: `release-86c9e369ef587b6ff8968b11ab434bcdd22d9744`, digest `sha256:b699d6fbad851eb58987b21f28061682500f5d8ceee663b05334ff7383bc1283`.
- ECR tags are immutable and the lifecycle policy retains at most three `release-*` images.
- The manual GitHub release workflow expects the repository secret `AWS_ROLE_TO_ASSUME` for GitHub OIDC and deploys through SSM.

These resources are intentionally small. The ECR, S3, CloudWatch, IAM, security group, and EC2 resources are configured and the API has passed live health, readiness, prediction, `/docs`, and `/metrics` checks. The deployment also starts resource-limited `loan-prometheus` and `loan-grafana` containers with persistent Docker volumes. Grafana is available at `http://<EC2_PUBLIC_IP>:3000` only from the configured client IP, and its provisioned dashboard reads the API over the private Docker network.

Prediction inputs for drift are written as JSON events under the private S3 prefix
`production/predictions/`. The EC2 role can write only that prefix, while the GitHub
OIDC role can list/read only that prefix for the scheduled drift workflow.

## GitHub Release

Run **Build and publish release image** from the Actions tab. It rebuilds the synthetic
dataset and model, builds the container, pushes an immutable `release-<commit-sha>` tag
to ECR, and sends an `AWS-RunShellScript` command to the configured EC2 instance. The
instance pulls that exact tag, restarts `loan-approval-api`, and verifies `/health` and
`/ready` locally before the workflow succeeds.

The successful reference deployment is GitHub Actions run `39638313606` for commit
`86c9e36`.

## Release Rules

- Build the image from the tested commit.
- Tag the image with the commit SHA, for example `mlops:<git-sha>`.
- Push that immutable tag to ECR.
- Deploy the exact tag or image digest. Never deploy `latest`.
- Verify `/health` and `/ready` before sending client traffic.

## Rollback

1. Identify the last known-good ECR tag or digest.
2. Update the deployment to that exact image reference.
3. Wait for the readiness probe to pass.
4. Send a test request to `/prediction_api`.
5. Record the failed tag, rollback tag, reason, and verification result.

For Kubernetes, the rollback command is:

```powershell
kubectl rollout undo deployment/mlops
kubectl rollout status deployment/mlops
```

For an EC2 Docker deployment, send the previous image tag through SSM, then verify the
same health and prediction endpoints before reopening traffic. The release workflow
uses this same pattern for the current image.

## Secrets and Permissions

- Use an EC2 instance role for production-event writes, CloudWatch, and ECR read access.
- Use GitHub OIDC or short-lived credentials for CI publishing.
- The GitHub role can push only to the loan-approval ECR repository, send/read SSM
  deployment commands for the single configured instance, and read only the
  `production/predictions/` S3 prefix.
- Keep AWS keys, registry tokens, and MLflow credentials out of source code,
  Docker build arguments, and image layers.
- Restrict inbound traffic to the required client or load-balancer ports.
- Do not use the AWS account root identity for routine deployment; switch the CLI to a least-privilege user or role before the release.
