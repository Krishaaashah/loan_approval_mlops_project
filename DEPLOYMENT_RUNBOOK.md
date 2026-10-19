# Deployment Runbook

This runbook describes the release flow for the loan approval API in AWS.
The current target region is `ap-south-1`.

## Configured Resources

- Deployment target: one free-tier-sized `t3.micro` in `ap-south-1a`, instance `i-02fd037d73f8a7b68`, currently running at `43.205.216.66`.
- DVC remote: `s3://loan-approval-mlops-049681811107-ap-south-1/dvcstore`.
- ECR repository: `049681811107.dkr.ecr.ap-south-1.amazonaws.com/loan-approval-mlops`.
- CloudWatch log group: `/mlops/loan-approval`, retained for 7 days.
- Security group: `sg-0b199416088c16b23`; ports 22 and 80 are restricted to the current client IP.
- Current image: `release-local-20261019-v2`, digest `sha256:a78b848f242dbd4648c2683e96c69d0c5870ad3a25f5455d62410f1944aafa68`.
- ECR tags are immutable and the lifecycle policy retains at most three `release-*` images.
- The manual GitHub release workflow expects the repository secret `AWS_ROLE_TO_ASSUME` for GitHub OIDC.

These resources are intentionally small. The ECR, S3, CloudWatch, IAM, security group, and EC2 resources are configured and the API has passed live health, readiness, prediction, `/docs`, and `/metrics` checks.

## GitHub Release

After the repository is connected, run **Build and publish release image** from the Actions tab.
It rebuilds the synthetic dataset and model, builds the container, and pushes an immutable
`release-<commit-sha>` tag to ECR. The workflow cannot run until `AWS_ROLE_TO_ASSUME` points
to a least-privilege GitHub OIDC role.

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

For an EC2 Docker deployment, stop the current container, start the previous
image tag, and verify the same health and prediction endpoints before reopening
traffic.

## Secrets and Permissions

- Use an EC2 instance role for S3, CloudWatch, and ECR read access.
- Use GitHub OIDC or short-lived credentials for CI publishing.
- Keep AWS keys, registry tokens, and MLflow credentials out of source code,
  Docker build arguments, and image layers.
- Restrict inbound traffic to the required client or load-balancer ports.
- Do not use the AWS account root identity for routine deployment; switch the CLI to a least-privilege user or role before the release.
