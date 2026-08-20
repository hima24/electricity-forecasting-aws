# Deploying to AWS (ECS Fargate)

This deploys two containers from the same image:

1. **`electricity-dashboard`** — an ECS **service** (always running) behind
   an Application Load Balancer, serving the Streamlit UI.
2. **`electricity-scheduler`** — an ECS **scheduled task**, triggered every
   hour by **EventBridge Scheduler**, running `python scheduler.py` once and
   exiting.

Both share:
- **AWS Secrets Manager** for `ENTSOE_API_KEY` / `OPENWEATHER_API_KEY`
- **Amazon EFS** for the SQLite database, so data persists across restarts,
  redeploys, and between the two tasks
- **Amazon ECR** for the container image
- **CloudWatch Logs** for both containers' stdout/stderr

Estimated cost: **~$15–30/month** (Fargate vCPU/memory, ALB, EFS, NAT — see
cost notes at the bottom for how to trim this to near-zero for a
resume/demo project).

---

## Prerequisites

- AWS CLI installed and configured (`aws configure`) with an account that
  has permissions for ECS, ECR, EFS, Secrets Manager, IAM, EventBridge, VPC
- Docker installed locally
- Your ENTSO-E and OpenWeatherMap API keys on hand

Set these once — reused in every command below:

```bash
export AWS_REGION=us-east-1
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export APP_NAME=electricity-forecast
```

---

## 1. Push the image to ECR

```bash
aws ecr create-repository --repository-name $APP_NAME --region $AWS_REGION

aws ecr get-login-password --region $AWS_REGION \
  | docker login --username AWS --password-stdin \
    $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

docker build -t $APP_NAME .
docker tag $APP_NAME:latest \
  $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$APP_NAME:latest
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$APP_NAME:latest
```

---

## 2. Store credentials in Secrets Manager

Real keys live **only** here — never in the repo, the image, or plain
environment variables in the task definition.

```bash
aws secretsmanager create-secret \
  --name electricity-forecast/entsoe-api-key \
  --secret-string "YOUR_REAL_ENTSOE_KEY" \
  --region $AWS_REGION

aws secretsmanager create-secret \
  --name electricity-forecast/openweather-api-key \
  --secret-string "YOUR_REAL_OPENWEATHER_KEY" \
  --region $AWS_REGION
```

Note the returned ARNs — you'll reference them in the task definitions
below.

---

## 3. Persistent storage — EFS for the SQLite DB

```bash
# Use your default VPC for simplicity; adjust for a custom VPC.
export VPC_ID=$(aws ec2 describe-vpcs --filters Name=isDefault,Values=true \
  --query 'Vpcs[0].VpcId' --output text --region $AWS_REGION)

aws efs create-file-system \
  --creation-token electricity-forecast-data \
  --encrypted \
  --tags Key=Name,Value=electricity-forecast-data \
  --region $AWS_REGION
# Note the returned FileSystemId (fs-xxxxxxxx)

export EFS_ID=fs-xxxxxxxx   # <- paste the id from above

# Create a mount target in each subnet the ECS tasks will run in
aws ec2 describe-subnets --filters Name=vpc-id,Values=$VPC_ID \
  --query 'Subnets[].SubnetId' --output text --region $AWS_REGION
# For each subnet id returned:
aws efs create-mount-target \
  --file-system-id $EFS_ID \
  --subnet-id <SUBNET_ID> \
  --security-groups <SECURITY_GROUP_ID> \
  --region $AWS_REGION
```

The security group attached to the mount targets must allow inbound NFS
(port 2049) from the ECS tasks' security group.

---

## 4. IAM roles

**Task execution role** (pulls image, reads secrets, writes logs) — reuse
the AWS-managed one and attach Secrets Manager read access:

```bash
aws iam attach-role-policy \
  --role-name ecsTaskExecutionRole \
  --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy

cat > secrets-policy.json << 'EOF'
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": ["secretsmanager:GetSecretValue"],
    "Resource": "arn:aws:secretsmanager:*:*:secret:electricity-forecast/*"
  }]
}
EOF

aws iam put-role-policy \
  --role-name ecsTaskExecutionRole \
  --policy-name ElectricityForecastSecretsAccess \
  --policy-document file://secrets-policy.json
```

---

## 5. ECS cluster

```bash
aws ecs create-cluster --cluster-name electricity-forecast-cluster \
  --region $AWS_REGION
```

---

## 6. Task definition — dashboard (long-running service)

```json
{
  "family": "electricity-dashboard",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "512",
  "memory": "1024",
  "executionRoleArn": "arn:aws:iam::ACCOUNT_ID:role/ecsTaskExecutionRole",
  "volumes": [{
    "name": "efs-data",
    "efsVolumeConfiguration": { "fileSystemId": "EFS_ID" }
  }],
  "containerDefinitions": [{
    "name": "dashboard",
    "image": "ACCOUNT_ID.dkr.ecr.REGION.amazonaws.com/electricity-forecast:latest",
    "command": ["streamlit", "run", "dashboard.py",
                "--server.port=8501", "--server.address=0.0.0.0"],
    "portMappings": [{ "containerPort": 8501 }],
    "mountPoints": [{ "sourceVolume": "efs-data", "containerPath": "/app/data" }],
    "secrets": [
      { "name": "ENTSOE_API_KEY",
        "valueFrom": "arn:aws:secretsmanager:REGION:ACCOUNT_ID:secret:electricity-forecast/entsoe-api-key" },
      { "name": "OPENWEATHER_API_KEY",
        "valueFrom": "arn:aws:secretsmanager:REGION:ACCOUNT_ID:secret:electricity-forecast/openweather-api-key" }
    ],
    "logConfiguration": {
      "logDriver": "awslogs",
      "options": {
        "awslogs-group": "/ecs/electricity-dashboard",
        "awslogs-region": "REGION",
        "awslogs-stream-prefix": "dashboard",
        "awslogs-create-group": "true"
      }
    }
  }]
}
```

Replace the placeholders (`ACCOUNT_ID`, `REGION`, `EFS_ID`), save as
`task-def-dashboard.json`, then register and run it:

```bash
aws ecs register-task-definition \
  --cli-input-json file://task-def-dashboard.json --region $AWS_REGION

aws ecs create-service \
  --cluster electricity-forecast-cluster \
  --service-name electricity-dashboard \
  --task-definition electricity-dashboard \
  --desired-count 1 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[<SUBNET_ID>],securityGroups=[<SG_ID>],assignPublicIp=ENABLED}" \
  --region $AWS_REGION
```

For a public URL, put an **Application Load Balancer** in front of this
service (target group on port 8501, health check path `/`). AWS docs:
[ECS service with ALB](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/service-load-balancing.html).

---

## 7. Task definition — scheduler (runs hourly, exits)

Same as above but:
- `"command": ["python", "scheduler.py"]`
- No `portMappings`
- `"awslogs-stream-prefix": "scheduler"`

Save as `task-def-scheduler.json` and register it:

```bash
aws ecs register-task-definition \
  --cli-input-json file://task-def-scheduler.json --region $AWS_REGION
```

Don't create a *service* for this one — EventBridge Scheduler will launch
it as a one-off task every hour.

---

## 8. EventBridge Scheduler — hourly trigger

```bash
aws scheduler create-schedule \
  --name electricity-forecast-hourly \
  --schedule-expression "rate(1 hour)" \
  --flexible-time-window "Mode=OFF" \
  --target "{
    \"Arn\": \"arn:aws:ecs:$AWS_REGION:$AWS_ACCOUNT_ID:cluster/electricity-forecast-cluster\",
    \"RoleArn\": \"arn:aws:iam::$AWS_ACCOUNT_ID:role/ecsTaskExecutionRole\",
    \"EcsParameters\": {
      \"TaskDefinitionArn\": \"arn:aws:ecs:$AWS_REGION:$AWS_ACCOUNT_ID:task-definition/electricity-scheduler\",
      \"LaunchType\": \"FARGATE\",
      \"NetworkConfiguration\": {
        \"awsvpcConfiguration\": {
          \"Subnets\": [\"<SUBNET_ID>\"],
          \"SecurityGroups\": [\"<SG_ID>\"],
          \"AssignPublicIp\": \"ENABLED\"
        }
      }
    }
  }" \
  --region $AWS_REGION
```

This replaces `scheduler.py`'s own `BlockingScheduler` loop — in the cloud,
EventBridge is the scheduler, and each invocation runs `hourly_job()` once
via `python scheduler.py` and exits, which is a more idiomatic serverless
pattern than a long-running polling loop.

---

## 9. Verify

```bash
# Watch the dashboard service come up
aws ecs describe-services --cluster electricity-forecast-cluster \
  --services electricity-dashboard --region $AWS_REGION

# Tail logs
aws logs tail /ecs/electricity-dashboard --follow --region $AWS_REGION
aws logs tail /ecs/electricity-scheduler --follow --region $AWS_REGION
```

Open the ALB's DNS name in a browser to see the live dashboard.

---

## Cost-control notes (for a demo/resume project, not production)

- **Fargate Spot** for the scheduler task cuts its cost significantly since
  it's short-lived and tolerant of interruption — add
  `"capacityProviderStrategy": [{"capacityProvider": "FARGATE_SPOT", "weight": 1}]`.
- Skip the ALB and just use the task's public IP + security group rule for
  a demo (saves ~$16/mo) — trade-off is a changing IP on every restart.
- Scale the dashboard service to `desired-count 0` when not actively
  demoing it, and back to `1` before an interview/demo.
- EFS is billed per GB-month and this database stays tiny (tens of MB), so
  it's a negligible cost either way.

## What to say about this in an interview

*"I containerized the training pipeline and live dashboard with Docker,
deployed the dashboard as an ECS Fargate service behind an ALB, and used
EventBridge Scheduler to run the hourly data-collection job as a separate
Fargate task instead of a long-running polling loop. Credentials are
managed through Secrets Manager and injected as task environment
variables — nothing sensitive is in the image or the repo. The SQLite
database lives on an EFS volume shared between both tasks so state
persists across deploys."*
