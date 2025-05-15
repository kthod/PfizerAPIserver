# Quantum Optimization API

This API provides access to a quantum optimization service that uses Compressed VQE for solving optimization problems.

## Deployment

### Local Deployment

1. Build the Docker container:
```bash
docker build -t quantum-optimization-api .
```

2. Run the container:
```bash
docker run -p 8000:8000 quantum-optimization-api
```

### AWS Deployment

1. Build and tag the Docker image:
```bash
docker build -t quantum-optimization-api .
docker tag quantum-optimization-api:latest <your-aws-account-id>.dkr.ecr.<region>.amazonaws.com/quantum-optimization-api:latest
```

2. Push to Amazon ECR:
```bash
aws ecr get-login-password --region <region> | docker login --username AWS --password-stdin <your-aws-account-id>.dkr.ecr.<region>.amazonaws.com
docker push <your-aws-account-id>.dkr.ecr.<region>.amazonaws.com/quantum-optimization-api:latest
```

3. Deploy to EC2 or ECS using the pushed image.

## API Usage

The API exposes a single endpoint:

### POST /optimize

Request body:
```json
{
    "num_jobs": 10,
    "num_tasks": 30,
    "num_machines": 5,
    "p": [1.0, 2.0, 3.0, ...],
    "Jobs": [[1, 1], [1, 2], ...],
    "Brands": ["brand1", "brand2", ...],
    "mg": ["mg1", "mg2", ...],
    "task_machine_mapping": {
        "0": [0],
        "1": [1],
        ...
    },
    "layers": 3,
    "n_measurements": 200000,
    "number_of_experiments": 8,
    "maxiter": 300
}
```

Response:
```json
{
    "solution": "010101...",
    "start_times": [0, 1, 2, ...],
    "assigned_machines": [0, 1, 2, ...],
    "cost_function_value": 123.45
}
```

## Security

The Compressed VQE implementation is protected within the Docker container and is not accessible from outside. The API only exposes the necessary endpoints for optimization while keeping the core implementation secure. 