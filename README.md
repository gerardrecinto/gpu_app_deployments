# GPU App Deployments

![CI](https://github.com/gerardrecinto/gpu-ml-deployments/actions/workflows/ci.yml/badge.svg)
![Release](https://github.com/gerardrecinto/gpu-ml-deployments/actions/workflows/release.yml/badge.svg)
![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![Kubernetes](https://img.shields.io/badge/Kubernetes-GPU%20Workloads-326CE5?logo=kubernetes&logoColor=white)
![CUDA 12.1](https://img.shields.io/badge/CUDA-12.1-76B900?logo=nvidia&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-22c55e)

![GPU App Deployments logo](docs/assets/logo.svg)

![Demo](docs/assets/demo.gif)

Containerized PyTorch training workloads deployed on Kubernetes with GPU scheduling, plus Slurm support for HPC clusters.

Commercial angle and consulting hooks: [docs/go-to-market.md](docs/go-to-market.md).

## Docker

```bash
docker pull ghcr.io/gerardrecinto/gpu-ml-deployments:latest
```

## Files

| File | Purpose |
|---|---|
| `train.py` | PyTorch training loop with argparse — runs on CUDA or CPU |
| `Dockerfile` | CUDA 12.1 + cuDNN 8 image, torch/torchvision installed from the cu121 wheel index |
| `requirements-dev.txt` | CPU-only torch + pytest for local dev and CI (no CUDA needed) |
| `pytorch-gpu-deployment.yaml` | K8s Deployment: `nvidia.com/gpu: 1` per pod, resource requests + limits |
| `pytorch-distributed-job.yaml` | Kubeflow `PyTorchJob` for multi-node distributed training |
| `pytorch_job.sh` | Slurm batch job script |
| `gres.conf` / `slurm.conf` | Slurm GPU resource config |

## Running on Kubernetes

**1. Install the NVIDIA device plugin:**

```bash
# GPU Operator (recommended for production)
kubectl create -f https://operatorhub.io/install/nvidia-gpu-operator.yaml

# or lightweight daemonset
kubectl apply -f https://github.com/NVIDIA/k8s-device-plugin/raw/main/deployments/k8s-device-plugin-daemonset.yaml
```

**2. Build and push the image:**

```bash
docker build -t gerardrecinto/pytorch-gpu:latest .
docker push gerardrecinto/pytorch-gpu:latest
```

**3. Deploy:**

```bash
kubectl create namespace gpu-workloads
kubectl apply -f pytorch-gpu-deployment.yaml
kubectl get pods -n gpu-workloads -l app=pytorch-gpu-app
kubectl logs -f <pod-name> -n gpu-workloads
```

## Running on Slurm (HPC)

```bash
sbatch pytorch_job.sh
squeue -u $USER
```

The `gres.conf` declares the GPU device files per node. `slurm.conf` sets the `Gres=gpu:2` count for that node — the two must agree on the number of GPUs.

## train.py flags

```bash
# Set a fixed random seed for reproducible runs
python train.py --seed 42

# Save the trained model to a specific path
python train.py --output-model /models/run1.pt

# Both together
python train.py --seed 42 --output-model /models/run1.pt
```

`--seed` sets `torch.manual_seed` and `random.seed` before training starts. `--output-model` writes the final state dict to the given path after training completes.

## Local dev / tests

```bash
pip install -r requirements-dev.txt
pytest -v
```

`requirements-dev.txt` installs CPU-only torch, so this works without a GPU or CUDA toolchain.

## Multi-node distributed training

`pytorch-distributed-job.yaml` runs `train.py` across multiple pods using the [Kubeflow Training Operator](https://github.com/kubeflow/training-operator)'s `PyTorchJob` CRD instead of the single `Deployment`:

```bash
kubectl apply -f pytorch-distributed-job.yaml
kubectl get pytorchjobs -n gpu-workloads
kubectl logs -f pytorch-distributed-master-0 -n gpu-workloads
```

Requires the Training Operator installed in the cluster (`kubectl apply -k "github.com/kubeflow/training-operator/manifests/overlays/standalone"`).

## CI/CD Pipeline

The `Jenkinsfile` at the repo root uses [groovylibrary](https://github.com/gerardrecinto/groovylibrary) to:

1. Build the Docker image on commit to `main`
2. Push to registry with commit SHA and `latest` tags
3. Apply the K8s manifest to the `gpu-workloads` namespace
4. Wait on rollout status before marking the build green

```bash
# Restart a running deployment manually
kubectl rollout restart deployment/pytorch-gpu-deployment -n gpu-workloads
```

## Notes

- `train.py` falls back to CPU automatically if no GPU is detected (`torch.cuda.is_available()`)
- Each pod requests 1 GPU — you need at least 2 GPU nodes for `replicas: 2`
- For multi-node distributed training, use `pytorch-distributed-job.yaml` (see [Multi-node distributed training](#multi-node-distributed-training)) instead of `pytorch-gpu-deployment.yaml`
