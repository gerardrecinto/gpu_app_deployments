# gpu_app_deployments
# Install NVIDIA GPU Operator
kubectl create -f https://operatorhub.io/install/nvidia-gpu-operator.yaml
# or NVIDIA device plugin for K8s
kubectl apply -f https://github.com/NVIDIA/k8s-device-plugin/raw/main/deployments/k8s-device-plugin-daemonset.yaml

# Submit job to Slurm scheduler:
sbatch pytorch_job.sh

