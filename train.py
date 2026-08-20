import argparse
import os
import random
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="PyTorch GPU training demo")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--input-dim", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output-model", default=None, help="Save trained model weights to path")
    return parser.parse_args(argv)


def is_distributed():
    """True when launched under a PyTorchJob (or torchrun) with WORLD_SIZE > 1."""
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def build_model(args, device):
    return nn.Sequential(
        nn.Linear(args.input_dim, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
    ).to(device)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    distributed = is_distributed()
    rank = int(os.environ.get("RANK", "0"))

    if distributed:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        device = torch.device(f"cuda:{rank % torch.cuda.device_count()}" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    is_main = rank == 0
    if is_main:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        print(f"Training on: {device} (seed={args.seed}, world_size={world_size})")

    model = build_model(args, device)
    if distributed:
        model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None)

    data = torch.randn(args.batch_size, args.input_dim, device=device)
    target = torch.randn(args.batch_size, 1, device=device)

    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    if is_main:
        print(f"Epochs: {args.epochs} | LR: {args.lr} | Batch: {args.batch_size}")
        print("-" * 40)

    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if is_main:
            print(f"Epoch [{epoch:2d}/{args.epochs}]  loss: {loss.item():.6f}")

    if is_main:
        print("-" * 40)
        print("Training complete.")

        if args.output_model:
            out_dir = os.path.dirname(args.output_model)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            state_dict = model.module.state_dict() if distributed else model.state_dict()
            torch.save(state_dict, args.output_model)
            print(f"Model saved to {args.output_model}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
