"""Train a small CIFAR-10 CNN on CPU and record its test accuracy.

Usage: python -m tools.train_cifar_cnn [--epochs 8] [--out models/cifar]
Writes models/cifar/cifar_cnn.pt and results/train_cifar.json.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from tools import envinfo

ROOT = Path(__file__).resolve().parent.parent
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


class CifarCNN(nn.Module):
    """4 conv layers (BatchNorm + ReLU), 2 max-pools, global average pool, one FC layer."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()

        def block(cin: int, cout: int) -> list[nn.Module]:
            return [nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU()]

        self.features = nn.Sequential(
            *block(3, 32),
            *block(32, 32),
            nn.MaxPool2d(2),
            *block(32, 64),
            *block(64, 64),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.features(x))


def cifar_transform() -> transforms.Compose:
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def load_cifar(root: Path, train: bool) -> Dataset:
    tf = cifar_transform()
    if train:
        tf = transforms.Compose(
            [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), tf]
        )
    return datasets.CIFAR10(str(root), train=train, download=True, transform=tf)


def evaluate(model: nn.Module, loader: DataLoader) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            correct += int((model(x).argmax(1) == y).sum())
            total += len(y)
    return correct / total


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", type=Path, default=ROOT / "data")
    ap.add_argument("--out", type=Path, default=ROOT / "models" / "cifar")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train_dl = DataLoader(load_cifar(args.data, True), args.batch_size, shuffle=True)
    test_dl = DataLoader(load_cifar(args.data, False), 500)

    model = CifarCNN()
    opt = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4, nesterov=True
    )
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, args.lr, epochs=args.epochs, steps_per_epoch=len(train_dl)
    )
    history = []
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        for x, y in train_dl:
            opt.zero_grad()
            nn.functional.cross_entropy(model(x), y).backward()
            opt.step()
            sched.step()
        acc = evaluate(model, test_dl)
        history.append({"epoch": epoch + 1, "test_accuracy": acc})
        print(f"epoch {epoch + 1}/{args.epochs} test_acc={acc:.4f} ({time.time() - t0:.0f}s)")

    args.out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.out / "cifar_cnn.pt")
    result = {
        "env": envinfo.collect("python -m tools.train_cifar_cnn " + shlex.join(sys.argv[1:])),
        "what": "FP32 PyTorch test accuracy (10000 CIFAR-10 test images)",
        "epochs": args.epochs,
        "seed": args.seed,
        "final_test_accuracy": history[-1]["test_accuracy"],
        "history": history,
    }
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "train_cifar.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
