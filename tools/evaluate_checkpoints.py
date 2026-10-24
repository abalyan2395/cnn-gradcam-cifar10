"""Evaluate saved CNN checkpoints on the official CIFAR-10 splits.

This utility evaluates trained models without changing their weights.
Dataset download is optional. Each evaluation writes a separate metrics file.
"""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms


def evaluate(model, dataset, batch_size):
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    totals = torch.zeros(10, dtype=torch.long)
    correct = torch.zeros(10, dtype=torch.long)
    model.eval()
    with torch.inference_mode():
        for images, labels in loader:
            scores = model(images)
            if not torch.isfinite(scores).all():
                raise ValueError("Nonfinite output encountered")
            predictions = scores.argmax(dim=1)
            totals += torch.bincount(labels, minlength=10)
            correct += torch.bincount(labels[predictions == labels], minlength=10)
    return {"samples": int(totals.sum()), "correct": int(correct.sum()),
            "accuracy_percent": 100 * correct.sum().item() / totals.sum().item(),
            "per_class": {name: {"samples": int(totals[i]), "correct": int(correct[i]),
                                 "accuracy_percent": 100 * correct[i].item() / totals[i].item() if totals[i] else None}
                          for i, name in enumerate(dataset.classes)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--download", action="store_true", help="Download CIFAR-10 if missing")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output", type=Path, default=Path("runs/evaluation/metrics.json"))
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.output.exists():
        parser.error("Output already exists; select a new --output to preserve prior evaluations")
    root = Path(__file__).resolve().parents[1]
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,) * 3, (0.5,) * 3)])
    datasets = {split: torchvision.datasets.CIFAR10(root=str(args.data_root), train=train,
                download=args.download, transform=transform) for split, train in (("train", True), ("test", False))}
    record = {"kind": "new_saved_checkpoint_evaluation", "device": "cpu",
              "timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "environment": {"python": platform.python_version(), "architecture": platform.machine(),
                              "torch": torch.__version__, "torchvision": torchvision.__version__, "numpy": np.__version__},
              "batch_size": args.batch_size, "models": {}}
    for name, problem in (("baseline_cnn", 1), ("deeper_cnn", 2)):
        source = root / name / f"problem{problem}_cifar10.py"
        tree = ast.parse(source.read_text())
        definition = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Net")
        scope = {"torch": torch, "nn": nn, "F": F}
        exec(compile(ast.Module(body=[definition], type_ignores=[]), str(source), "exec"), scope)
        model = scope["Net"]()
        checkpoint = root / name / "cifar_net.pth"
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        if not all(torch.isfinite(t).all() for t in state.values()):
            raise ValueError("Nonfinite checkpoint tensor encountered")
        entry = {"checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                 "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
        for split, dataset in datasets.items():
            entry[split] = evaluate(model, dataset, args.batch_size)
            print(f"{name} {split}: {entry[split]['accuracy_percent']:.2f}% ({entry[split]['samples']} images)", flush=True)
        record["models"][name] = entry
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        json.dump(record, output, indent=2)
        output.write("\n")
    print(f"Saved a new evaluation record to {args.output}")


if __name__ == "__main__":
    main()
