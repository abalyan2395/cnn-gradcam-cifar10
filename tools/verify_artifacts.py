"""Check local CNN models and checkpoints without running training scripts.

No dataset download or saved artifact writes. Run from the repository root.
The AST extraction avoids import-time training; it is not a sandbox for untrusted code.
"""
import ast
import hashlib
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def main():
    root = Path(__file__).resolve().parents[1]
    torch.manual_seed(0)  # This check only; original experiments were unseeded.
    findings = {}
    for problem in (1, 2):
        folder = root / ("baseline_cnn" if problem == 1 else "deeper_cnn")
        source = folder / f"problem{problem}_cifar10.py"
        tree = ast.parse(source.read_text())
        definition = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Net")
        scope = {"torch": torch, "nn": nn, "F": F}
        exec(compile(ast.Module(body=[definition], type_ignores=[]), str(source), "exec"), scope)
        model = scope["Net"]()
        checkpoint = folder / "cifar_net.pth"
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        assert all(torch.isfinite(t).all() for t in state.values())
        model.eval()
        sample = torch.randn(4, 3, 32, 32)
        logits = model(sample)
        assert logits.shape == (4, 10) and torch.isfinite(logits).all()
        loss = nn.CrossEntropyLoss()(logits, torch.tensor([0, 1, 2, 3]))
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        record = {"parameters": sum(p.numel() for p in model.parameters()),
                  "output_shape": list(logits.shape), "checkpoint_strict_load": True,
                  "finite_forward_backward": True,
                  "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
        if problem == 1:
            model.zero_grad(set_to_none=True)
            # Execute the original calculation section on synthetic input only.
            lines = source.read_text().splitlines()
            start = next(i for i, s in enumerate(lines) if s.startswith("conv1_layer_features="))
            stop = next(i for i, s in enumerate(lines) if s.startswith("image_unnormal="))
            original = sample[:1]
            scope.update({"net": model, "x": original, "classes": tuple(str(i) for i in range(10)), "label": 0})
            exec(compile("\n".join(lines[start:stop]), str(source), "exec"), scope)
            assert torch.allclose(model(original), scope["prediction_scores"])
            for name in ("grad_cam_conv1", "grad_cam_conv2"):
                heatmap = scope[name]
                assert heatmap.shape == (32, 32) and torch.isfinite(heatmap).all()
                assert heatmap.min() >= 0 and heatmap.max() <= 1.000001
            record["gradcam_original_calculation"] = "finite 32x32 maps; manual forward matches Net.forward"
        findings[f"problem{problem}"] = record
    print(json.dumps(findings, indent=2))


if __name__ == "__main__":
    main()
