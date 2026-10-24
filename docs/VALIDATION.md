# Validation

## Full-dataset checkpoint evaluation

Both saved models were evaluated on the official CIFAR-10 splits on October 23, 2026: 50,000 training and 10,000 test images per model.

| Model | Training accuracy | Test accuracy |
|---|---:|---:|
| Baseline CNN | 78.584% | 61.87% |
| Deeper CNN | 57.788% | 53.56% |

The complete record is in `results/checkpoint_evaluation.json`. It includes source/checkpoint hashes, sample counts, per-class accuracies, and environment versions. The per-class counts add up to the overall counts. The evaluated accuracies match the original results for these models.

Evaluation ran on CPU using native arm64 Python 3.13.9, PyTorch 2.14.0, torchvision 0.29.0, NumPy 2.5.3, and batch size 256. Loss curves show the original 20-epoch training runs. Accuracies were confirmed by evaluating the saved models on CIFAR-10.

## Model and Grad-CAM checks

- Each checkpoint loads into its matching model architecture on CPU using `weights_only=True`.
- The baseline has 62,006 parameters; the deeper model has 16,656.
- Synthetic inputs of shape `(4, 3, 32, 32)` produce finite `(4, 10)` logits and finite parameter gradients.
- The baseline's manual Grad-CAM forward path matches `Net.forward()`.
- Both Grad-CAM calculations produce finite, nonnegative normalized maps of shape `(32, 32)`.
- Both training scripts complete 20 epochs on a small synthetic dataset with all 10 classes, using 20 training and 10 test images. These tests check training, saving/loading, evaluation, and figure generation. The accuracy results above use the full CIFAR-10 dataset.

Run the offline checks from the repository root:

```bash
python tools/verify_artifacts.py
```

The checker uses a fixed seed for synthetic inputs. It extracts model definitions and the Grad-CAM calculation from local source files to avoid executing top-level training code. The utility executes local model definitions, so use it only with trusted source files.

## Reproduce checkpoint evaluation

```bash
python tools/evaluate_checkpoints.py --data-root data --download
```

Results are written to `runs/evaluation/metrics.json`. For another evaluation, specify a new output path with `--output`. Omit `--download` when the dataset is already available.

## Reproducibility notes

Training, checkpoint evaluation, and synthetic tests serve different purposes:

- Training adjusts model weights and produces the loss curves.
- Checkpoint evaluation loads the trained weights and measures accuracy on CIFAR-10.
- Synthetic tests check that code paths, tensor shapes, gradients, and file outputs work.

Training does not set a random seed, and the checkpoints do not include optimizer state. A new training run may produce different weights and accuracy. The original training process has not been rerun to reproduce the same weights.

The models run on CPU. A fresh dependency installation, GPU/MPS execution, memory use, and full training runtime have not been tested. Accuracy checks and Grad-CAM heatmaps alone do not establish explanation faithfulness or fairness.
