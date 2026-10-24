# Results

## Classification accuracy

Both models were evaluated on the complete official CIFAR-10 splits: 50,000 training and 10,000 test images. Each class contains 5,000 training and 1,000 test examples.

| Model | Train correct / total | Training accuracy | Test correct / total | Test accuracy |
|---|---:|---:|---:|---:|
| Baseline CNN | 39,292 / 50,000 | 78.584% | 6,187 / 10,000 | 61.87% |
| Deeper CNN | 28,894 / 50,000 | 57.788% | 5,356 / 10,000 | 53.56% |

The baseline outperforms the deeper model by 8.31 percentage points on the test split. The deeper model reduces the representation entering the classifier from 400 values to 10, while reducing total parameters from 62,006 to 16,656. The comparison changes pooling, representation size, and capacity as well as depth.

## Per-class test accuracy

| Class | Baseline CNN | Deeper CNN |
|---|---:|---:|
| airplane | 66.2% | 46.3% |
| automobile | 74.9% | 69.1% |
| bird | 52.1% | 37.7% |
| cat | 36.2% | 36.3% |
| deer | 51.1% | 47.8% |
| dog | 45.7% | 35.1% |
| frog | 79.7% | 60.6% |
| horse | 62.8% | 66.3% |
| ship | 80.1% | 69.8% |
| truck | 69.9% | 66.6% |

The baseline's highest test accuracy is for ships (80.1%). Cats and dogs are among its more difficult classes, at 36.2% and 45.7% respectively. The Grad-CAM example examines a cat predicted as a dog.

## Training loss

| Model | Epoch 1 mean loss | Epoch 20 mean loss |
|---|---:|---:|
| Baseline CNN | 1.690 | 0.638 |
| Deeper CNN | 1.906 | 1.238 |

Loss curves show the original 20-epoch training runs. Accuracies were confirmed by evaluating the saved models on CIFAR-10.

## Evaluation setup

The saved models were evaluated on October 23, 2026 using CPU, native arm64 Python 3.13.9, PyTorch 2.14.0, torchvision 0.29.0, NumPy 2.5.3, and batch size 256. Model definitions and weights are identified by SHA-256 hashes in the evaluation record.

## Result files

| File | Contents |
|---|---|
| `results/checkpoint_evaluation.json` | Evaluation configuration, source/checkpoint hashes, exact counts, and overall/per-class accuracy |
| `results/evaluation_summary.csv` | Overall accuracy and correct counts for each model and split |
| `results/per_class_accuracy.csv` | Per-class accuracy and correct counts for each model and split |
| `results/training_loss.csv` | Mean training loss for each of 20 epochs per model |

The checkpoints contain model weights and biases. Optimizer state and training seeds are not stored.
