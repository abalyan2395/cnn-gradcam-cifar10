# References and attribution

## PyTorch tutorial

The model definitions, training loop, and portions of the script comments build on the PyTorch [Training a Classifier tutorial](https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html).

The [PyTorch tutorials license](https://github.com/pytorch/tutorials/blob/main/LICENSE) is BSD 3-Clause. Its notice is included in `licenses/pytorch-tutorials-BSD-3-Clause.txt`. The exact historical tutorial revision used by the starter code is unknown. No project-wide license has been assigned to the author's additions.

## CIFAR-10

CIFAR-10 was created by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton. Dataset information and citation guidance are available at the [CIFAR dataset page](https://www.cs.toronto.edu/~kriz/cifar.html).

The scripts retrieve the dataset through torchvision. The dataset is not included in the repository. The Grad-CAM figure contains one CIFAR-10 example image.

## Grad-CAM

Ramprasaath R. Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh, and Dhruv Batra. *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*.

[Paper](https://arxiv.org/abs/1610.02391)

The paper's doctor/nurse dataset-bias experiment is separate from this project's CIFAR-10 implementation.
