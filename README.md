# byop_2y-qkformer-sssa_snnvit-
# Spiking Vision Transformer Routes for Event-Based Vision
This project contains multiple architectural implementations of a Spiking Neural Network–based Vision Transformer (SNN-ViT) evaluated on the CIFAR10-DVS event-based vision dataset.
I have implemented architectures from 2 different research papers and combining them to create 4 distinct routes and compared them
# Environment Setup
## Requirements

* Python ≥ 3.8

* PyTorch

* snntorch
* aedat
* Ipython

* sklearn

* tonic

* timm

* numpy

* matplotlib

# Running the Code
Each architectural route is implemented as a separate Colab notebook.

You can run each cell by cell in Google colab ,the requirements will be downloaded along the way.

If you are running this in jupyter or other notebook make sure to download all required libraries.
# Implemented Routes

| Route   | Attention in Stage 1 | Attention in stage 2 | Test Accuracy |
|---------|----------------------|----------|---------------|
|  1 | QKTA     | SSA      | 56.95%        |
|  2 | QKCA      | SSA      | **63.45%**    |
|  3 | QKTA      | SSSA     | 54.35%        |
|  4 | QKCA     | SSSA     | 51.80%        |

# Limitations
* Limited number of epochs(10)
* Fixed learning rate in optimizer, no use of schedulers
* Didn't implement SSSA_v2

# Observations
* QKCA + SSA achieved the highest test accuracy (63.45%)
* SSSA reduced overall performance
* Combining QKFormer and SSSA might need more optimization
* Works better with higher optimizer learning rates(1e-3 performs better than 5e-3)

# References
* QKFormer: Hierarchical Spiking Transformer using Q-K Attention(https://arxiv.org/abs/2403.16552)
* Spiking Vision Transformer with Saccadic Attention(https://arxiv.org/abs/2502.12677)
* Spikformer: When Spiking Neural Network Meets Transformer(https://arxiv.org/abs/2209.15425)
