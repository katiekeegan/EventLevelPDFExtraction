# Contrastive Learning for Parton Distribution Function Inference

This repository contains code for an AI-driven approach for learning parameters from event data in quamtum chromodynamics.

Training takes place in two stages. First, a PointNet-style embedding of simulated data is learned through contrastive learning. The idea here is to learn an embedding which preserves distances in the space of event data. Then, an embedding-to-parameters network is trained.

How to train:

1. Run ```python cl.py```. This will automatically save the PointNet embedding model.
2. Run ```python PDF_learning.py.``` This will automatically save the embedding-to-parameters model.

To plot parameters and their relative errors, run ```python plotting_driver.py```.

This repository should require relatively few dependencies. Up-to-date PyTorch,  Numpy, and Matplotlib should suffice. MCEG simulator usage is limited to those who have permission to access the relevant repository.
