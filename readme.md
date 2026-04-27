# Quantum Variational Recommender Network

A novel quantum machine learning approach for recommendation systems, combining Neural Collaborative Filtering (NCF) with Quantum Variational Circuits (QVC).

## Overview

This repository provides an implementation of a smart shopping Recommendation System. It extends classical standard Neural Collaborative Filtering (NeuMF) by integrating Parameterized Quantum Circuits using **PyTorch** and **PennyLane**, exploring the benefits of quantum representation learning within collaborative filtering architectures.

## Acknowledgements

The baseline **Neural Collaborative Filtering (NCF)** framework is cited, derived, and adapted from the following open-source implementation:
* **Source:** [yihong-chen/neural-collaborative-filtering](https://github.com/yihong-chen/neural-collaborative-filtering/tree/master

## Project Structure

The codebase is divided into two primary sub-modules representing the core architectures: `NCF` (classical baseline) and `QVC` (quantum-enhanced).

### 1. Classical Baseline: `NCF/`
Contains the standard implementation of Neural Collaborative Filtering algorithms.
* `GMF.py`: Generalized Matrix Factorization model implementation.
* `MLP.py`: Multi-Layer Perceptron recommendation model.
* `NeuMF.py`: Neural Matrix Factorization, combination of GMF and MLP modules.
* `Engine.py`: PyTorch-based training, validation, and evaluation engine routines.
* `data.py`: Dataset definition and data loaders.
* `metrics.py`: Evaluation metrics including Hit Ratio (HR) and Normalized Discounted Cumulative Gain (NDCG).
* `train.py`: The entry-point script for training and evaluating standard NCF models.
* `utils.py`: Various utility functions and helpers.

### 2. Quantum-Enhanced Model: `QVC/`
Contains the Quantum-Enhanced NeuMF models, incorporating variational quantum circuits.
* `NeuMF_QVC.py`: The core quantum-enhanced recommendation model (`QVCNeuMF`). A Quantum Variational Circuit is placed after the final MLP layer and before the concatenation of the two module outputs.
* `Dressed_Quantum_Net.py`: Implementation of the variational quantum circuit using PennyLane (`DressedQuantumNetwork`). Includes classical pre- and post-processing networks to act as a bridge between quantum states and classical PyTorch tensors.
* `Gaussian_Dressed_Quantum_Net.py`: An alternative formulation of the dressed quantum circuit utilizing Gaussian embeddings.
* `Engine.py`: Specialized training and evaluation engine adapted for quantum layers, handling specific nuances like gradient tracking for hybrid quantum-classical networks.
* `train.py`: Standard entry script to train and evaluate the Quantum-Enhanced NeuMF variants.
* `visualize_quantum_circuit.ipynb`: A Jupyter Notebook dedicated to visualizing the structure, topology, and parameters of the synthesized quantum circuits.
* `data.py`, `metrics.py`, `utils.py`: Specialized dataset loaders, metrics, and helper functions specific to the QVC pipeline.

### 3. Data Preprocessing : `data_preprocessing/`
Contains the code to create the 1000-user and 500-user subsets.

### Artifacts and Output Directories
* `data/` (excluded): Intended directory for datasets and generated subsets (e.g., `test.csv`, `test_subset.csv`).
* `checkpoints/` (excluded) & `worthwhile_checkpoints/`: Directories storing `.model` weight binaries saved across training epochs for both classical NeuMF and QVC models. 
* `runs/`: TensorBoard event logs storing training progression metrics.
* `visualizations/`: Assets and plots for monitoring gradient descent and embeddings.

## Requirements

Requirements are listed in the `requirements.txt` file in project root.
* PyTorch might need to be commented out and installed separately.


This experiment was conducted on a Windows 11 machine with an Nvidia GPU.
