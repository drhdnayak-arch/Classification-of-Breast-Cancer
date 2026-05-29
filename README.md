# Classification of Breast Cancer using Voting Ensemble Learning with GT-PCA based Feature Selection via Simulated Annealing SEM tuning

# Overview

This project presents an intelligent breast cancer classification framework that integrates:
* General Transform-Invariant Principal Component Analysis (GT-PCA)
* Soft Voting Ensemble Learning
* Simulated Annealing Stochastic Expectation Maximization (SASEM)

The proposed framework is designed to improve classification accuracy, robustness, convergence stability, and computational efficiency for breast cancer diagnosis using the Wisconsin Diagnostic Breast Cancer (WDBC) dataset.

The framework combines dimensionality reduction, ensemble learning, and stochastic hyperparameter optimization to achieve reliable benign and malignant tumor classification.

# Proposed Architecture
The proposed framework consists of the following stages:
1.	Data Collection
2.	Data Preprocessing
3.	GT-PCA Feature Extraction
4.	Soft Voting Ensemble Classification
5.	SASEM Hyperparameter Optimization
6.	Performance Evaluation

The ensemble classifier combines:

* K-Nearest Neighbors (KNN)
* Logistic Regression (LR)
* Gradient Boosting (GB)
* Random Forest (RF)
* Feedforward Neural Network (FFNN)
  
using soft voting strategy for final prediction.

# Dataset
Dataset Used:

  Wisconsin Diagnostic Breast Cancer (WDBC) Dataset

Source:

  UCI Machine Learning Repository

Dataset Characteristics:
* Total samples: 569
*	Features: 30 numerical attributes
*	Classes:
      *	Malignant
      *	Benign
    
Important features include:
* Radius
*	Texture
*	Perimeter
*	Area
*	Smoothness
*	Compactness
*	Concavity
*	Symmetry
*	Fractal Dimension

# Technologies and Libraries
The implementation uses Python and the following libraries:
*	NumPy
*	Pandas
*	Scikit-learn
*	Matplotlib
*	Seaborn
*	Optuna
*	Psutil

# Installation
Clone the repository:

git clone https://github.com/your-repository-name.git
cd repository-name

Install required dependencies:

  pip install -r requirements.txt
  
Or manually install:

  pip install numpy pandas scikit-learn matplotlib seaborn optuna psutil

# Running the Project
Run the Python script:

python main.py

Or execute the notebook in Google Colab/Jupyter Notebook.

# Workflow Description
# Step 1: Preprocessing
The dataset is normalized using StandardScaler to improve training stability and convergence.
# Step 2: GT-PCA Feature Extraction
GT-PCA performs transformation-invariant dimensionality reduction by projecting high-dimensional features into compact principal components while preserving discriminative information.
# Step 3: Ensemble Classification
The reduced features are passed into multiple classifiers:
*	KNN
*	LR
*	GB
*	RF
*	FFNN

The outputs are combined using soft voting.

# Step 4: SASEM Optimization
SASEM combines:
*	Simulated Annealing (SA)
*	Expectation Maximization (EM)
to optimize classifier hyperparameters through stochastic exploration and adaptive convergence.

# Performance Metrics
The framework evaluates the following metrics:
*	Accuracy
*	Precision
*	Recall
*	F1-Score
*	ROC-AUC

# Experimental Results
The proposed framework achieved:

| Metric    | Value   |
|-----------|---------|
| Accuracy  | 99.56%  |
| Precision | 99.48%  |
| Recall    | 99.75%  |
| F1-Score  | 99.63%  |
| AUC-ROC   | 0.99    |

# Visualization Outputs
The implementation generates:
*	Confusion Matrix
*	GT-PCA Component Importance
*	Runtime Analysis
*	Memory Usage Analysis

# Advantages of Proposed Framework
*	Improved classification accuracy
*	Reduced dimensionality
*	Better convergence stability
*	Avoidance of local optima
*	Reduced computational complexity
*	Enhanced robustness and generalization

# Project Structure

```text
├── main.py
├── README.md
├── requirements.txt
├── dataset/
├── outputs/
└── figures/
