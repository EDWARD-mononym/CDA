# Introduction
Implementation of our work EverAdapt: Continuous Adaptation for Machine Fault Diagnosis Under Dynamic Environments.
To ensure fair comparison, we use the same feature extractor and classifier as shown in architecture folder.

# Data preparation
Run Paderborn_preproc.ipny and UO_preproc.ipny to preprocess the raw data file into suitable datasets.

# Baseline comparison
To evaluate the performance of a model, run
 ```bash
 python main.py --dataset [dataset] --algo [algorithm] 
 ```
 The results will be be saved in a result folder in the same directory as main.py. Dataset can be either `PU_Artificial`, `PU_Real` or `UO`. Algorithm can be any of the methods listed in `Supported models`.

# Ablation, Replay & Stability study
To replicate our experiments, run
 ```bash
 python experiments.py --experiment [experiment]
 ```
 The results will be be saved in a result folder in the same directory as main.py. Available experiments can be either of `Ablation`, `Replay` or `Stability`

# Running a sweep
To run a sweep with wandB, run
 ```bash
 python main_sweep.py --dataset [dataset] --algo [algorithm] 
  ```
Dataset can be either `PU_Artificial`, `PU_Real` or `UO`. Algorithm can be any of the methods listed in `Supported models`


