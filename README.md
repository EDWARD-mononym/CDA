# Introduction
Implementation of our work EverAdapt: Continuous Adaptation for Machine Fault Diagnosis Under Dynamic Environments.
To ensure fair comparison, we use the same feature extractor and classifier as shown in architecture folder.

# Data preparation
Create a folder named `PU_raw` in the `data_preproc` folder, download the [PU dataset](http://groups.uni-paderborn.de/kat/BearingDataCenter/) files, extract and save them in data_preproc/PU_raw.
```code block
├── data_preproc
│ ├── PU_raw
│ ├── K001
│ ├── K002
│ ├── K003
│ └── ...
│
└── Paderborn_preproc.ipynb
```
Run Paderborn_preproc.ipny to preprocess the raw data file.

# Baseline comparison
To evaluate the performance of a model, run
 ```bash
 python main.py --dataset [dataset] --algo [algorithm] 
 ```
The results will be be saved in a result folder in the same directory as main.py

# Running a sweep
To run a sweep with wandB, run
 ```bash
 python main_sweep.py --dataset [dataset] --algo [algorithm] 
 ```

 # Adding an algorithm

