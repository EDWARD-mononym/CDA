# Introduction
Official implementation of the paper "Grounded in Reality: Generalizable and Continual Domain Adaptation for Machine Fault Diagnosis"

# Data preparation
Download the [CWRU dataset](https://engineering.case.edu/bearingdatacenter/download-data-file) and [PU dataset](http://groups.uni-paderborn.de/kat/BearingDataCenter/) and save them in data_preproc/CWRU_raw and data_preproc/PU_raw respectively.
Run CWRU_preproc.ipny and Paderborn_preproc.ipny to preprocess the raw data file.

# Training a model
To train a model
 ```bash
 python main.py --dataset
 ```

# Running a sweep
To run a sweep