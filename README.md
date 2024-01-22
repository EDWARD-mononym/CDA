# Introduction
Official implementation of the paper "Grounded in Reality: Generalizable and Continual Domain Adaptation for Machine Fault Diagnosis"

# Data preparation
Create a folder named PU_raw in data_preproc folder, download the [PU dataset](http://groups.uni-paderborn.de/kat/BearingDataCenter/) and save them in data_preproc/PU_raw.
Run Paderborn_preproc.ipny to preprocess the raw data file.

# Training a model
To train a model
 ```bash
 python main.py --dataset [dataset] --algo [algorithm] 
 ```

# Running a sweep
To run a sweep