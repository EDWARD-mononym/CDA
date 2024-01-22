# Introduction
Implementation of our work EverAdapt: Continuous Adaptation for Machine Fault Diagnosis Under Dynamic Environments.
To ensure fair comparison, we use the same feature extractor and classifier as shown in architecture folder.

# Data preparation


Create a folder named PU_raw in the data_preproc folder, download the [PU dataset](http://groups.uni-paderborn.de/kat/BearingDataCenter/) files, extract and save them in data_preproc/PU_raw.
Run Paderborn_preproc.ipny to preprocess the raw data file.

```code block
├── dataset
│ ├── DomainNet
│ │ └── splits
│ ├── clipart
│ ├── infograph
│ ├── painting
│ ├── quickdraw
│ ├── real
│ └── sketch
└── Office31
├── image_list
├── amazon
├── dslr
└── webcam
```

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

