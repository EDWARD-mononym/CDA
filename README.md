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
Run Paderborn_preproc.ipny to preprocess the raw data file into suitable datasets.

# Baseline comparison
To evaluate the performance of a model, run
 ```bash
 python main.py --dataset [dataset] --algo [algorithm] 
 ```
 The results will be be saved in a result folder in the same directory as main.py. Dataset can be either `PU_Artificial` or `PU_Real`. Algorithm can be any of the methods listed in `Supported models`.

# Running a sweep
To run a sweep with wandB, run
 ```bash
 python main_sweep.py --dataset [dataset] --algo [algorithm] 
  ```
Dataset can be either `PU_Artificial` or `PU_Real`. Algorithm can be any of the methods listed in `Supported models`

# Supported models
Along with EverAdapt, we've also implemented the following methods as a comparison
* CDAN - Conditional Adversarial Domain Adaptation. [Long, M., Cao, Z., Wang, J., & Jordan, M. I. (2018)](https://proceedings.neurips.cc/paper_files/paper/2018/file/ab88b15733f543179858600245108dd8-Paper.pdf)
* COSDA - CoSDA: Continual Source-Free Domain Adaptation. [Feng, H., Yang, Z., Chen, H., Pang, T., Du, C., Zhu, M., ... & Yan, S. (2023)](https://arxiv.org/abs/2304.06627)
* CUA - Adapting to Continuously Shifting Domains. [Bobu, A., Tzeng, E., Hoffman, J., & Darrell, T. (2018)](https://openreview.net/forum?id=BJsBjPJvf)
* DANN - Domain-Adversarial Training of Neural Networks. [Ganin, Y., Ustinova, E., Ajakan, H., Germain, P., Larochelle, H., Laviolette, F., ... & Lempitsky, V. (2016)](https://www.jmlr.org/papers/volume17/15-239/15-239.pdf)
* DeepCORAL - Deep CORAL: Correlation Alignment for Deep Domain Adaptation. [Sun, B., & Saenko, K. (2016)](https://ieeexplore.ieee.org/document/9085896)
* DSAN - Deep Subdomain Adaptation Network for Image Classification. [Zhu, Y., Zhuang, F., Wang, J., Ke, G., Chen, J., Bian, J., ... & He, Q. (2020)](https://ieeexplore.ieee.org/document/9085896)
* MMDA - On Minimum Discrepancy Estimation for Deep Domain Adaptation. [Rahman, M. M., Fookes, C., Baktashmotlagh, M., & Sridharan, S. (2020)](https://link.springer.com/chapter/10.1007/978-3-030-30671-7_6)


