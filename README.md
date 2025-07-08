# Intransigent teachers guide better test-time adaptation students
This code is based on an open source online test-time adaptation repository [(link)](https://github.com/mariodoebler/test-time-adaptation). 

## Prerequisites
To use this code, conda environment is provided:
```bash
conda update conda
conda env create -f environment.yml
conda activate tta 
```

### Get Started
To run one of the following benchmarks, the corresponding datasets need to be downloaded.
- *CIFAR10-to-CIFAR10-C*: the data is automatically downloaded.
- *ImageNet-to-ImageNet-C*: for non source-free methods, download [ImageNet](https://www.image-net.org/download.php) and [ImageNet-C](https://zenodo.org/record/2235448#.Yj2RO_co_mF).
- *ImageNet-to-ImageNet-R*: for non source-free methods, download [ImageNet](https://www.image-net.org/download.php) and [ImageNet-R](https://github.com/hendrycks/imagenet-r).
- *DomainNet-126*: download the 6 splits of the [cleaned version](http://ai.bu.edu/M3SDA/). Following [MME](https://arxiv.org/abs/1904.06487), DomainNet-126 only uses a subset that contains 126 classes from 4 domains.
- *ImageNet-to-CCC*: for non source-free methods, download [ImageNet](https://www.image-net.org/download.php). CCC is integrated as a webdataset and does not need to be downloaded! Please note that it cannot be combined with settings such as correlated.

If you are going to use the `test_time.py` script directly to run experiments, specify the root folder for all datasets `_C.DATA_DIR = "./data"` in the file `conf.py`. For the individual datasets, the directory names are specified in `conf.py` as a dictionary (see function `complete_data_dir_path`). In case your directory names deviate from the ones specified in the mapping dictionary, you can simply modify them.

You can also utilize `run*.py` scripts which allow to run multiple experiments at once. You can adjust the experiment parameters in the scripts directly in the /*TO MODIFY*/ section. The root folder for all datasets can be specified using `DATADIR` variable.

### PETAL Pretraining
PETAL method requires additional pretraining on source data. Run `petal_train_swag.py` script for further training on source domain training data, with optional `DATA_DIR` param: 
```bash
python petal_train_swag.py --cfg cfgs/[dataset_name]/petal.yaml DATA_DIR [dataset_root_folder] 
```
This will create the files `[model_name]_cov.pt` and `[model_name]_swa.pt` inside the directory `ckpt/petal/[dataset_name]/`. 

### Run Experiments

Config files for all experiments and methods are provided. To conduct a single experiment, simply run the following Python file with the corresponding config file.
```bash
python test_time.py --cfg cfgs/[ccc/cifar10_c/imagenet_c/imagenet_others/domainnet126]/[source/tent/eata/rdumb/sar/cotta/rotta/adacontrast/petal/memo/norm_test/losstest].yaml
```

To run AdaContrast, CoTTA, RoTTA or PETAL experiments with an `intransigent teacher` set the `M_TEACHER.FROZ` argument to True:
```bash
python test_time.py --cfg cfgs/[ccc/cifar10_c/imagenet_c/imagenet_others/domainnet126]/[cotta/rotta/adacontrast/petal].yaml M_TEACHER.FROZ True
```

For imagenet_others, the argument CORRUPTION.DATASET has to be passed:
```bash
python test_time.py --cfg cfgs/imagenet_others/[source/tent/eata/rdumb/sar/cotta/rotta/adacontrast/petal/memo/norm_test/losstest].yaml CORRUPTION.DATASET imagenet_r
```

E.g., to run SAR for the ImageNet-to-ImageNet-R benchmark, run the following command.
```bash
python test_time.py --cfg cfgs/imagenet_others/sar.yaml CORRUPTION.DATASET imagenet_r
```

Alternatively, you can run multiple experiment by modifying and running `run.py`script.

To run the different continual DomainNet-126 sequences, you have to pass the `MODEL.CKPT_PATH` argument. When not specifying a `CKPT_PATH`, the sequence using the *real* domain as the source domain will be used.
The checkpoints are provided by [AdaContrast](https://github.com/DianCh/AdaContrast) and can be downloaded [here](https://drive.google.com/drive/folders/1OOSzrl6kzxIlEhNAK168dPXJcHwJ1A2X). Structurally, it is best to download them into the directory `./ckpt/domainnet126`.

### Reproduce results from Table 3
To reproduce results from Table 3, run `run_tab3_bs10.py` script for BS=10 or `run_tab3_bs64.py` script for BS=64.
