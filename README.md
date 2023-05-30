

## DeepFish (A Realistic Fish-Habitat Dataset to Evaluate Algorithms for Underwater Visual Analysis) 

### Accepted at Nature Scientific Reports [[Paper]](https://www.nature.com/articles/s41598-020-71639-x)   

![CNN](docs/Figure_4.png)
![counting](docs/count.gif) 
![Segmentation](docs/seg.gif) 

## Prerequisites: 

**Install tkn** from [here](https://docs.openshift.com/container-platform/4.5/cli_reference/tkn_cli/installing-tkn.html)
To add directory to the path add this line to the ~/.bashrc / ~/.zshrc ... terminal manager:
export PATH=$PATH:/path/to/directory


## Install requirements

Freezing has been done with this command because done from a conda env to avoid format issues: 

`pip list --format=freeze > requirements.txt `

Then installation itself:

`pip install -r requirements.txt` 

`pip install git+https://github.com/ElementAI/LCFCN`

Note that the pytorch installation is q bit different to get the pytorch version compatible with Mac M1 chip as suggested [there](https://stackoverflow.com/questions/68820453/how-to-run-pytorch-on-macbook-pro-m1-gpu)

## Download

*  Download the DeepFish dataset from [here](https://cloudstor.aarnet.edu.au/plus/s/NfjObIhtUYO6332)

## 1. Train and test on single image

### Localization
```
python scripts/train_single_image.py -e loc -d ${PATH_TO_DATASET}
```

This outputs the following image 

![CNN](docs/single_image_loc.png)

### Segmentation

```
python scripts/train_single_image.py -e seg -d ${PATH_TO_DATASET}
```

This outputs the following image 

![CNN](docs/single_image_seg.png)

## 2. Train and test on the dataset

Run the following command to reproduce the experiments in the paper:

`python trainval.py -e ${TASK} -sb ${SAVEDIR_BASE} -d ${DATADIR} -r 1`

The variables (`${...}`) can be substituted with the following values:

* `TASK` : loc, seg, clf, reg
* `SAVEDIR_BASE`: Absolute path to where results will be saved
* `DATADIR`: Absolute path containing the downloaded datasets

Experiment hyperparameters are defined in `exp_configs.py`

## Citations

If you use the DeepFish dataset in your work, please cite it as:

```
@article{saleh2020realistic,
    title={A Realistic Fish-Habitat Dataset to Evaluate Algorithms for Underwater Visual Analysis},
    author={Alzayat Saleh and Issam H. Laradji and Dmitry A. Konovalov and Michael Bradley and David Vazquez and Marcus Sheaves},
    year={2020},
    eprint={2008.12603},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
