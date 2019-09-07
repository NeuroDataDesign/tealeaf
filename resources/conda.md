# Setting up a reproducible computing environment

Vivek Gopalakrishnan | September 6, 2019

> If 2+2=4 on my computer, it better equal 4 on your computer too.

Being able to reproduce results is an essential part of data science.
An easy first step is to make sure we all use the same computing environment!

We do this using Conda, an open-source package management system and environment 
management system that works across all operating systems. More information is
available [here](https://conda.io/en/latest/).

## Procedure

1. Install miniconda for Python 3 by download and running [this script](https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh))
2. Run the following script in terminal to setup your environment
```
# Specify environment details
PROJECT_NAME=tealeaf
PYTHON_VERSION=3.7

# Create environment
### nb_conda_kernals allows all conda environments to be activated from within Jupyter
conda create --name $PROJECT_NAME python=$PYTHON_VERSION \
  jupyter nb_conda_kernels \
  networkx numpy pandas scikit-learn scipy seaborn h5py \
  black flake8 \
  tqdm
 
# Activate environment and pip install remaining packages
conda activate $PROJECT_NAME

conda install -c conda-forge jupyter_contrib_nbextensions \
                             jupyter_nbextensions_configurator

pip install graspy mgcpy lolP
```

## Usage

When you want use the `tealeaf` environment, start by running `conda activate tealeaf` in terminal.

If you want to use the `tealeaf` environment in Jupyter, startup a normal Jupyter session
(with `jupyter notebook`) and choose `Python [conda env:tealeaf]`. 