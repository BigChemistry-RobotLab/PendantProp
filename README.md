# Pendantprop
### An autonomous platform for efficient surfactant characterization
The corresponding github repository for the work *An Autonomous Robotic Module for Efficient Surface Tension Measurements of Formulations* (doi: https://doi.org/10.1038/s41524-025-01842-9). Currently, this repository is still under development and it is recommend to go to the branch v1 for a stable version (see below).

## Conda setup
Use the following commands to clone the repository and set up the required conda environment: 
```
git clone https://github.com/BigChemistry-RobotLab/PendantProp
cd PendantProp
conda create -n pendantprop python=3.13.5
conda activate pendantprop
pip install -r requirements.txt
```
To go to the stable version, which was used to create the results and figures in the corresponding work, use:
```
git checkout v1
```

## Reproducing figures and tables of the corresponding work 
To reproduce the figures and tables from the paper, switch to the branch v1 . Then, run the Jupyter notebook [`figure_notebook.ipynb`](figure_notebook.ipynb), using the pendantprop Conda environment as the selected kernel.
