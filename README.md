# Pendantprop
### An autonomous platform for efficient surfactant characterization
The corresponding github repository for the work *An Autonomous Robotic Module for Efficient Surface Tension Measurements of Formulations* (currently under review).

## Conda setup
Use the following commands to clone the repository and set up the required conda environment: 
```
git clone https://github.com/BigChemistry-RobotLab/PendantProp
cd PendantProp
conda create -n pendantprop python=3.13.5
conda activate pendantprop
pip install -r requirements.txt
```

## Reproducing results of the paper 
To reproduce the figures and tables from the paper, run the Jupyter notebook [`figure_notebook.ipynb`](figure_notebook.ipynb), using the pendantprop Conda environment as the selected kernel.