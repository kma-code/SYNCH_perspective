# Odor space analysis, synthetic ORN dataset, and odor classification

This repository contains scripts and data to reproduce Figures 4,5,6 in _Synthetic Biology Meets Neuromorphic Computing: Towards a Bio-Inspired Olfactory Perception System_ by Max, Sames, Ye, Steinkühler, and Corradi (2025).

Electrophysiological recordings in `20170717_cell4.csv` from [Cryo-EM structure of the insect olfactory receptor Orco](https://www.nature.com/articles/s41586-018-0420-8), Butterwick et al. 2018, provided by Josefina del Mármol.

## Installation

Clone all contents of this folder, cd into it, make new python environment and activate:
```
git clone https://github.com/kma-code/SYNCH_perspective.git
cd SYNCH_perspective
python3 -m venv SYNCHenv
source SYNCHenv/bin/activate
pip3 install -r requirements.txt
python -m ipykernel install --user --name=SYNCHenv 
```

## What this repository does

This repo generates a synthetic (fake!) dataset of voltage traces using data extracted from [Cryo-EM structure of the insect olfactory receptor Orco](https://www.nature.com/articles/s41586-018-0420-8/figures/7) by Butterwick et al., 2018.

To understand the steps of generating the dataset, look at the [jupyter notebook](https://github.com/kma-code/SYNCH_perspective/blob/main/dataset_explanation.ipynb).

The actual dataloaders are generated using the following scripts:

### Defining a parameter file

You need to define a parameter file containing the following items.
A template can be found in [saved_datasets/dataset_template.json](https://github.com/kma-code/SYNCH_perspective/blob/main/saved_datasets/dataset_template.json) (make sure that you activate SYNCHenv).

- `odorant_names`: list of odorants. To see all available odors, run `python odor_space_analysis.py --list`. Include the `none` odor for better results, see paper.
- `N_OR`: number of receptors to be reconstituted
- `N_ORCOs`: how many Orcos to simulate per ORN
- `output`: `voltage` or `mean`, i.e. whether a trace should be generated or only the mean voltage
- `output_dt`: e.g. `0.005`, how big one time step (in seconds) in the generated traces should be
- `total_steps`: how many steps to simulate per trace
- `data_steps`: how many of `total_steps` should be 
- `dataset_size`: how many traces to generate per odorant

### Odor space analysis

To perform the analysis, run `python odor_space_analysis.py --params 'saved_datasets/dataset_template.json'`, using your parameter file.
The selected OR types will be saved back into the parameter file.

To see some examples, see the [jupyter notebook](https://github.com/kma-code/SYNCH_perspective/blob/main/odor_space_analysis.ipynb) (make sure that you activate SYNCHenv).

### Generating a dataset

To generate a dataset, run `python generate_dataset.py --params 'saved_datasets/dataset_template.json'` **after** running `odor_space_analysis.py`.
Simulation parameters will be saved back into the parameter file.

The dataset will be saved in `saved_datasets` as `.pkl` files. To import, use:
```
import dill
import json
# load dataset
with open('saved_datasets/dataset_size1_Nodor3_NOR7_voltage.pkl', 'rb') as f:
    dataset = dill.load(f)
# load corresponding dict
with open('saved_datasets/dataset_template.json') as f:
    dataset_dict = json.load(f)
```

### Running the classifier

To run the classification, check out the [jupyter notebook](https://github.com/kma-code/SYNCH_perspective/blob/main/SNN_classification.ipynb) (make sure that you activate SYNCHenv).
