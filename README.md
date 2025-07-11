This is the official implementation of 

# Embedded Inter-Subject Variability in Adversarial Learning for Inertial Sensor-Based Human Activity Recognition

published in IEEE International Workshop on Machine Learning for Signal Processing (MLSP) 2025.

Citation is coming...

Here you have some instructions in case you want to run the code in your local machine. First of all, you need to install the requirements. I have tried this code with Python 3.10.8 and 3.10.12. Both of them works.

## Datasets
The datasets used in this work are publicly available and can be easily downloaded from the following links:

- **PAMAP2:** [https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring](https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring)
- **MHEALTH:** [https://archive.ics.uci.edu/dataset/319/mhealth+dataset](https://archive.ics.uci.edu/dataset/319/mhealth+dataset)
- **REALDISP:** [https://archive.ics.uci.edu/dataset/305/realdisp+activity+recognition+dataset](https://archive.ics.uci.edu/dataset/305/realdisp+activity+recognition+dataset)

Once you have downloaded the datasets, place them into the `./datasets` directory. Decompress the zip files and move every file corresponding to each particiapnt into the `normal` subdirectory as follows:

```bash
datasets
├── MHEALTH
│   └── normal
│        └── File 1
│        └── File 2...
├── PAMAP2
│   └── normal
│        └── File 1
│        └── File 2...
└── REALDISP
    └── normal
        └── File 1
        └── File 2...
```
## Setting Up Environment Variables

Before running the project, you need to define the required environment paths. This is done using the `SET_ENV_VARIABLES.sh` script included in the repository.

This script sets the root path of the project and the path to the datasets, based on your system's hostname.

### Script Structure

```bash
if [[ $(hostname) == "YOURHOSTNAME" ]]; then
    export MLSP_ROOT="/path/to/your/project/folder/MLSP2025/"
    export MLSP_DATA_ROOT="/path/to/your/project/folder/MLSP2025/"
fi
```
If you are on a Linux-based system, simply run the following command in your terminal to obtain your hostname:
```bash
hostname
To get the absolute paths for your project or dataset directories, run:
```bash
pwd
```
The pwd command outputs the current working directory, which you can use to set the paths.


## Setting up Virtual Environment

Now that we have the datasets and environment variables set up, it's time to create the Python virtual environment.

Create the virtual environment using:

```bash
python3 -m venv .MLSP
source .MLSP/bin/activate
```


Once the virtual environment is created, install the required packages by running:

`pip install -r requirements.txt`

Now you can run a script for initializing everything at one:
```bash
source MODULES.sh
```

## Preparing Classification and Discrimination Datasets

Finally, to prepare all the datasets (classification and discrimination tasks), execute the following script:

```bash
source PrepareDatasetFiles.sh
```

## Run the Code for the experiments

The File for running the experiment is `pipeline.py`. This file has the following input arguments

```python
    "dataset": ['PAMAP2', 'REALDISP', 'MHEALTH'], 
    "experiment": [1,2,3,4,5,6...], 
    "seed": [0,1,2,3,....],
    "device": ['cuda:0', 'cuda:1', ...],
    "step_1_lr": [0.1,0.001,0.0001,...], 
    'step_1_bs': [16,32,64,...],
    'step_2_decoder_lr': [0.1,0.001,0.0001,...],
    'step_2_decoder_bs':[16,32,64,...],
    'step_2_classifier_lr': [0.1,0.001,0.0001,...],
    'step_2_classifier_bs': [16,32,64,...],
    'step_2_discriminator_lr': [16,32,64,...],
    'step_3_discriminator_weight': 0.1,
    'step_3_reconstruction_weight': 0.2,
    'step_3_classification_weight': 0.7,
    'id': [0,1,2,3,4,...],
    'trainsize': [0.1,0.2,0.3...1.0]
```
"dataset": This defines the dataset in which you want to test the approach 
"experiment": This is the id of the participant that you will leave out in the Leave One Person Out Cross-Validation 
"seed": The seed you want to initialize to.
"device": The device in which you would like to run things.
"step_1_lr": Learning Rate
'step_1_bs': Batch Size
'step_2_decoder_lr': Learning Rate
'step_2_decoder_bs':Batch Size
'step_2_classifier_lr': Learning Rate
'step_2_classifier_bs': Batch Size
'step_2_discriminator_lr': Learning Rate
'step_3_discriminator_weight': The weight for the adversarial loss function (LA on the paper)
'step_3_reconstruction_weight': The weight for the reconstruction loss
'step_3_classification_weight': The weight for the classification loss
'id': This is for saving different versions of the experiments
'trainsize': This is for reducing the quantity of train dataset.


## Run the Code for the experiments

Before running the coder, you should run:


```bash
source MODULES.sh
```

This is an initialization of the virtual environment and the variables. After this, you can use the files:

```bash
source A_RunScriptREALDISP.sh
source B_RunScriptPAMAP2.sh
source C_RunScriptMHEALTH.sh
```

To run the 1 iteration of the LOSO Cross validation. This might take some hours.
