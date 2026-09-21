## Experimental Setup
This repository contains the code to reproduce the results of the study "Temporal Analysis of the AAVE Governance Token Transfer Network".

### Environment:
Set up a Python `3.11.15` virtual environment.

### Dependencies:
Install the following packages:  
```
numpy~=2.4.4
pandas~=3.0.2
networkx~=3.6.1
matplotlib~=3.10.9
seaborn~=0.13.2
statsmodels~=0.14.6
 ```
Or run the following command in the activated environment

`pip install -r requirements.txt`

## Execution Instruction

To run the experiments, execute the command:

`python main.py`

This script will generate all the figures presented in the paper, and save them in the `output` folder.