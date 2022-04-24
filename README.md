# destirec-composite

## Description
This is a repository for my Master Thesis project at the Technical University Munich

## Thesis Title
**Item Combination in destination recommendation systems** 
## Time Frame 
15. November 2021 - April 2022.


## Problem Statement
Recommending not just one destination but a combination of items, including how long to stay in suggested regions, is a computationally hard problem. 

In this thesis, a **clear, scalable and efficient algorithm(s)** and solution is to be developed, and a prototype is to be implemented. Results are to be evaluated from users perspective with pre-defined criteria.

## Solution
Different variations to a non-dominated sorting genetic algorithm (nsga-iii) was implemented.

## Requirements 
Run `pip install -r requirements.txt` to install requirements

## Implementation File Structure
```
├── data
│   ├── config.yml
│   └── Pre_Emo_data.csv
├── logs
│   ├── input*
│   └── results: Different csv files and pickle file of results obtained  
├── src      
│    └── DestiRec.py: Main class for destination recommendation algorithm
│    └── evaluation.py: Class used for evaluating the different variants
│    └── main.ipnyb: Notebook for example run of algorithm using feasibility based initialization
│    └── PreEmo.py: Data extraction, transformation and load routine
│    └── variants_evaluation.ipynb: Notebook used for running evaluations
├── requirements.txt: used to install require modules
```

