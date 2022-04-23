from json import tool
from PreEmo import *
from DestiRec import DestiRec
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid')
import pickle
import warnings
import os
warnings.filterwarnings("ignore", 'This pattern has match groups')
warnings.filterwarnings("ignore", 'invalid value encountered in true_divide')
warnings.filterwarnings("ignore", 'A class named ')
from csv import writer

def run():
    config = get_config("config.yml")
    inputs = config['input']
    tests = config['tests']
    header = ['Nrun','Variant','Input','Score','Nregions','EvaluatedRegions','EvaluatedCombinations','Totaltime','SuggestedRegions','SuggestedDuration','SuggestedBudget']
    filename = f'logs/results/evaluation.csv'
    with open(filename, 'w', newline='') as convert_file:
        csv = writer(convert_file)
        csv.writerow([h for h in header])
    for num,testConfig in  tests.items():
        parameters = testConfig['parameters']
        nruns = testConfig['nruns']
        for inputnum, input in inputs.items():
            considered, othercategories, needed_months, months_weeks_dict, start_df  = get_dataset(input, rank=parameters['prerank'])
            user_input = input
            region_groups = get_region_groups(start_df)
            region_index_info = get_region_index_info(start_df)
            emo = DestiRec(user_input, considered, othercategories, needed_months, months_weeks_dict, start_df, region_groups, region_index_info)
            toolbox = emo.prepare_toolbox( NOBJ=parameters['objs'], 
                                            POP_SIZE=parameters['pop'], 
                                            GEN=parameters['max'], 
                                            P=parameters['refs'], 
                                            prerank=parameters['prerank'], 
                                            feasibleOnly=parameters['FeasibleOnly'], 
                                            penalizeInObj=parameters['PenaltyBased'])
            bestRun = None
            bestRun2 = None
            for run in range(nruns):
                experiment_name = f'{num}-{inputnum}'
                toolbox.experiment_name = experiment_name
                population, stats, result, _ = emo.main(toolbox, run=run+3, input=inputnum[-1], variant=num[-1])
                if bestRun != None and result["Best_Score"] > bestRun["Best_Score"]:
                    bestRun = result
                elif bestRun == None:
                    bestRun = result
                
            del toolbox
            del emo
            with open(f'logs/results/Test_{experiment_name}.pickle', 'wb') as f:
                pickle.dump(bestRun, f, protocol=pickle.HIGHEST_PROTOCOL)
            


if __name__ == "__main__":
    init_logging('logs/evaluation.log', append=True)
    run()

            