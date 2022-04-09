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
    #filename = f'logs/results/evaluation.csv'
    #with open(filename, 'w', newline='') as convert_file:
    #        csv = writer(convert_file)
    #        csv.writerow([h for h in header])
    for num,testConfig in  tests.items():
        if num=='test1' or num=='test2' or num=='test6' or num=='test4' or num=='test3':
            continue
        parameters = testConfig['parameters']
        nruns = testConfig['nruns']
        for inputnum, input in inputs.items():
           
            considered, othercategories, needed_months, months_weeks_dict, start_df  = get_dataset(input, rank=parameters['prerank'])
            user_input = input
            region_groups = get_region_groups(start_df)
            region_index_info = get_region_index_info(start_df)
            emo = DestiRec(user_input, considered, othercategories, needed_months, months_weeks_dict, start_df, region_groups, region_index_info)
            toolbox = emo.prepare_toolbox('', NOBJ=parameters['objs'], 
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
                population, stats, result = emo.main(toolbox, run=run+3, input=inputnum[-1], variant=num[-1])
                if bestRun != None and result["Best_Score"] > bestRun["Best_Score"]:
                    bestRun = result
                elif bestRun == None:
                    bestRun = result
                
            del toolbox
            del emo
            os.system('cls||clear')
            with open(f'logs/results/Test_{experiment_name}.pickle', 'wb') as f:
                pickle.dump(bestRun, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            draw_statistics(parameters['objs'], parameters['max'], experiment_name, bestRun)
            draw_feasibility_stats(parameters['objs'], experiment_name, bestRun)
           


def draw_statistics(obj, ngen, experiment_name, result):
    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111)
    mins = []
    maxes = []
    means = []
    std = []
    for i, gen in enumerate(result['Process']):    
        df = pd.DataFrame(gen['fitnesses'])
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df = df.dropna(subset=[x for x in range(obj)], how="all")
        sums = df.sum(axis=1)
        means.append(sums.mean())
        mins.append(sums.min())
        maxes.append(sums.max())
        std.append(sums.std())

    mins = np.array(mins)
    maxes = np.array(maxes)
    means = np.array(means)
    std = np.array(std)
    ax.errorbar(np.arange(ngen-1), means, std, fmt='ok', lw=3, ecolor='black', color='red')
    ax.errorbar(np.arange(ngen-1), means, [means - mins, maxes- means],
                fmt='.k', ecolor='gray', lw=1, color='red')
    plt.xlim(0, len(result['Process']))
    lim = -40 if mins.min() < -40 else mins.min()
    plt.yticks(np.arange(lim, maxes.max(), step=5))
    ax.set_xlabel('Iteration', fontsize=15)
    ax.set_ylabel('Item Combination Score', fontsize=15)
    #plt.autoscale(tight=True) 
    ax.set_title('Statistics on Generated Item Combination Per Iteration')   
    plt.savefig(f'logs/results/statistics {experiment_name}.png')

def draw_feasibility_stats(obj, experiment_name, result):
    fig = plt.figure(figsize=(10,10))
    spec = fig.add_gridspec(2, 2, hspace=0.3)
    ax = fig.add_subplot(spec[0, :])
    ax2 = fig.add_subplot(spec[1, :], xticklabels=[], yticklabels=[])
    main_df = pd.DataFrame()
    for i, gen in enumerate(result['Process']):    
        df = pd.DataFrame(gen['feasibility'])
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df = df.dropna(subset=[0, 1], how="all")
        df['Iteration'] = i
        if(main_df.empty):
            main_df = df
        else:
            main_df = pd.concat([main_df, df], ignore_index=True)
    main_df = main_df.rename(columns={1:'Combination'})
    palette = {
    'Feasible': 'tab:green',
    'Infeasible': 'tab:red',
    }
    sns.lineplot(data = main_df, x = 'Iteration', y = 0, hue='Combination', palette=palette,
                ax=ax)
    sns.histplot(main_df, x='Iteration', hue='Combination', multiple='stack', ax=ax2, legend=False, palette=palette)  
    ax.set(xlabel='', ylabel='Item Combination Score', title='Generated Combination Feasibility')
    ax.title.set_fontsize(15) 
    for item in ([ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(14)
    xlabels = ax.get_xticks().tolist()
    ax.set_xticklabels([int(x) for x in xlabels])
    ylabels = ax.get_yticks().tolist()
    min_y = np.min(ylabels) if np.min(ylabels) > -40 else -40
    ax.set_ylim(bottom=min_y)
    plt.autoscale(tight=True) 
    plt.savefig(f'logs/results/feasibility {experiment_name}.png')

if __name__ == "__main__":
    init_logging('logs/evaluation.log', append=True)
    run()

            