from json import tool
from tabnanny import verbose
import time, array, random, copy, math
from collections import Counter
from unittest import result
import numpy as np
import logging
import pandas as pd
from operator import add, sub
import random
import PreEmo
import pickle
from selection import *
from deap import algorithms, base, benchmarks, tools, creator
from collections import Sequence
import json
import os

ref_points = list()
#P=15 recommended
run = 0
def prepare_toolbox(experiment_name, POP_SIZE = 92, GEN=250, NOBJ=3, CROSSPB=0.3, P=4, prerank=True, feasibleOnly=False, penalizeCombo=True):
    global ref_points
    pd.options.mode.chained_assignment = None
    user_input, considered, othercategories, needed_months, months_weeks_dict, start_df = PreEmo.get_dataset(rank=prerank)
    region_groups = PreEmo.get_region_groups(start_df)
    region_index_info = PreEmo.get_region_index_info(start_df)
    categories = [x for x in considered if x not in needed_months]
    NDIM = len(considered)
    MUTPB = 1/NDIM
    NUM_REGIONS = len(region_index_info) 
    BUDGET = user_input['Budget']
    DURATION = user_input['Duration']
    toolbox = base.Toolbox()
    toolbox.df = start_df
    toolbox.region_groups = region_groups
    toolbox.region_indexInfo = region_index_info
    toolbox.othercategories = othercategories
    toolbox.pop_size = POP_SIZE
    toolbox.max_gen = GEN
    toolbox.mut_prob = MUTPB
    toolbox.ref_p = P
    toolbox.cross_prob = CROSSPB
    toolbox.num_obj = NOBJ
    toolbox.penalize_combo = penalizeCombo
    toolbox.experiment_name = experiment_name.format(toolbox.pop_size, toolbox.max_gen, NOBJ, P, str(prerank), str(feasibleOnly), str(penalizeCombo))
    print(toolbox.experiment_name)
    toolbox.user_input = user_input
    ref_points = tools.uniform_reference_points(NOBJ, p=P)
    creator.create("FitnessMoop", base.Fitness, weights=(1.0, )*NOBJ)
    creator.create("Individual", list)
    creator.create("Strategy", list)
    creator.create("BudgetFeasibility", set)
    creator.create("DurationFeasibility", set)
    creator.create("BooleanIndividuals", list, fitness=creator.FitnessMoop)
    creator.create("ViolationDegree", int, typecode='i')
    creator.create("Feasible", int, typecode='i')
    creator.create("DistanceToFeasible", list)
    toolbox.register("mate", tools.cxUniform, indpb=CROSSPB)
    toolbox.register("mutate", tools.mutFlipBit, indpb=MUTPB)
    toolbox.register("select", tools.selNSGA3, ref_points=ref_points)

    def evaluate(individual):
        scores = start_df.loc[(start_df['RId'].isin(individual.strategy)) & (start_df['category'].isin(considered))].groupby('RId')['cat_score'].sum()*individual
        result = scores.values
        if(penalizeCombo):
            result = np.subtract(result, individual.count(0)/NOBJ)
        return result.tolist()

    toolbox.register("evaluate", evaluate)

    def generateES(bcls, scls, dcls, bdcls, vcls, fcls, fdcls):
        '''
        individual: binary values eg., [0,0,1]
        individual.strategy: unique id of item in dataframe eg., [2, 6, 25]
        individual.feasibleDuration: {uniqueId:duration} 
        individual.feasibleBudget: {uniqueId:weekly budget}
        individual.violation_degree: range of violation of individual [0 inf], 0 is no violation
        '''
        indx = random.sample(range(NUM_REGIONS), NOBJ)
        binaryWeights = tools.mutFlipBit(np.ones(NOBJ).tolist(), MUTPB)[0]
        ind = bcls(binaryWeights)
        ind.strategy = scls(indx)
        ind.feasibleDuration = dcls(dict())
        ind.feasibleBudget = bdcls(dict())
        ind.violation_degree = vcls(0)
        ind.feasible = fcls(0)
        ind.distance = fdcls([0.0]*NOBJ)
        return ind

    def generateFeasible(bcls, scls, dcls, bdcls, vcls, fcls, fdcls):
        feasibleGenerated = False
        ind = None
        while(feasibleGenerated == False):
            ind = generateES(bcls, scls, dcls, bdcls, vcls, fcls, fdcls)
            feasibleGenerated = feasible(ind)
        return ind

    def distance(individual):
        return individual.distance

    if (feasibleOnly):
        toolbox.register("booleanIndividuals", generateFeasible, creator.BooleanIndividuals, 
                                                creator.Strategy, 
                                                creator.DurationFeasibility, 
                                                creator.BudgetFeasibility,
                                                creator.ViolationDegree,
                                                creator.Feasible,
                                                creator.DistanceToFeasible)
    else:
        toolbox.register("booleanIndividuals", generateES, creator.BooleanIndividuals, 
                                                creator.Strategy, 
                                                creator.DurationFeasibility, 
                                                creator.BudgetFeasibility,
                                                creator.ViolationDegree,
                                                creator.Feasible,
                                                creator.DistanceToFeasible)
    
    toolbox.register("population", tools.initRepeat, list, toolbox.booleanIndividuals)

    def max_possible_mixed_utility(weekstoupper, weekstolower):
        sumlist = []
        curr_max = 0
        for i, group in weekstoupper.groupby('RId'):
            score = group['cat_score'].values
            if(score[0] >0 and weekstolower.loc[(weekstolower['RId'] != i & (weekstolower['cat_score'] > 0)), 'cat_score'].count()>0):
                result = score + weekstolower.loc[(weekstolower['RId'] != i ), 'cat_score'].sum()
                result = result[0] if hasattr(result, "__len__") else result
                if result > curr_max:
                    indx = {'u':[i], 'l':weekstolower.loc[((weekstolower['RId'] != i) & (weekstolower['cat_score'] > 0)), 'RId'].values.tolist()}
                    sumlist = [result, indx]
                    curr_max = result
        for i, group in weekstolower.groupby('RId'):
            if(score[0] >0 and weekstoupper.loc[(weekstoupper['RId'] != i & (weekstoupper['cat_score'] > 0)), 'cat_score'].count()>0):
                score = group['cat_score'].values
                result = score + weekstoupper.loc[weekstoupper['RId'] != i, 'cat_score'].sum()
                result = result[0] if hasattr(result, "__len__") else result
                if result > curr_max:
                    indx = {'l':[i], 'u':weekstoupper.loc[((weekstoupper['RId'] != i) & (weekstoupper['cat_score'] > 0)), 'RId'].values.tolist()}
                    sumlist = [result, indx]
                    curr_max = result
        return sumlist
        
    def c1_feasible(individual):
            feasible = True
            if individual.count(1) == 0:
                feasible = False
                individual.violation_degree += np.infty
                individual.distance = list(map(add, individual.distance, [np.infty]*NOBJ))
            return feasible 

    def c2_feasible(individual, scores, selectedPos):
        feasible = True
        info_region_duration = dict()
        weeks_to_upper = scores.loc[scores['category'] =='Weeks to Upper Quantile']
        weeks_to_lower = scores.loc[scores['category'] =='Weeks to Lower Quantile']
        scores_per_reg_upper = weeks_to_upper.loc[weeks_to_upper['RId'].isin(selectedPos)].groupby(['RId', 'category'])['cat_score'].sum().values.tolist()
        scores_per_reg_lower = weeks_to_lower.loc[weeks_to_lower['RId'].isin(selectedPos)].groupby(['RId', 'category'])['cat_score'].sum().values.tolist()

        sum_lower = weeks_to_lower.loc[weeks_to_lower['RId'].isin(selectedPos)].groupby('category')['cat_score'].sum().values[0]
        sum_upper = weeks_to_upper.loc[weeks_to_upper['RId'].isin(selectedPos)].groupby('category')['cat_score'].sum().values[0]
            #if (sum_upper > DURATION and sum_lower > 0 and sum_lower <= DURATION) :
            # failed-> check with mix of upper and lower
            # mixed = max_possible_mixed_utility(weeks_to_upper, weeks_to_lower)
            # if(len(mixed) >0 and mixed[0] > 0 and mixed[0] > sum_lower and mixed[0] <= DURATION):
            #     temp = weeks_to_lower.loc[weeks_to_lower['RId'].isin(mixed[1]['l']), ['RId','cat_score']]
            #     info_region_duration = dict(zip(temp['RId'], temp['cat_score']))
            #     temp = weeks_to_upper.loc[weeks_to_upper['RId'].isin(mixed[1]['u']), ['RId','cat_score']]
            #     info_region_duration.update(dict(zip(temp['RId'], temp['cat_score'])))
            #     individual.feasibleDuration = info_region_duration
            
        if (scores_per_reg_upper.count(0.0) == 0 and sum_upper <= DURATION):
             temp = weeks_to_upper.loc[(weeks_to_upper['RId'].isin(selectedPos)), ['RId','cat_score']]
             info_region_duration = dict(zip(temp['RId'], temp['cat_score']))
             if (len(info_region_duration) < len(selectedPos)):
                 print("Bug found")
             individual.feasibleDuration = info_region_duration
        elif (scores_per_reg_lower.count(0.0)  == 0 and sum_lower <= DURATION):
            temp = weeks_to_lower.loc[(weeks_to_lower['RId'].isin(selectedPos)), ['RId','cat_score']]
            info_region_duration = dict(zip(temp['RId'], temp['cat_score']))
            if (len(info_region_duration) < len(selectedPos)):
                 print("Bug found")
            individual.feasibleDuration = info_region_duration
        else:
            norm = (sum_lower / DURATION) - 1
            individual.violation_degree += norm if norm > 0 else -1*norm
            per_region_sum = weeks_to_lower.groupby('RId')['cat_score'].sum().values
            div = np.divide(per_region_sum ,np.abs(norm)*DURATION) 
            individual.distance = list(map(add, individual.distance, div))
            feasible = False
        return feasible

    def c3_feasible(individual, scores, selectedPos):
        feasible = True
        average_cost_pair = scores.loc[scores['category'] =='average weekly cost']
        weeklycost = average_cost_pair.loc[average_cost_pair['RId'].isin(selectedPos)].groupby('category')['cat_score'].sum().values[0]
        #scores_per_reg = average_cost_pair.loc[average_cost_pair['RId'].isin(selectedPos)].groupby(['RId', 'category'])['cat_score'].sum().values.tolist()

        info_region_budget = dict()
        
        total_budget = weeklycost * DURATION
        #if (scores_per_reg.count(0.0) == 0 and total_budget > 0 and total_budget <= BUDGET):
        if (total_budget > 0 and total_budget <= BUDGET):

            temp = average_cost_pair.loc[((average_cost_pair['RId'].isin(selectedPos))), ['RId','cat_score']]
            info_region_budget = dict(zip(temp['RId'], temp['cat_score']))
            individual.feasibleBudget = info_region_budget
        else:
            feasible = False
            norm = (total_budget / BUDGET) - 1  
            individual.violation_degree += norm if norm > 0 else -1*norm 
            per_region_sum = average_cost_pair.groupby('RId')['cat_score'].sum().map(lambda x: x*DURATION).values
            div = np.divide(per_region_sum, np.abs(norm)*BUDGET) 
            individual.distance = list(map(add, individual.distance, div))
        return feasible
    


    def c4_feasible(individual, scores, selectedPos):
        feasible = True
        scores_per_cat = scores.loc[((scores['category'].isin(categories)) & (scores['RId'].isin(selectedPos)))].groupby('category')['cat_score'].sum().values.tolist()
        norm = scores_per_cat.count(0.0)
        if norm > 0:
            feasible = False
            individual.violation_degree += norm
            per_region_sum = scores.loc[((scores['category'].isin(categories)) & (scores['RId'].isin(selectedPos)))].groupby(['RId','category'])['cat_score'].sum().unstack().values.tolist()
            per_region_count = [regsum.count(0.0) for regsum in per_region_sum]
            div = np.divide(per_region_count, len(categories)) 
            individual.distance = list(map(add, individual.distance, div))
        return  feasible

    def c5_feasible(individual, selectedPos):
        feasible = True
        region_combo_string = [region_index_info[pos] for i, pos in enumerate(selectedPos) if individual[i] == 1]
        unfulfilledCat = []
        for region in region_combo_string:
            unfulfilledCat.append([True for x in region_combo_string if (x != region) and (region not in region_groups[x])])
        norm = sum([len(listElem) for listElem in unfulfilledCat])
        if norm > 0:
            individual.violation_degree += norm
            individual.distance = list(map(add, individual.distance, [0.1]*NOBJ))
            feasible = False
        return feasible

    def feasible(individual):
        feasible = bool(individual.feasible)
        if(feasible == False):
            individual.violation_degree = 0
            individual.distance = [0.0]*NOBJ
            selectedPos = [pos for i, pos in enumerate(individual.strategy) if individual[i]==1 ]
            neededRegions = start_df.loc[(start_df['RId'].isin(individual.strategy))]
            scores_1 = neededRegions.loc[neededRegions['category'].isin(othercategories)]
            c1 = scores_1.loc[:, ['RId','category', 'cat_score']]
            #factors = np.array(individual)[c1.RId.factorize()[0]]
            #c1['cat_score'] = c1['cat_score'] * factors
            scores_2 = neededRegions.loc[neededRegions['category'].isin(considered)]
            #factors = np.array(individual)[scores_2.RId.factorize()[0]]
            #scores_2.loc[:,'cat_score'] = scores_2.loc[:,'cat_score'] * factors
            feasible1 = c1_feasible(individual)
            if(feasible1):
                feasible2 = c2_feasible(individual, c1, selectedPos)
                feasible3 = c3_feasible(individual, c1, selectedPos)
                feasible4 = c4_feasible(individual, scores_2, selectedPos)
                feasible5 = c5_feasible(individual, selectedPos)
            feasible = feasible1 and feasible2 and feasible3 and feasible4 and feasible5
            individual.feasible = int(feasible)
        
        
        return feasible
        
    toolbox.decorate("evaluate", tools.DeltaPenalty(feasible, (np.zeros(NOBJ)).tolist(), distance))
    return toolbox

def main2(toolbox, stats=True, seed=None, verbose=False):
    population = toolbox.population(n=toolbox.pop_size)
    if stats:
        stats = tools.Statistics()
        #stats = tools.Statistics(key=lambda ind: ind.fitness.values)
        stats.register('population', copy.deepcopy)
        stats.register("avg", np.mean, axis=0)
        stats.register("std", np.std, axis=0)
        stats.register("min", np.min, axis=0)
        stats.register("max", np.max, axis=0)
    return algorithms.eaMuPlusLambda(population, toolbox,
                              mu=toolbox.pop_size, 
                              lambda_=toolbox.pop_size, 
                              cxpb=toolbox.cross_prob, 
                              mutpb=toolbox.mut_prob,
                              ngen=toolbox.max_gen,
                              stats=stats,
                              verbose=verbose)
    
    

def main(toolbox, seed=None):
    random.seed(seed)
    stats = tools.Statistics()
    #stats.register('population', copy.deepcopy)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)
    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"
    population = toolbox.population(n=toolbox.pop_size)
    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    print('Starting evaluating fitnesses', end='\r')
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    # Compile statistics about the population
    print('Done evaluating fitnesses', end='\r')
    #record = stats.compile(population)
    print('Done compiling record', end='\r')
    logbook.record(gen=0, evals=len(invalid_ind))
    print('Done parsing record',end='\r')
    print(logbook.stream, end='\r')
     # Begin the generational process
    gen_fitnesses = []
    evaluated_regions = set()
    evaluated_combinations = set()
    for gen in range(1, toolbox.max_gen):
        offspring = algorithms.varAnd(population, toolbox, toolbox.cross_prob, toolbox.mut_prob)
        it = {'gen': gen, 'fitnesses':list(), 'feasibility':list()}
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
            it['fitnesses'].append(fit)
            it['feasibility'].append([np.sum(fit), 'Feasible' if ind.feasible else 'Infeasible'])
            eval_str = [str(pos) for i, pos in enumerate(ind.strategy) if ind[i]==1 ]
            evaluated_regions.update(eval_str)
            evaluated_combinations.add(', '.join(eval_str))
        gen_fitnesses.append(it)
        # Select the next generation population from parents and offspring
        population = toolbox.select(population + offspring, toolbox.pop_size)
        # Compile statistics about the new population
        record = stats.compile(population)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)
        print(logbook.stream,end='\r')
    with open(f'logs/Test_Instance{toolbox.experiment_name}.pickle', 'wb') as f:
        pickle.dump(stats, f, protocol=pickle.HIGHEST_PROTOCOL)
    result = get_results(population, toolbox)
    result['Process'] = gen_fitnesses
    result['TotalRegions'] = len(toolbox.region_indexInfo) 
    result['Evaluated_Regions'] = len(evaluated_regions)
    result['Evaluated_Combinations'] = len(evaluated_combinations)
    return population, logbook, result

def get_results(population, toolbox):
    first_front = tools.emo.sortLogNondominated(population, len(population), first_front_only=True)
    result = dict()
    handled = list()
    region_index_info = toolbox.region_indexInfo
    result['Query'] = toolbox.user_input
    result['Experiment'] = toolbox.experiment_name
    for indx, ind in enumerate(first_front):
        if (ind not in handled and ind.feasible):
            handled.append(ind)
            selectedPos = [pos for i, pos in enumerate(ind.strategy) if ind[i]==1 ]
            recommend_regions = [region_index_info[pos] for pos in selectedPos if pos in ind.feasibleDuration.keys()]
            recommended_duration = {region_index_info[pos]: ind.feasibleDuration[pos] for pos in selectedPos if pos in ind.feasibleDuration.keys()}
            recommended_weekly_budget = {region_index_info[pos]: ind.feasibleBudget[pos] for pos in selectedPos if pos in ind.feasibleBudget.keys()}
            result[indx] = {'regions':recommend_regions, 'total_score':np.sum(toolbox.evaluate(ind)),  'duration':recommended_duration, 'budget_weekly':recommended_weekly_budget}
    filename = f'logs/results/{toolbox.experiment_name}.json'
    if os.path.exists(filename):
        append_write = 'a'
    else:
        append_write = 'w'
    parsed = json.loads([result]) 
    with open(filename, append_write) as convert_file:
        convert_file.write(json.dumps(parsed, indent=4, sort_keys=True))
    return result

def get_feasible_budget(start_df, selectedPos, othercategories):
        neededRegions = start_df.loc[(start_df['RId'].isin(selectedPos))]
        scores = neededRegions.loc[neededRegions['category'].isin(othercategories)]
        c1 = scores.loc[:, ['RId','category', 'cat_score']]
        average_cost_pair = c1.loc[c1['category'] =='average weekly cost']
        temp = average_cost_pair.loc[:, ['RId','cat_score']]
        info_region_budget = dict(zip(temp['RId'], temp['cat_score']))
        return info_region_budget

def get_feasible_stay(strategy, start_df, selectedPos, othercategories, duration):
        neededRegions = start_df.loc[(start_df['RId'].isin(selectedPos))]
        scores = neededRegions.loc[neededRegions['category'].isin(othercategories)]
        c1 = scores.loc[:, ['RId','category', 'cat_score']]
        weeks_to_upper = c1[c1['category'] =='Weeks to Upper Quantile']
        weeks_to_lower = c1.loc[c1['category'] =='Weeks to Lower Quantile']
        scores_per_reg_upper = weeks_to_upper.groupby(['RId', 'category'])['cat_score'].sum().values.tolist()
        sum_upper = weeks_to_upper.groupby('category')['cat_score'].sum().values[0]
        if (scores_per_reg_upper.count(0.0) == 0 and sum_upper <= duration):
             temp = weeks_to_upper.loc[(weeks_to_upper['RId'].isin(selectedPos)), ['RId','cat_score']]
             info_region_duration = dict(zip(temp['RId'], temp['cat_score']))
            
        else:
            temp = weeks_to_lower.loc[(weeks_to_lower['RId'].isin(selectedPos)), ['RId','cat_score']]
            info_region_duration = dict(zip(temp['RId'], temp['cat_score']))
            
        
        return info_region_duration







    
    









