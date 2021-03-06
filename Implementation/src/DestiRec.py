from tabnanny import verbose
import time, array, random, copy, math
from collections import Counter
import numpy as np
import logging
import pandas as pd
from operator import add, sub
import random
import PreEmo
from deap import algorithms, base, tools, creator
from collections import Sequence
import json
import os
from csv import DictWriter

class DestiRec:
    
    def __init__(self, user_input, considered, othercategories, needed_months, months_weeks_dict, start_df, region_groups, region_index_info) -> None:
        pd.options.mode.chained_assignment = None
        self.user_input = user_input
        self.considered = considered
        self.othercategories = othercategories
        self.needed_months = needed_months
        self.months_weeks_dict = months_weeks_dict
        self.start_df = start_df
        self.region_groups = region_groups
        self.region_index_info = region_index_info
        self.ref_points = list()

    

    def prepare_toolbox(self, POP_SIZE = 92, GEN=150, NOBJ=4, CROSSPB=0.3, P=12,  feasibleOnly=True, penalizeInObj =False):
        '''
        The prepare toolbox function encapsulates the configuration of the toolbox
        Configuration info are as follows;
        FitnessMoop: Fitness function
        BooleanIndividuals: The actual individual which is a list of boolean values
        Strategy: The index of each Region that are coded as boolean in Individual array
        BudgetFeasibility: Dictionary of Budget Feasibility info of the region combination that is computed during evolutionary process
        DurationFeasibility: Dictionary of Duration Feasibility info of the region combination that is computed during evolutionary process 
        ViolationDegree: Integer value that represents the degree of violation of the individual
        ''' 
        # Get needed dataset 
        user_input = self.user_input
        considered = self.considered
        othercategories = self.othercategories
        needed_months = self.needed_months
        months_weeks_dict = self.months_weeks_dict
        start_df = self.start_df
        region_groups = self.region_groups
        region_index_info = self.region_index_info
        categories = [x for x in considered if x not in needed_months]
        NDIM = len(self.considered)
        MUTPB = 1/NDIM
        NUM_REGIONS = len(region_index_info) 
        BUDGET = user_input['Budget']
        DURATION = user_input['Duration']
        #Start toolbox preparation
        toolbox = base.Toolbox()
        #Save data needed later during evolutionary process into the toolbox
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
        self.ref_points = tools.uniform_reference_points(NOBJ, p=P)
        #We define the maximization objective function as follows in DEAP
        creator.create("FitnessMoop", base.Fitness, weights=(1.0, )*NOBJ)
        #We specify the structure of an individual by calling creator.create, and passing the name and type of the individual
        creator.create("BooleanIndividuals", list, fitness=creator.FitnessMoop)
        creator.create("Strategy", list)
        creator.create("BudgetFeasibility", set)
        creator.create("DurationFeasibility", set)
        creator.create("ViolationDegree", int, typecode='i')
        creator.create("Feasible", int, typecode='i')
        creator.create("DistanceToFeasible", list)
        #Register the mate, mutation and selection operators
        toolbox.register("mate", tools.cxESTwoPoint)
        toolbox.register("mutate", tools.mutFlipBit, indpb=MUTPB)
        toolbox.register("select", tools.selNSGA3WithMemory(self.ref_points))
        
        def evaluate(individual):
            '''The evaluation function to evaluate an individual. This function sums up the score of each region for all categories
            If penalty based constraint handling is used, the violation degree is substracted'''
            scores = start_df.loc[(start_df['RId'].isin(individual.strategy)) & (start_df['category'].isin(considered))].groupby('RId')['cat_score'].sum()
            result = [scores.at[pos]*individual[i] for (i, pos) in enumerate(individual.strategy)]
            if(penalizeInObj):
                res = feasible(individual)
                result = np.subtract(result, individual.violation_degree).tolist()
            return result
        # Register the evaluation function
        toolbox.register("evaluate", evaluate)

        def checkduplicates():
            '''The Flipbit muatation operator sometimes creates duplicate regions in one individual. 
            Here we check for and remove duplicates'''
            def decorator(func):
                def wrapper(*args, **kargs):
                    offspring = func(*args, **kargs)
                    for child in offspring:
                        if not child.fitness.valid:
                            child.feasible = 0
                        
                        if(len(child.strategy)==len(set(child.strategy))):
                            continue
                        dupIndx = [idx for idx, item in enumerate(child.strategy) if item in child.strategy[:idx]]
                        for i in dupIndx:
                            child[i] = child[i]*0
                    return offspring
                return wrapper
            return decorator

        #Deap allows us to decorate functions. Hence we decorate the mate and mutate functions with the check duplicates
        #The decorator is called after each operator is called
        toolbox.decorate("mate", checkduplicates())
        toolbox.decorate("mutate", checkduplicates())


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
            print("Finding next feasible individual..", end='\r')
            while(feasibleGenerated == False):
                
                ind = generateES(bcls, scls, dcls, bdcls, vcls, fcls, fdcls)
                feasibleGenerated = feasible(ind)
            return ind

        def distance(individual):
            return individual.distance
        #We need to register the individual and other parameters belonging to it
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

            
        def c1_feasible(individual):
            '''Constraint 1: Atleast one bit must be true, that is one region is part of the region combination 
            '''
            feasible = True
            if individual.count(1) == 0:
                feasible = False
                individual.violation_degree += np.infty
                individual.distance = list(map(add, individual.distance, [np.infty]*NOBJ))
            return feasible 

        def c2_feasible(individual, scores, selectedPos):
            '''
            Constraint 2: The input duration must be feasbile with the current region combination
            Here we check that the duration for 75% or atleast 25% percent utility in the region does not exceed given duration
            '''
            feasible = True
            info_region_duration = dict()
            weeks_to_upper = scores.loc[scores['category'] =='Weeks to Upper Quantile']
            weeks_to_lower = scores.loc[scores['category'] =='Weeks to Lower Quantile']
            scores_per_reg_upper = weeks_to_upper.loc[weeks_to_upper['RId'].isin(selectedPos)].groupby(['RId', 'category'])['cat_score'].sum().values.tolist()
            scores_per_reg_lower = weeks_to_lower.loc[weeks_to_lower['RId'].isin(selectedPos)].groupby(['RId', 'category'])['cat_score'].sum().values.tolist()

            sum_lower = weeks_to_lower.loc[weeks_to_lower['RId'].isin(selectedPos)].groupby('category')['cat_score'].sum().values[0]
            sum_upper = weeks_to_upper.loc[weeks_to_upper['RId'].isin(selectedPos)].groupby('category')['cat_score'].sum().values[0]
                
            if (scores_per_reg_upper.count(0.0) == 0 and sum_upper <= DURATION):
                temp_df = weeks_to_upper.loc[(weeks_to_upper['RId'].isin(selectedPos)), ['RId','cat_score']]
                info_region_duration = dict(zip(temp_df['RId'], temp_df['cat_score']))
                individual.feasibleDuration = info_region_duration
            
            elif (scores_per_reg_lower.count(0.0)  == 0 and sum_lower <= DURATION):
                temp_df = weeks_to_lower.loc[(weeks_to_lower['RId'].isin(selectedPos)), ['RId','cat_score']]
                info_region_duration = dict(zip(temp_df['RId'], temp_df['cat_score']))
                
                individual.feasibleDuration = info_region_duration
                
            else:
                norm = (sum_lower / DURATION) - 1
                individual.violation_degree += norm **2 if norm > 0 else (-1*norm)**2
                per_region_sum = weeks_to_lower.groupby('RId')['cat_score'].sum()
                result = [per_region_sum.at[pos] for pos in individual.strategy]
                div = np.divide(result,np.abs(norm)*DURATION) 
                individual.distance = list(map(add, individual.distance, div))
                feasible = False
            return feasible

        def c3_feasible(individual, scores, selectedPos):
            '''
            Constraint 3: Total budget needed for the whole trip with the current individual, must not exceed input budget
            '''
            feasible = True
            average_cost_pair = scores.loc[scores['category'] =='average weekly cost']
            weeklycost = average_cost_pair.loc[average_cost_pair['RId'].isin(selectedPos)].groupby('category')['cat_score'].sum().values[0]
            #scores_per_reg = average_cost_pair.loc[average_cost_pair['RId'].isin(selectedPos)].groupby(['RId', 'category'])['cat_score'].sum().values.tolist()

            info_region_budget = dict()
            
            total_budget = weeklycost * DURATION
            #if (scores_per_reg.count(0.0) == 0 and total_budget > 0 and total_budget <= BUDGET):
            if (total_budget > 0 and total_budget <= BUDGET):

                temp_df = average_cost_pair.loc[((average_cost_pair['RId'].isin(selectedPos))), ['RId','cat_score']]
                info_region_budget = dict(zip(temp_df['RId'], temp_df['cat_score']))
                individual.feasibleBudget = info_region_budget
            else:
                feasible = False
                norm = (total_budget / BUDGET) - 1  
                individual.violation_degree += norm**2 if norm > 0 else (-1*norm)**2 
                per_region_sum = average_cost_pair.groupby('RId')['cat_score'].sum().map(lambda x: x*DURATION)
                result = [per_region_sum.at[pos] for pos in individual.strategy]
                div = np.divide(result, np.abs(norm)*BUDGET) 
                individual.distance = list(map(add, individual.distance, div))
            return feasible
        


        def c4_feasible(individual, scores, selectedPos):
            '''
            Constraint 4: All input preference must be fufilled. 
            Here, for each input preference, we check to ensure that atleast one region has score greater than zero 
            '''
            feasible = True
            scores_per_cat = scores.loc[((scores['category'].isin(categories)) & (scores['RId'].isin(selectedPos)))].groupby('category')['cat_score'].sum().values.tolist()
            norm = scores_per_cat.count(0.0)
            if norm > 0:
                feasible = False
                individual.violation_degree += norm**2
                per_region_sum = scores.loc[scores['category'].isin(categories)].groupby(['RId','category'])['cat_score'].sum()
                
                per_region_count = [per_region_sum.at[pos].values.tolist().count(0.0) for pos in individual.strategy]
                div = np.divide(per_region_count, len(categories)) 
                individual.distance = list(map(add, individual.distance, div))
                    
                    
            return  feasible

        def c5_feasible(individual, selectedPos):
            '''
            Constraint 5: Here we ensure that the regions belong to same region group
            '''
            feasible = True
            region_combo_string = [region_index_info[pos] for _, pos in enumerate(selectedPos)]
            unfulfilledCat = []
            for region in region_combo_string:
                unfulfilledCat.append([True for x in region_combo_string if (x != region) and (region not in region_groups[x])])
            norm = sum([len(listElem) for listElem in unfulfilledCat])
            if norm > 0:
                individual.violation_degree += norm ** 2
                individual.distance = list(map(add, individual.distance, [2]*NOBJ))
                feasible = False
            return feasible

        def feasible(individual):
            '''
            Putting it all together, this function is called by the evalution function decorator, and is called before evaluation
            It returns false if one or more of the constraints are violated
            '''
            feasible = False
            individual.violation_degree = 0
            individual.distance = [0.0]*NOBJ
            selectedPos = [pos for i, pos in enumerate(individual.strategy) if individual[i]==1 ]
            neededRegions = start_df.loc[(start_df['RId'].isin(individual.strategy))]
            scores_1 = neededRegions.loc[neededRegions['category'].isin(othercategories)]
            c1 = scores_1.loc[:, ['RId','category', 'cat_score']]
            scores_2 = neededRegions.loc[neededRegions['category'].isin(considered)]
            feasible1 = c1_feasible(individual)
            if(feasible1):
                feasible2 = c2_feasible(individual, c1, selectedPos)
                feasible3 = c3_feasible(individual, c1, selectedPos)
                feasible4 = c4_feasible(individual, scores_2, selectedPos)
                feasible5 = c5_feasible(individual, selectedPos)
            feasible = feasible1 and feasible2 and feasible3 and feasible4 and feasible5
            individual.feasible = int(feasible)
            return feasible

        if not penalizeInObj:
            toolbox.decorate("evaluate", tools.DeltaPenalty(feasible, (np.zeros(NOBJ)).tolist(), distance))
        return toolbox

    
    def main(self, toolbox, algopt=1, seed=None, run=0, input=0, variant=0):
        '''
        This is the main routine of the evolutionary algorithm
        The routine follows the algorithm of the nsga-iii
        '''
        random.seed(seed)
        stats = tools.Statistics()
        #stats.register('population', copy.deepcopy)
        stats.register("avg", np.mean, axis=0)
        stats.register("std", np.std, axis=0)
        stats.register("min", np.min, axis=0)
        stats.register("max", np.max, axis=0)
        logbook = tools.Logbook()
        logbook.header = "gen", "evals", "std", "min", "avg", "max"
        start = time.time()
        population = toolbox.population(n=toolbox.pop_size)
        # Evaluate the individuals with an invalid fitness; Invalid fitnesses only means that the fitness value of the individual hasnt been evaluated
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        # Compile statistics about the population
        #record = stats.compile(population)
        logbook.record(gen=0, evals=len(invalid_ind))
        print(logbook.stream, end='\r')
        # Begin the generational process
        gen_fitnesses = []
        evaluated_regions = set()
        evaluated_combinations = set()
        best_point = worst_point = extreme_points = None
        for gen in range(1, toolbox.max_gen):
            offspring = algorithms.varAnd(population, toolbox, toolbox.cross_prob, toolbox.mut_prob)
            it = {'gen': gen, 'fitnesses':list(), 'feasibility':list()}
            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
                it['fitnesses'].append(ind.fitness.values)
                it['feasibility'].append([np.sum(ind.fitness.values), 'Feasible' if ind.feasible else 'Infeasible'])
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
        end = time.time()
        first_front = tools.emo.sortLogNondominated(population, len(population), first_front_only=True)

        result = dict()
        result["EvaluatedRegions"] = len(evaluated_regions)
        result["EvaluatedCombinations"] = len(evaluated_combinations)
        result["Totaltime"] = round(end - start, 2)
        result["Input"] = input
        result["Variant"] = variant
        result["Nrun"] = run
        result = self.save_results(result, toolbox, first_front)
        result["Process"] = gen_fitnesses

        return population, logbook, result, first_front

    def save_results(self, result, toolbox, first_front):
        handled = list()
        region_index_info = toolbox.region_indexInfo
        header = ['Nrun','Variant','Input','Score','Nregions','EvaluatedRegions','EvaluatedCombinations','Totaltime','SuggestedRegions','SuggestedDuration','SuggestedBudget']
        filename = f'../logs/results/result.csv'

        temp_result = result.copy()
        tmp = dict()
        result['Best_Score'] = 0
        for indx, ind in enumerate(first_front):
            selectedPos = list(set([pos for i, pos in enumerate(ind.strategy) if ind[i]==1]))
            if (selectedPos not in handled and ind.feasible):
                handled.append(selectedPos)
                recommend_regions = [region_index_info[pos] for pos in selectedPos if pos in ind.feasibleDuration.keys()]
                recommended_duration = {region_index_info[pos]: ind.feasibleDuration[pos] for pos in selectedPos if pos in ind.feasibleDuration.keys()}
                recommended_weekly_budget = {region_index_info[pos]: ind.feasibleBudget[pos] for pos in selectedPos if pos in ind.feasibleBudget.keys()}
                
                temp_result['Score'] = np.sum(toolbox.evaluate(ind))
                temp_result['Nregions'] = len(recommend_regions)
                temp_result['SuggestedRegions'] = '; '.join(recommend_regions)
                temp_result['SuggestedDuration'] = '; '.join(recommended_duration.keys())+'-'+'; '.join([str(x) for x in recommended_duration.values()])
                temp_result['SuggestedBudget'] = '; '.join(recommended_weekly_budget.keys())+'-'+'; '.join([str(x) for x in recommended_weekly_budget.values()])
                with open(filename, 'w+', newline='') as convert_file:
                    dictwriter_object = DictWriter(convert_file, delimiter=',', fieldnames=header)
                    dictwriter_object.writerow(temp_result)
                if indx not in tmp.keys():
                    tmp[indx] = list()
                tmp[indx].append(temp_result)
                if temp_result['Score'] > result['Best_Score']:
                    result['Best_Score'] = temp_result['Score']
        result['results'] = tmp
        return result







    
    









