import numpy as np
import pandas as pd
import yaml
import mysql.connector as connection
import networkx as nx
from networkx.algorithms.traversal.depth_first_search import dfs_tree

def get_start_df(min_start_rank = 0.7):
    user_input = get_config("config.yml")
    input = user_input['input']
    query = parse_query(input)
    df = get_df(query)
    df['cat_score'] = df['cat_score'].astype(dtype=float)
    df['cat_score'] = normalize(df['cat_score'])
    considered = input['Preferences']
    pref_weight_dict = get_category_dict(df, considered)
    considered.extend(input['AdditionalPreferences'])
    ranking = get_region_rankings(df.copy(), considered, pref_weight_dict)
    filtered_df = get_new_df(df.copy(), input['Preferences'], ranking, min_start_rank)
    return input, considered, filtered_df



def get_config(path):
    parsed_yaml = None
    with open(path, 'r') as stream:
        try:
            parsed_yaml=yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return parsed_yaml




def get_df(query):
    df = None
    config = get_config("special_files/dbconfig.yml")
    try:  
        db_connection = connection.connect(host=config['host'], database=config['database'], user=config['user'], passwd=config['passwd'],use_pure=True)
        df = pd.read_sql(sql=query, con=db_connection)
        db_connection.close()
    except Exception as e:
        db_connection.close()
        print(str(e))
    return df

    __
def parse_query(user_input):
    
    queryParam = ', '.join(f'"{x}"' for x in user_input['Exclude'])
    query = ('Select r.Region, r.weeks_to_upperQuantile_utility, r.weeks_to_lowerQuantile_utility, ' 
        'r.average_cost_per_week, t.category, t.score as cat_score, p.childRegions '
            'FROM Regions r INNER JOIN ParentRegions p ON p.parentRegion = r.parentRegion '
            'INNER JOIN Travelcategoryscores t ON r.Region = t.region ' 
            'WHERE r.Region NOT IN ({0}) AND r.parentRegion NOT IN ({1})').format(queryParam, queryParam)
    return query

def weight_scores(df_column, pref_weight_dict):
    df_column *= df_column.map(pref_weight_dict).fillna(1)
    return df_column

def normalize(df_column):
    return (df_column - df_column.min()) / (df_column.max() -  df_column.min())

def sim_feature(desired_score, actual_score):
    return 1 - (abs(desired_score - actual_score) / max(desired_score, actual_score))

def region_similarity(df, region, considered, pref_weight_dict):
    '''
    pref_weight_dict = {'hiking': 2, 'beach':2, 'watersports':2, 'culinary':2, 'entertainment':2, ''}
    considered = preference list and safety and crime
    '''
    desired_score = 1
    df['cat_score'] = weight_scores(df['cat_score'], pref_weight_dict)
    numerator = df.loc[(df['Region'] == region) & (df['category'].isin(considered)), 'cat_score']\
                .map(lambda x: sim_feature(desired_score, x)).sum()
    denominator = sum([weight  for pref, weight in pref_weight_dict.items() if pref in considered])
    return numerator/denominator

def get_category_dict(df, preference_List):
    category_weight_dict = dict()
    all_cat = df.category.unique()
    for i in all_cat:
        if i == 'Safety from crime':
            category_weight_dict[i] = 1
        elif (i in preference_List):
            category_weight_dict[i] = 2
        else:
            category_weight_dict[i] = 0
    return category_weight_dict

def get_region_rankings(df, considered, pref_weight_dict):
    regions = df.Region.unique()
    ranking_dict = dict()
    for region in regions:
        ranking_dict[region] = region_similarity(df, region, considered, pref_weight_dict)
    ranks = pd.DataFrame.from_dict(ranking_dict, orient='index').reset_index()
    ranks[0] = normalize(ranks[0])
    # ranks = ranks.sort_values(by=['index'], ascending=False)
    return  ranks.set_index('index')[0].to_dict()

def get_new_df(df, considered, ranking_dict, min_rank = 0.7):
    new_dict = {k: v for k, v in ranking_dict.items() if v >= min_rank} 
    df = df.loc[(df['Region'].isin(new_dict.keys())) &  (df['category'].isin(considered))]
    return pivot_columns(df)

def pivot_columns(df):
    group = df.groupby('Region')
    df1 = group[['Region','weeks_to_lowerQuantile_utility', 'childRegions']].first()
    df1['category'] = 'Weeks to Upper Quantile'
    df2 = group[['Region','weeks_to_upperQuantile_utility', 'childRegions']].first()
    df2['category'] = 'Weeks to Lower Quantile'
    df5 = group[['Region', 'average_cost_per_week', 'childRegions']].first()
    df5['category'] = 'average weekly cost'
    df3 = pd.concat([df5.rename(columns={'average_cost_per_week':'cat_score'}), 
                    df1.rename(columns={'weeks_to_lowerQuantile_utility':'cat_score'}), 
                    df2.rename(columns={'weeks_to_upperQuantile_utility':'cat_score'})], ignore_index=True)
    df4 = pd.concat([df.loc[:, ('Region', 'category', 'cat_score', 'childRegions')], df3], ignore_index=True)
    return df4.sort_values(by=['Region'], ascending=True)

##
# ## Helper Functions
##

def get_descendants(parent, g, x):
    descendants = list(dfs_tree(g, parent).edges())
    return ', '. join([x[1] for x in descendants])

def generate_childRegions(df):
    g = nx.DiGraph()
    g.add_edges_from(df[['parentRegion', 'Region']].to_records(index=False))
    x = dfs_tree(g, 'World')
    df['childRegions'] = df["parentRegion"].map(lambda parent: get_descendants(parent, g, x))
    return df

def update_descendants():
    config = get_config("special_files/dbconfig.yml")
    query = 'Select Id, p.parentRegion, r.Region, childRegions from ParentRegions p INNER JOIN Regions r ON p.parentRegion = r.parentRegion'
    try:  
        db_connection = connection.connect(host=config['host'], database=config['database'], user=config['user'], passwd=config['passwd'],use_pure=True)
        df = pd.read_sql(sql=query, con=db_connection)
        db_connection.close()
        df = generate_childRegions(df)
        df.to_csv('descendants.csv', index=False)
    except Exception as e:
        db_connection.close()
        print(str(e))
    