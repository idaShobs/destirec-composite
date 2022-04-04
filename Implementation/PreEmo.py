import numpy as np
import pandas as pd
import yaml
import mysql.connector as connection
import networkx as nx
from networkx.algorithms.traversal.depth_first_search import dfs_tree
import logging
import calendar

df = pd.DataFrame()
filtered_df = pd.DataFrame()

def get_dataset(min_start_rank = 0.7, rank=True):
    '''
    Return: input, considered, othercategories, needed_months, months_weeks_dict, filtered_df
    '''
    
    global filtered_df
    global df
    input = get_user_input()
    query = parse_query(input)
    if(df.empty):
        df = get_df(query)
        df['cat_score'] = df['cat_score'].astype(dtype=float)
        df['cat_score'] = normalize(df['cat_score'])
    considered = get_considered(input)
    all_months = weeks_in_month(input['Year'])
    needed_months = [pref for pref in considered if pref in calendar.month_abbr]
    month_weeks_dict = {pref: all_months[pref] for pref in needed_months}
    pref_weight_dict = get_category_dict(df, considered)
    
    if rank:
        ranking = get_region_rankings(df.copy(), considered, pref_weight_dict)
        new_df = get_ranked_df(df.copy(), input['Preferences'], ranking, min_start_rank)
    else:
        new_df = df.copy()
    filtered_df = get_new_df(new_df)
    othercategories = other_categories(filtered_df, considered)
    return input, considered, othercategories, needed_months, month_weeks_dict, filtered_df 

def get_user_input():
    user_input = get_config("config.yml")
    input = user_input['input']
    return input

def get_considered(input):
    considered = input['Preferences']
    considered.extend(input['AdditionalPreferences'])
    return considered

def other_categories(filtered_df, considered):
    categories = set(filtered_df.category.unique())
    return [x for x in categories if x not in considered]


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

def get_ranked_df(df, considered, ranking_dict, min_rank = 0.7):
    new_dict = {k: v for k, v in ranking_dict.items() if v >= min_rank} 
    df = df.loc[(df['Region'].isin(new_dict.keys())) &  (df['category'].isin(considered))]
    return df

def get_new_df(df):
    new_df = pivot_columns(df)
    return new_df.assign(RId=new_df.groupby('Region').ngroup())

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

def get_region_groups(start_df):
    '''
    return: {region: region group string}'''
    region_groups = dict()
    for region in set(start_df.Region):
        if region not in region_groups.keys():
            region_groups[region] = set()
        region_groups[region].update(set(start_df.loc[(start_df['childRegions'].str.contains(f'(?:{region})', case=False)), 'childRegions']))
    return region_groups

def get_region_index_info(start_df):
    '''
    return: {index:region}
    '''
    return dict(enumerate(start_df.Region.unique()))

def data_selection(start_df, considered):
    data_set = list()
    strategy_set = list()
    data_index_info = dict()
    strategy_index_info = dict()
    cat_data_index = 0
    cat_strategy_index = 0
    region_index_info = None
    region_groups = dict()
    for cat, group in start_df.groupby('category'):
        if cat in considered:
            data_index_info[cat] = cat_data_index
            curr = group.loc[:, 'cat_score'].values.tolist()
            data_set.append(curr)
            cat_data_index +=1
        else:
            strategy_index_info[cat] = cat_strategy_index
            curr = group.loc[:, 'cat_score'].values.tolist()
            strategy_set.append(curr)
            cat_strategy_index += 1
        if region_index_info is None:
            region_index_info = dict(enumerate(group.Region.unique()))
    
    for region in set(start_df.Region):
        if region not in region_groups.keys():
            region_groups[region] = set()
        region_groups[region].update(set(start_df.loc[(start_df['childRegions'].str.contains(region, case=False)), 'childRegions']))
    return data_set, strategy_set, data_index_info, strategy_index_info, region_index_info, region_groups 

##
# ## Helper Functions
##
def init_logging(log_file=None, append=False, console_loglevel=logging.INFO):
    if log_file is not None:
        if append:
            filemode_val = 'a'
        else:
            filemode_val = 'w'
        logging.basicConfig(level=logging.DEBUG,
                            format="%(asctime)s %(levelname)s %(name)s %(message)s",
                            datefmt='%d-%m %H:%M',
                            filename=log_file,
                            filemode=filemode_val)
    else:
        # define a Handler which writes INFO messages or higher to the sys.stderr
        console = logging.StreamHandler()
        console.setLevel(console_loglevel)
        # set a format which is simpler for console use
        formatter = logging.Formatter("%(message)s")
        console.setFormatter(formatter)
        # add the handler to the root logger
        logging.getLogger('').addHandler(console)
    global LOG
    LOG = logging.getLogger(__name__) 

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

def weeks_in_month(year):
    month_weeks = dict()
    for month in range(1, 13):
        weeks = len(calendar.monthcalendar(year, month))
        month_weeks[calendar.month_abbr[month]] = weeks
    return month_weeks




    