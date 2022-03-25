import numpy as np
import pandas as pd
import yaml
import mysql.connector as connection

def get_start_df(min_start_rank = 0.7):
    user_input = get_config("config.yml")
    input = user_input['input']
    df = get_df(parse_query(input))
    df['weather_score'] = df['weather_score'].astype(dtype=float)
    df['cat_score'] = df['cat_score'].astype(dtype=float)
    considered = input['Preferences']
    pref_weight_dict = get_category_dict(df, considered)
    considered.extend(input['AdditionalPreferences'])
    ranking = get_region_rankings(df.copy(), considered, pref_weight_dict)
    filtered_df = get_new_df(df.copy(), ranking, min_start_rank)
    return filtered_df



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
        'r.average_cost_per_week, t.category, t.score as cat_score, w.month_name, w.score as weather_score '
            'FROM Regions r INNER JOIN ParentRegions p ON p.parentRegion = r.parentRegion '
            'INNER JOIN Travelcategoryscores t ON r.Region = t.region ' 
            'INNER JOIN Weatherscores w on r.Region = w.region '
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
    df['cat_score'] = normalize(df['cat_score'])
    df['weather_score'] = normalize(df['weather_score'])
    df['weather_score'] = weight_scores(df['weather_score'], pref_weight_dict)
    df['cat_score'] = weight_scores(df['cat_score'], pref_weight_dict)
    numerator = df.loc[(df['Region'] == region) & (df['month_name'].isin(considered)), 'weather_score']\
                .map(lambda x: sim_feature(desired_score, x)).sum()\
                + df.loc[(df['Region'] == region) & (df['category'].isin(considered)), 'cat_score']\
                .map(lambda x: sim_feature(desired_score, x)).sum()
    denominator = sum([weight  for pref, weight in pref_weight_dict.items() if pref in considered])
    return numerator/denominator

def get_category_dict(df, preference_List):
    category_weight_dict = dict()
    all_cat = np.concatenate((df.month_name.unique(),df.category.unique()))
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
    ranks = ranks.sort_values(by=['index'], ascending=False)
    return  ranks.set_index('index')[0].to_dict()

def get_new_df(df, ranking_dict, min_rank = 0.7):
    new_dict = {k: v for k, v in ranking_dict.items() if v >= min_rank}
    return df.loc[df['Region'].isin(new_dict.keys())]
    