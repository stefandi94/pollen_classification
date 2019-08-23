import json

import inflect
import numpy as np
import pandas as pd
from train.data_science.data_reader.consts import data_mappings
from wm.settings import MEDIA_ROOT

p = inflect.engine()

RM = {
    0: 'Brittany Patterson',
    1: 'Cheryl Miller',
    2: 'Barbara Hoover',
    3: 'Brittany Tran',
    4: 'Catherine Garcia',
    5: 'John Perry',
    6: 'Janet McCann',
    7: 'Andrew Perkins',
    8: 'Stephanie Mccoy',
    9: 'Caitlin Khan'
}


def create_graph(graph):
    """
    Helper function to transform data in the format required for histogram based graphs
    :param graph:
    :return:
    """
    data = []
    for i, val in enumerate(list(graph.values)):
        item = dict()
        item['name'] = '>{0}%'.format(i * 10)
        item['value'] = int(val)
        data.append(item)

    out = dict()
    out['data'] = data
    return out


def get_geographic_data(df, filter_results=False):
    """
    Create data for geo distribution chart
    :param df: uploaded df
    :param filter_results: if true use only identified
    :return:
    """
    if filter_results:
        df = df.loc[df['probability'] >= 80]
    return df.groupby('STATE')['Total Liquid Investible Assets'].count().to_dict()


def liquid_investible_assets(df, filter_results=False):
    if filter_results:
        df = df.loc[df['probability'] >= 80]
    sums = df['Total Liquid Investible Assets'].replace(-1, 0).sort_values(ascending=False).cumsum()
    total = df['Total Liquid Investible Assets'].sum()
    percentage = (sums / total) * 100
    ids = range(1, len(sums) + 1)

    out_df = pd.DataFrame()
    out_df['sums'] = sums
    out_df['ids'] = ids
    out_df['percentage'] = percentage

    return json.loads(out_df.to_json(orient='records'))


def graphs_data(df):
    """
    Generate data for all graphs. Probability distribution, Net worth distribution and geo chart.
    :param df:
    :return:
    """
    graph1 = df["probability"].groupby(pd.cut(df["probability"], np.arange(0, 110, 10))).count()  ## dist
    graph2 = df.groupby(pd.cut(df["probability"], np.arange(0, 110, 10)))['Estimated Net Worth (B) (Financial)'].sum()
    graph3 = df.groupby(pd.cut(df["probability"], np.arange(0, 110, 10)))['Total Liquid Investible Assets'].mean()

    out = dict()
    out['probability'] = create_graph(graph1)
    out['net_worth'] = create_graph(graph2)
    out['avg_invest_assets'] = create_graph(graph3)
    out['map'] = get_geographic_data(df, filter_results=True)
    out['map_total'] = get_geographic_data(df)
    out['liquid_assets'] = liquid_investible_assets(df, filter_results=True)
    out['liquid_assets_total'] = liquid_investible_assets(df, filter_results=False)

    return out


def table_data(df):
    """
    Serialize data for the Results page. The result is all records with conversion probability > 80.
    :param df:
    :return:
    """
    # df = df.loc[df['probability'] >= 80]  # or no
    table_columns = ['probability', data_mappings['PREFIXTTL'], data_mappings['FIRSTNAME'], data_mappings['LASTNAME'],
                     data_mappings['ADDRESS'], data_mappings['STATE'],
                     data_mappings['Estimated Net Worth (B) (Financial)'],
                     data_mappings['AP001370_Total_Liquid_Investible_Assets_V2_dollar_beggc284'],
                     data_mappings['#8688: Gender - Input Individual'],
                     data_mappings['#8623_1: Date of Birth - Input Individual (YYYY/MM) - Year']]

    df = df[table_columns]
    df.loc[df['probability'] >= 80, 'status'] = 'identified'
    df['Age'] = 2019 - df['Age'].astype(int)
    df['Age'].replace(2020, 50, inplace=True)
    df['idx'] = np.random.randint(0,10,size=(len(df), 1))
    df['RM'] = df.idx.apply(lambda x: RM[x])
    df.drop(['idx'], axis=1, inplace=True)
    return df.to_json(orient='records')


def tracking(df):
    """
    Fill tracking data. All values are zero except identification
    :param df:
    :return:
    """
    response = dict()
    response['total_prospects'] = df.shape[0]
    response['identification'] = df.loc[df['probability'] >= 80].shape[0]
    response['out-reach'] = 0
    response['on-boarding'] = 0
    response['converted'] = 0

    return response


def get_features(id):
    """
    Helper function to load historical features
    :param id:
    :return:
    """
    df = pd.read_json(MEDIA_ROOT + '/new-features{0}.json'.format(id))
    df.columns = ['label', 'value']
    return json.loads(df.to_json(orient='records'))


def nlg_summary(df):
    """
    Generate NLG summary report based on uploaded file and predictions
    :param df:
    :return:
    """
    total = df.shape[0]
    df = df.loc[df['probability'] >= 80]  # filtering
    likely_to_convert = df.shape[0]

    has_boat = df.loc[(df['Total Liquid Investible Assets'] > 5000000) & (df['Boat Owner'] == 1)].shape[0]
    sum_assets = df['Total Liquid Investible Assets'].sum()
    sum_assets = int(sum_assets / 1000000)
    text1 = "Based on our analysis of {0} prospects, we’ve identified {1} prospects with an 80% or greater conversion probability".format(
        total, likely_to_convert)
    text2 = 'Overall, we expect ${0} million in new assets to be generated based on expected conversion probabilities and a base fee of 1%.'.format(
        sum_assets)
    percent = float(likely_to_convert) / float(total)
    percent_boat = float(has_boat) / float(likely_to_convert)

    percent = round(percent * 100, 2)
    percent_boat = round(percent_boat * 100, 2)
    text3 = 'Of the {0}% of prospects likely to convert, {1}%  have total liquid investible assets > $5m and own a boat.'.format(
        percent, percent_boat)
    # final output
    summary = dict()
    summary['1'] = text1
    summary['2'] = text2
    summary['3'] = text3
    return summary
