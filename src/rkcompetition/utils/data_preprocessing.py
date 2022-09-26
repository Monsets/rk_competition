import os
import re

import pandas as pd
import numpy as np
import umap.umap_ as umap

from sklearn.manifold import TSNE
from sklearn.covariance import EllipticEnvelope
from sklearn.cluster import KMeans


clustering_params_inner = {
    'second_df': None,
    'embedding_components': 3,
    'n_clusters': 20,
    'outlier_strength': 0.15,
    'min_number_of_clicks_for_user': 150,
    'perplexity': 10
}

clustering_params_outer = {
    'second_df': None,
    'embedding_components': 4,
    'n_clusters': 20,
    'outlier_strength': 0.15,
    'min_number_of_clicks_for_user': 100,
    'perplexity': 10
}


def create_sample(clf, x_test, sample, thresh, name):
    '''
        Creates sample predictions.

        :param clf - classifier
        :param x_test - test dataset
        :param thresh - threshold to use for classes
        :param sample - sample df from a competition
        :param name - name for a sample
    '''

    x_test = x_test.drop(['blocked'], axis=1)
    sample['blocked'] = clf.predict_proba(x_test)
    sample['blocked'] = (sample['blocked'] < thresh).astype(int)
    print(sample.blocked.sum())
    sample.to_csv(name, index=False)


def get_cluster_transformation(df, quantization=False, method='t-sne', embedding_components=2, n_clusters=30,
                               outlier_strength=0,
                               min_number_of_clicks_for_user=50, perplexity=3):
    '''
        Computes transformation from event to a cluster

        :param df - dataframe of a wide format
        :return
    '''
    pivot = df.copy()
    pivot = pivot.T
    # Select frame with min number of clicks
    pivot_locked = pivot.loc[:, np.sum(pivot, axis=0) > min_number_of_clicks_for_user]


    # Embedding
    if method == 't-sne':
        embedder = TSNE(n_components=embedding_components,
                        learning_rate=100, init='random',
                        perplexity=perplexity, random_state=42, n_jobs=-1)
    elif method == 'umap':
        embedder = umap.UMAP(random_state=42, n_components=embedding_components, n_neighbors=perplexity)
    else:
        raise ValueError('Embedding method is not defined')

    X_embedded = embedder.fit_transform(pivot_locked)

    # outlier detection
    if outlier_strength:
        take_index = np.where(EllipticEnvelope(assume_centered=True,
                                               contamination=outlier_strength).fit_predict(X_embedded) == 1)
        X_embedded = X_embedded[take_index]
    else:
        take_index = np.arange(pivot_locked.shape[0])

    # Clustering
    km = KMeans(n_clusters=n_clusters, random_state=0).fit(X_embedded)
    preds = km.predict(X_embedded)
    clusters = np.zeros((pivot_locked.shape[0])) - 1
    clusters[take_index] = preds
    answer = pd.DataFrame()
    answer['event_type'] = pivot_locked.index
    answer['cluster'] = clusters
    answer = answer.set_index('event_type')

    return answer, X_embedded


def transform_event_to_cluster(df, second_df=None, embedding_components=4, n_clusters=20,
                               outlier_strength=0.15, min_number_of_clicks_for_user=100, quantization=False,
                               perplexity=10, method='umap'):
    '''
        Transforms event statistics to cluster statistics
    '''
    if second_df is None:
        second_df = df.copy()

    clusters, embed = get_cluster_transformation(second_df, embedding_components=embedding_components,
                                                 n_clusters=n_clusters,
                                                 outlier_strength=outlier_strength,
                                                 min_number_of_clicks_for_user=min_number_of_clicks_for_user,
                                                 quantization=False,
                                                 perplexity=perplexity, method=method)
    # long to wide
    df_for_clustering = df.copy()
    df_for_clustering = df_for_clustering.T
    # transform original data
    df_for_clustering.index = clusters.cluster
    transformed_data = df_for_clustering.groupby('cluster').sum().T

    return transformed_data


def long_to_wide(time_series_df: pd.DataFrame, index='contract_id', columns='event_type',
                 agg_func='sum', values='val') -> pd.DataFrame:
    '''
        Transforms time-series dataframe to a long format table

        :param time_series - dataframe to transform
        :param **params - specific to a pivot_table params
        :return long-format dataframe
    '''
    time_series_df['val'] = 1
    pivot = pd.pivot_table(time_series_df, index=index,
                           columns=columns, aggfunc=agg_func, values=values, fill_value=0)
    pivot = pivot.fillna(0)

    return pivot


def col_difference_mean(df, col):
    df['{}_mean'.format(col)] = np.mean(df[col]) - df[col]
    df['{}_divide_mean'.format(col)] = df[col] / np.mean(df[col])

    return df


def long_to_wide_with_statistics(df, prefix='', cluster_data=False, second_df=None, get_difference=False, **params):
    pivot = long_to_wide(df)

    if cluster_data:
        pivot = transform_event_to_cluster(df, second_df=second_df, **params)
        pivot['perc_of_outlier_action'] = pivot[-1] / pivot.sum(axis=1)

    # get action statistics
    total_actions = np.sum(pivot, axis=1)
    unique_actions = np.sum(pivot > 0, axis=1)
    max_actions = np.max(pivot, axis=1)
    min_actions = np.min(pivot, axis=1)
    mean_actions = np.mean(pivot, axis=1)
    std_actions = np.std(pivot, axis=1)
    pivot = pivot.fillna(0)
    # minmax scaler
    pivot = pivot.div(pivot.sum(axis=1), axis=0)
    # get difference statistics
    pivot['total_actions'] = total_actions
    pivot['max_actions'] = max_actions
    pivot['min_actions'] = min_actions
    pivot['mean_actions'] = mean_actions
    pivot['std_actions'] = std_actions
    pivot['unique_actions'] = unique_actions
    pivot['unique_actions_fraction'] = pivot['unique_actions'] / pivot['total_actions']
    if get_difference:
        pivot = col_difference_mean(pivot, 'total_actions')
        pivot = col_difference_mean(pivot, 'unique_actions')
        pivot = col_difference_mean(pivot, 'unique_actions_fraction')
    # get unique actions

    pivot.columns = [prefix + '_' + str(col) for col in pivot.columns]

    return pivot


def get_week_statistic(df, prefix='', get_difference=False):
    df.event_date = pd.to_datetime(df.event_date)
    # get last action week
    df_weeks = df.merge(df.groupby('contract_id').event_date.max(), left_on='contract_id', right_index=True, )
    # datetime transform
    df_weeks.event_date_x = pd.to_datetime(df_weeks.event_date_x)
    df_weeks.event_date_y = pd.to_datetime(df_weeks.event_date_y)
    # difference till last action
    df_weeks['weeks_until_last_action'] = df_weeks.event_date_y.dt.week - df_weeks.event_date_x.dt.week

    # action statistic

    actions_per_last_date = df_weeks.groupby(['contract_id', 'weeks_until_last_action']).count().reset_index()
    actions_per_last_date['val'] = 1
    action_week_tab = pd.pivot_table(actions_per_last_date, index='contract_id', columns='weeks_until_last_action',
                                     values='val', aggfunc='sum')
    action_week_tab = action_week_tab.fillna(0)
    action_week_tab = action_week_tab.div(action_week_tab.sum(axis=1), axis=0)

    # action difference for every week

    for i in range(1, np.max(df_weeks.weeks_until_last_action) + 1):
        action_week_tab['{}_{}'.format(i, i - 1)] = action_week_tab[i] / (action_week_tab[i - 1] + 1)
        action_week_tab = action_week_tab.fillna(0)

    action_week_tab = (action_week_tab.
                       merge(df_weeks.loc[:, ['contract_id', 'weeks_until_last_action']].
                             groupby('contract_id').max(), left_index=True, right_index=True))

    action_week_tab['num_active_weeks'] = (action_week_tab.iloc[:, :9] > 0).sum(axis=1)

    # get last active week data
    t = (action_week_tab > 0).loc[:, ::-1].idxmax(axis=1)
    action_week_tab['last_action_activity'] = action_week_tab.lookup(t.index, t.values)

    if get_difference:
        cols = list(action_week_tab)
        for col in cols:
            action_week_tab = col_difference_mean(action_week_tab, col)

    action_week_tab.columns = [prefix + '_' + str(col) for col in action_week_tab.columns]

    return action_week_tab


def read_raw_data(path_to_raw_data ='../') -> tuple:
    '''
        Read raw data with unification step

        :return all raw dataframes
    '''

    # reading
    inner_actions = pd.read_csv(os.path.join(path_to_raw_data, 'log.csv'))
    outer_actions = pd.read_csv(os.path.join(path_to_raw_data, 'named.csv'))
    type_contract = pd.read_csv(os.path.join(path_to_raw_data, 'type_contract.csv'))
    train = pd.read_csv(os.path.join(path_to_raw_data, 'train_dataset_train.csv'))
    sample = pd.read_csv(os.path.join(path_to_raw_data, 'sample_solution.csv'))
    # unification
    outer_actions = outer_actions.rename(columns={'date': 'event_date', 'url': 'event_type'})
    # take only that are in test dataset
    outer_actions = outer_actions.loc[
        outer_actions.event_type.isin(outer_actions.loc[outer_actions.contract_id.isin(sample.contract_id)].event_type)]
    inner_actions = inner_actions.loc[
        inner_actions.event_type.isin(inner_actions.loc[inner_actions.contract_id.isin(sample.contract_id)].event_type)]

    # append test set to process it the same as training one
    sample['blocked'] = -1
    # train = train.loc[train.contract_id < 55e3]
    train = train.append(sample)

    # select only info with users presented in dataset|
    outer_actions_in_train = outer_actions.loc[outer_actions.contract_id.isin(train.contract_id)]

    return inner_actions, outer_actions, type_contract, train, sample, outer_actions_in_train

def unify_drop_columns(df: pd.DataFrame, col_tag: str = '', pattern: str = '', col_name: str = '', drop: bool = True) -> pd.DataFrame:
    '''
        Unifies different columns through re pattern into one.

        :param df - dataframe to unify
        :param col_tag - substring as a key for unification
        :param pattern - re pattern as a key for unification
        :param col_name - new name of a unified column
        :param drop - drop columns that are being unified
        :return pd.DataFrame - a transformed dataframe
    '''

    # find by a substring if it is not empty
    if col_tag:
        cols = [col for col in df.columns if col_tag in str(col)]
    # otherwise use regular expression
    else:
        cols = [col for col in df.columns if re.match(pattern, str(col)) is not None]
    # use substring as a new column name
    if col_tag:
        col_name = col_tag
    df[col_name] = np.sum(df[cols], axis = 1)
    # drop redundant columns
    if drop:
        df = df.drop(cols, axis = 1)

    return df
