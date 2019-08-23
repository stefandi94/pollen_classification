import pandas as pd
from api.train.data_science.data_reader.consts import data_mappings
from api.train.data_science.data_reader.data_loader import DataLoader
from api.train.data_science.models.all_models import RandomForest, XGBoost, ExtraTrees, PermutationImportance
from api.wm.settings import MEDIA_ROOT

N_CHUNKS = 7
NB_FEATURES = 30
def process_model(model, features, labels, features_test, labels_test, permutation=False):
    """
    Process single model. Train the model, get feature importance list, measure accuracy on the test data.
    :param model:
    :param features:
    :param labels:
    :param features_test:
    :param labels_test:
    :param permutation:
    :return:
    """
    model.train(features, labels)
    features_imp = model.get_feature_importance(features, nb_features=NB_FEATURES)
    metrics = model.get_metrics(features_test, labels_test)
    if permutation:
        pi = PermutationImportance(model.model)
        pi.train(features, labels)
        pi_features = pi.get_feature_importance(features, nb_features=NB_FEATURES)
        return pd.concat([features_imp, pi_features]), metrics
    return features_imp.to_frame(), metrics


def train_and_save(frame, target='AP005200_Likely_To_Switch_Investment_Provider_Fin_rank_base_20_AP005200',
                   threshold=15):
    """
    Load data, transform and split data. Train all models and get features.
    :param args:
    :return:
    """
    dataLoader = DataLoader(df=frame)
    dataLoader.label_data(target, threshold)
    features, _, labels, _ = dataLoader.split_data(0.0, target)
    chunk_size = features.shape[0] // 6

    for i in range(1, N_CHUNKS+1):
        X_Features = features.iloc[:i * chunk_size]
        Y_Labels = labels.iloc[:i * chunk_size]

        rf = RandomForest()
        et = ExtraTrees()
        xgb = XGBoost()

        rf_df, m_rf = process_model(rf, X_Features, Y_Labels, X_Features, Y_Labels, permutation=True)
        # et_df, m_et = process_model(et, X_Features, Y_Labels, X_Features, Y_Labels, permutation=False)
        xgb_df, m_xgb = process_model(xgb, X_Features, Y_Labels, X_Features, Y_Labels)

        # xgb.save_model(MEDIA_ROOT + '/xgb.pkl')

        df = pd.concat([rf_df, xgb_df])
        df.reset_index(level=0, inplace=True)
        final = df.groupby(['index']).agg('sum')[0].nlargest(NB_FEATURES)
        print(m_xgb['F1'])
        output = []
        for row in final.reset_index().values:
            item = [data_mappings[row[0]], row[1]]
            output.append(item)

    # print(output)
        df = pd.DataFrame(output)
        df.to_json(MEDIA_ROOT + '/new-features{0}.json'.format(i), orient='values')

    return df

if __name__ == '__main__':
    df = pd.read_csv('/home/mkondic/data/Synechron/wm---client-prospecting/train.csv')
    train_and_save(df)
