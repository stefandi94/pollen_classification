import pandas as pd
from train.data_science.data_reader.consts import data_mappings
from train.data_science.data_reader.data_loader import DataLoader
from train.data_science.models.all_models import RandomForest, XGBoost, ExtraTrees, PermutationImportance
from wm.settings import MEDIA_ROOT

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
        pi_features = pi.get_feature_importance(features)
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

    rf = RandomForest()
    et = ExtraTrees()
    xgb = XGBoost()

    rf_df, m_rf = process_model(rf, features, labels, features, labels, permutation=True)
    # et_df, m_et = process_model(et, features, labels, features, labels, permutation=False)
    xgb_df, m_xgb = process_model(xgb, features, labels, features, labels)

    xgb.save_model(MEDIA_ROOT + '/xgb.pkl')

    df = pd.concat([rf_df, xgb_df])
    df.reset_index(level=0, inplace=True)
    final = df.groupby(['index']).agg('sum')[0].nlargest(NB_FEATURES)
    print(m_xgb['F1'])
    output = []
    for row in final.reset_index().values:
        item = [data_mappings[row[0]], row[1]]
        output.append(item)

    print(output)
    df = pd.DataFrame(output)
    df.to_json(MEDIA_ROOT + '/features.json', orient='values')
    return df


def load_and_predict(data, target='AP005200_Likely_To_Switch_Investment_Provider_Fin_rank_base_20_AP005200'):
    """
    Load all models and make predictions
    :param data:
    :return:
    """
    xgb = XGBoost()
    xgb.load_model(MEDIA_ROOT + '/xgb.pkl')
    dataLoader = DataLoader(df=data)
    features = dataLoader.clean_data(target)
    xgb_results = xgb.predict_proba(features)
    return xgb_results[:, 0] * 100
