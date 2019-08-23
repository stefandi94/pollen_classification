import json

import pandas as pd
from django.http import HttpResponse
from django.shortcuts import render
from train.data_science.data_reader.consts import data_mappings
from train.data_science.main import train_and_save, load_and_predict
from train.forms import DocumentForm
from train.utils import nlg_summary, table_data, tracking, graphs_data, get_features
from wm.settings import MEDIA_ROOT


def features(request):
    """
    Endpoint for the latest driving factors.
    :param request:
    :return:
    """
    try:
        df = pd.read_json(MEDIA_ROOT + '/features.json')
        df.columns = ['label', 'value']
        return HttpResponse(df.to_json(orient='records'), content_type="application/json")
    except:
        return HttpResponse({}, content_type="application/json")


def train(request):
    """
    Endpoint for training.
    :param request:
    :return:
    """
    if request.method == 'POST':
        form = DocumentForm(request.POST, request.FILES)
        if form.is_valid():
            df = pd.read_csv(request.FILES['docfile'])
            rez = train_and_save(df)
            return HttpResponse(rez.reset_index().to_json(orient='values'), content_type="application/json")
    else:
        form = DocumentForm()  # A empty, unbound form
    # Render list page with the documents and the form
    return render(request, 'train.html', {'form': form})


def predict(request):
    """
    Endpoint for prediction. Creates conversion probability using pre-trained model.
    Returns NLG summary report, tracking data, graphs data and table data for Results page.
    :param request:
    :return:
    """
    if request.method == 'POST':
        form = DocumentForm(request.POST, request.FILES)

        if form.is_valid():
            df = pd.read_csv(request.FILES['docfile'])
            results = load_and_predict(df)

            df = df[list(data_mappings.keys())]
            df.rename(columns=data_mappings, inplace=True)
            df['probability'] = results
            response = dict()
            response['summary'] = nlg_summary(df)
            response['results'] = json.loads(table_data(df))
            response['tracking'] = tracking(df)
            response['graphs'] = graphs_data(df)
            return HttpResponse(json.dumps(response), content_type="application/json")
        else:
            return HttpResponse(json.dumps({'valid': False}), content_type="application/json")
    else:
        form = DocumentForm()  # A empty, unbound form
        # Render list page with the documents and the form
    return render(request, 'predict.html', {'form': form})


def history(request):
    """
    Endpoint for Performance page. Returns historical accuracy over 7 years, including the number of data used for
    training and driving factors
    :param request:
    :return:
    """
    hist_accuracy = dict()
    hist_accuracy['2017 Q2'] = {'accuracy': 0.88892, 'amount': 10000, 'features': get_features(1)}
    hist_accuracy['2017 Q3'] = {'accuracy': 0.90568, 'amount': 20000, 'features': get_features(2)}
    hist_accuracy['2017 Q4'] = {'accuracy': 0.92886, 'amount': 30000, 'features': get_features(3)}
    hist_accuracy['2018 Q1'] = {'accuracy': 0.94895, 'amount': 40000, 'features': get_features(4)}
    hist_accuracy['2018 Q2'] = {'accuracy': 0.96009, 'amount': 50000, 'features': get_features(5)}
    hist_accuracy['2018 Q3'] = {'accuracy': 0.97013, 'amount': 60000, 'features': get_features(6)}
    hist_accuracy['2018 Q4'] = {'accuracy': 0.98183, 'amount': 70000, 'features': get_features(7)}

    return HttpResponse(json.dumps(hist_accuracy, sort_keys=True), content_type="application/json")
