from keras.optimizers import Adam, Nadam, SGD, RMSprop

from source.grid_search import data
from source.learning_rates.others import choose_lr
from source.load_file import get_model
from utils.utilites import smooth_labels


def search():
    epochs = 20
    grid = {'optimizer': ['adam', 'nadam', 'sgd', 'rmsprop'],
                  'batch_size': [64, 128],
                  'learning_rate': ['cyclic', 'cosine', 'fixed'],
                  'data_type': ['standardized', 'normalized'],
                  'num_of_classes': [10, 20, 50],
                  'class_types': ['top', 'down'],
                  'dropout_rate': [0, 0.2, 0.4],
                  'smooth_factor': [0, 0.05, 0.1, 0.15],
                  'models': ['ANN', 'CNN', 'CNN_RNN' 'RNNLSTM', 'RNNGRU', 'SimpleRNN']}

    for data_type in grid['data_type']:
        for class_type in grid['class_types']:
            for num_classes in grid['num_of_classes']:
                X_train, y_train_cate, X_valid, y_valid_cate, X_test, y_test_cate, weight_class = data(standardized=data_type,
                                                                                                       num_of_classes=num_classes,
                                                                                                       top_classes=class_type)
                for smooth_factor in grid['smooth_factor']:
                    smooth_labels(y_train_cate, smooth_factor)

                    for optimizer in grid['optimizer']:
                        for batch_size in grid['batch_size']:
                            for lr_type in grid['learning_rate']:
                                for model_name in grid['models']:
                                    model = get_model(model_name)
                                    save_dir = f'./model_weights/ns/{data_type}/{model.__str__()}/{class_type}/Adam/{num_classes}'

                                    model(optimizer=optimizer,
                                          batch_size=batch_size,
                                          num_classes=num_classes,
                                          save_dir=save_dir,
                                          epochs=epochs)

                                    model.train(X_train,
                                                y_train_cate,
                                                X_valid,
                                                y_valid_cate,
                                                weight_class, lr_type=lr_type)
                                    model.