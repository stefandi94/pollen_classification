# new_x_train = flatten_data(X_train)
# new_x_valid = flatten_data(X_valid)
#
# print(f'Started PCA fitting at {datetime.now().time()}')
# pca = PCA(n_components=150)
# new_x_train = pca.fit_transform(new_x_train, y_train)
# new_x_valid = pca.transform(new_x_valid)
#
# xgb = XGBClassifier(max_depth=5,
#                     learning_rate=0.06,
#                     n_estimators=500,
#                     verbosity=2,
#                     silent=False,
#                     n_jobs=-1,
#                     nthread=1)
#
# print(f'Fitting started at {datetime.now().time()}')
# xgb.fit(new_x_train, y_train)
# print(f'Fitting finished at {datetime.now().time()}')
# y_pred = xgb.predict(new_x_valid)
#
# print(f'Accuracy score is {accuracy_score(y_valid, y_pred)}')
# print(f'Recall score is: {recall_score(y_valid, y_pred, average="macro")}')
# print(f'F1 score is: {f1_score(y_valid, y_pred, average="macro")}')