import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def model_rain(features_df):

    columns = ['dtAB', 'dtCD', 'dtAC', 'dtBD']

    #Цели таргета
    y_train_multi = features_df[['dtAB', 'dtCD', 'dtAC', 'dtBD']]

    #Исключение "лишних" обучающих данных
    X_train = features_df[[col for col in features_df.columns if col not in columns]]

    #Явное указание пула таргета и обучающих данных
    train_pool = Pool(data=X_train, label=y_train_multi)

    #Мультитаргетная модель
    model_multi = CatBoostRegressor(iterations=1000, depth=6, learning_rate=0.5, loss_function='MultiRMSE',
                                    custom_metric=['MultiRMSE'])
    model_multi.fit(train_pool)
    predictions_multi = model_multi.predict(X_train)

    return predictions_multi