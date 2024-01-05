import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def model_rain(features_df):
    """
    # Получаем последнюю букву из названия столбца таргета
    last_letter = target_column[-1]
    pred_last_letter = target_column[-2]
    # Чекаем ласт букву и подтягиваем необходимые столбцы
    columns_to_exclude = ['dtAB', 'dtCD', 'dtAC', 'dtBD', 'ID']
    feature_columns = [col for col in features_df.columns if col not in columns_to_exclude and (col.endswith(last_letter) or col.endswith(pred_last_letter))]
    df[f'{Channel%}]
    #print(feature_columns)

    X = features_df[feature_columns]
    Y = features_df[[target_column]]

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.5, random_state=42)

    model = CatBoostRegressor()

    model.fit(X_train, y_train)

    # Получение прогнозов на тестовом наборе

    y_pred = model.predict(X_test)

    print(y_pred)

    mse = mean_squared_error(y_test, y_pred)


    print(f'Ошибка (среднеквадратичное): {mse}')
    """
    columns = ['dtAB', 'dtCD', 'dtAC', 'dtBD']


    y_train_multi = features_df[['dtAB', 'dtCD', 'dtAC', 'dtBD']]
    X_train = features_df[[col for col in features_df.columns if col not in columns]]
    train_pool = Pool(data=X_train, label=y_train_multi)
    model_multi = CatBoostRegressor(iterations=1000, depth=6, learning_rate=0.1, loss_function='MultiRMSE',
                                    custom_metric=['MultiRMSE'])
    model_multi.fit(train_pool)
    predictions_multi = model_multi.predict(X_train)

    return predictions_multi