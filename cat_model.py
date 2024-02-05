import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt


def calculate_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return rmse, mape


def model_rain(features_df):
    columns = ['dtAB', 'dtCD', 'dtAC', 'dtBD']
    columns_to_delite = [ 'dtAB', 'dtCD', 'dtAC', 'dtBD', 'M', 'p in bar', 'Pump', 'Clyserin', 'Water', 'Glyzerin in g', 'Water in g']
    features_df.drop('ID', axis=1, inplace=True)
    """
    features_df.drop('Subpeak_1_A', axis=1, inplace=True)
    features_df.drop('Subpeak_1_B', axis=1, inplace=True)
    features_df.drop('Subpeak_1_C', axis=1, inplace=True)
    features_df.drop('Subpeak_1_D', axis=1, inplace=True)
    features_df.drop('Subpeak_2_A', axis=1, inplace=True)
    features_df.drop('Subpeak_2_B', axis=1, inplace=True)
    features_df.drop('Subpeak_2_C', axis=1, inplace=True)
    features_df.drop('Subpeak_2_D', axis=1, inplace=True)
    features_df.drop('Subpeak_3_A', axis=1, inplace=True)
    features_df.drop('Subpeak_3_B', axis=1, inplace=True)
    features_df.drop('Subpeak_3_C', axis=1, inplace=True)
    features_df.drop('Subpeak_3_D', axis=1, inplace=True)
    features_df.drop('Subpeak_4_A', axis=1, inplace=True)
    features_df.drop('Subpeak_4_B', axis=1, inplace=True)
    features_df.drop('Subpeak_4_C', axis=1, inplace=True)
    features_df.drop('Subpeak_4_D', axis=1, inplace=True)
    features_df.drop('Subpeak_5_A', axis=1, inplace=True)
    features_df.drop('Subpeak_5_B', axis=1, inplace=True)
    features_df.drop('Subpeak_5_C', axis=1, inplace=True)
    features_df.drop('Subpeak_5_D', axis=1, inplace=True)
    """


    #print(features_df)

    # цели таргета
    y_train_multi = features_df[columns]

    # исключение "лишних" обучающих данных
    X_train = features_df[[col for col in features_df.columns if col not in columns_to_delite]]
    # разделение на обучающую и тестовую выборки
    X_train, X_test, y_train_multi, y_test_multi = train_test_split(X_train, y_train_multi, test_size=0.5, shuffle=True)
    print(X_test)

    # явное указание пула таргета и обучающих данных
    train_pool = Pool(data=X_train, label=y_train_multi)

    # мультитаргетная модель
    model_multi = CatBoostRegressor(iterations=500, depth=3, learning_rate=0.21, loss_function='MultiRMSE',
                                    custom_metric=['MultiRMSE'])
    model_multi.fit(train_pool)

    # на тестовой выборке
    predictions_multi = model_multi.predict(X_test)

    # рассчитываем метрики
    rmse, mape = calculate_metrics(y_test_multi, predictions_multi)
    print(y_test_multi)
    print(f"RMSE on test set: {rmse}")
    print(f"MAPE on test set: {mape}%")
    df_predict = pd.DataFrame(predictions_multi, columns=columns)
    y_test_multi.reset_index(drop=True, inplace=True)

    df_predict.reset_index(drop=True, inplace=True)
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
    axs = axs.flatten()
    # Цикл по столбцам
    for i, column in enumerate(columns):

        axs[i].scatter(y_test_multi[column], df_predict[column], c='blue')
        axs[i].plot([min(y_test_multi[column]), max(df_predict[column])], [min(y_test_multi[column]), max(df_predict[column])], 'r--', label='y=x')

        # Настройка графика
        axs[i].set_title('График tr vs pr', fontsize=16)
        axs[i].set_xlabel('True', fontsize=12)
        axs[i].set_ylabel('Pred', fontsize=12)

        axs[i].grid(True)
        #axs[i].axis('equal')

    plt.tight_layout()
    plt.show()

    feature_importance = model_multi.feature_importances_
    sorted_idx = np.argsort(feature_importance)
    fig = plt.figure(figsize=(16, 10))
    plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
    plt.yticks(range(len(sorted_idx)), np.array(X_test.columns)[sorted_idx])
    plt.title('Feature Importance')
    plt.show()

    return predictions_multi


def model_rain_single(features_df):
    columns = ['A', 'B', 'C', 'D']
    columns_to_delite = ['dtAB', 'dtCD', 'dtAC', 'dtBD', 'M', 'p in bar', 'Pump', 'Clyserin', 'Water', 'Glyzerin in g', 'Water in g']
    features_df.drop('ID', axis=1, inplace=True)



    for col_name in ['A', 'B']:

            # Удаление признаков для B, C и D
            features_df.drop([f'Minimum_{col_name}', f'Maximum_{col_name}', f'Variance_{col_name}',
                              f'Kurtosis_{col_name}', f'Skewness_{col_name}', f'Median_{col_name}',
                              f'Mean_median_{col_name}', f'Std_{col_name}', f'Semimax_{col_name}',
                              f'K_{col_name}', f'Q10_{col_name}', f'Q25_{col_name}', f'Q75_{col_name}',
                              f'Q90_{col_name}', f'Subpeak_1_{col_name}', f'Subpeak_2_{col_name}',
                              f'Subpeak_3_{col_name}', f'Subpeak_4_{col_name}', f'Subpeak_5_{col_name}',
                              f'Semiwidth_{col_name}',  f'AbsEnergy_{col_name[-1]}', f'AbsoluteSumOfChanges_{col_name[-1]}',
                              f'BenfordCorrelation_{col_name[-1]}', f'CidCe_{col_name[-1]}', f'FirstLocMax_{col_name[-1]}',
                              f'MeanChange_{col_name[-1]}', f'MeanSecDerivative_{col_name[-1]}'], axis=1, inplace=True)



    print(features_df)

    # цели таргета
    X = features_df[[col for col in features_df.columns if col not in columns_to_delite]]
    y = features_df['dtCD']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, shuffle=True)

    # Инициализация и обучение CatBoostRegressor
    catboost_model = CatBoostRegressor(iterations=500, learning_rate=0.35, depth=3, loss_function='RMSE')
    catboost_model.fit(X_train, y_train)

    # Предсказание на тестовой выборке
    y_pred = catboost_model.predict(X_test)

    rmse, mape = calculate_metrics(y_test, y_pred)
    print(y_test)
    print(len(y_pred))
    print(f"RMSE on test set: {rmse}")
    print(f"MAPE on test set: {mape}%")


    plt.figure(figsize=(12, 10))

    # Цикл по столбцам

    plt.scatter(y_test, y_pred, c='blue', label='True vs Predicted')
    plt.plot([min(y_test), max(y_pred)], [min(y_test), max(y_pred)], 'r--', label='y=x')

    # Настройка графика
    plt.title('True vs Predicted ', fontsize=16)
    plt.xlabel('True', fontsize=12)
    plt.ylabel('Predicted', fontsize=12)
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
    feature_importance = catboost_model.feature_importances_
    sorted_idx = np.argsort(feature_importance)
    fig = plt.figure(figsize=(16, 10))
    plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
    plt.yticks(range(len(sorted_idx)), np.array(X_test.columns)[sorted_idx])
    plt.title('Feature Importance')
    plt.show()

    return catboost_model


if __name__ == "__main__":
    excel_file_path = 'output.xlsx'  # Укажите путь к вашему файлу
    features_df = pd.read_excel(excel_file_path)

    # Вызов функции model_rain с загруженными данными
    predictions = model_rain_single(features_df)
