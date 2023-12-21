import pandas as pd
from tsfresh import extract_features, select_features
from tsfresh.feature_extraction import feature_calculators
from tsfresh.utilities.dataframe_functions import impute


def extract_features_df(df):

    #settings = ComprehensiveFCParameters()

# Извлекаем характеристики
    #extracted_features = extract_features(df, column_id='ID', column_sort='Time', column_value='Channel A', default_fc_parameters=settings)

# Рассчитываем усредненные характеристики
    #averaged_features = extracted_features.groupby('ID').mean()
    """
# Добавляем усредненные характеристики в исходный DataFrame
    #df = pd.merge(df, averaged_features, left_on='ID', right_index=True, how='left')
    #print(df)
    features = {
        'min': feature_calculators.minimum,
        'max': feature_calculators.maximum,
        'variance': feature_calculators.variance,
        'skewness': feature_calculators.skewness,
        'kurtosis': feature_calculators.kurtosis,
    }


# Теперь df содержит усредненные характеристики от сигнала A
    for column in df:
        if column != 'Time' and column != 'ID':
            df['min'] = df['A'].apply(feature_calculators.minimum, raw=True)
            df['max'] = df['A'].apply(feature_calculators.maximum, raw=True)
            df['variance'] = df['A'].apply(feature_calculators.variance, raw=True)
            df['skewness'] = df['A'].apply(feature_calculators.skewness, raw=True)
            df['kurtosis'] = df['A'].apply(feature_calculators.kurtosis, raw=True)
    """


    extracted_features = extract_features(
        df,
        column_id='ID',
        column_value='Channel A'  # !
    )




    # Импьют (заполнение) пропущенных значений
    imputed_features = impute(extracted_features)

    # Выбор нужных характеристик (можно заменить на другие, если необходимо)
    selected_features = select_features(imputed_features, df['Channel A'])

    # Добавление усредненных характеристик в DataFrame
    df_mean_features = selected_features.groupby('ID').mean().reset_index() #!

    # Объединение с исходным DataFrame по столбцу 'ID'
    df = pd.merge(df, df_mean_features, on='ID', how='left')

    # Удаление временных столбцов

    # Вывод обновленного DataFrame
    print(df.head())





