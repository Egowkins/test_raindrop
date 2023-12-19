import pandas as pd
from tsfresh import extract_features
from tsfresh.feature_extraction import ComprehensiveFCParameters

def extract_features_df(df):
    # Создаем ComprehensiveFCParameters для извлечения различных характеристик
    settings = ComprehensiveFCParameters()

# Извлекаем характеристики
    extracted_features = extract_features(df, column_id='ID', column_sort='Time', column_value='Channel A', default_fc_parameters=settings)

# Рассчитываем усредненные характеристики
    averaged_features = extracted_features.groupby('ID').mean()

# Добавляем усредненные характеристики в исходный DataFrame
    df = pd.merge(df, averaged_features, left_on='ID', right_index=True, how='left')
    print(df)



# Теперь df содержит усредненные характеристики от сигнала A
