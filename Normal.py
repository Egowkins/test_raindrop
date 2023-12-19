import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def optimization(train, window_size=1000):

    """

    :param train: исходный датасет
    :param window_size: окно для "сглаживания"
    :return: возвращает оптимизированный датасет
    """
    for column in train.columns:
        train[column] = train[column].astype(str)
        train[column] = train[column].str.replace(',', '.')
        train[column] = pd.to_numeric(train[column], errors='coerce')
        print(f"Оптимизация столбца {column}") #check

    for column in train.columns:
        if column != "Time":
            train[column] = train[column].rolling(window=window_size).mean()
            print(f"Сглаживание столбца {column}")
    return train



#TODO: написать декоратор
def np_to_df(np_arr, df=None):

    df_1 = pd.concat(np_arr, ignore_index=True)
    df = pd.concat([df, df_1], ignore_index=True)

    return df


def heigh_search(dataframe):
  """
  Функция предназначена для нахождения уровня амплитуды сигнала А (ниже которой не берем – слишком маленькие капли)
  """

  positive_values = dataframe['Channel A'][dataframe['Channel A'] > 0]
  height = np.nanquantile(positive_values, 0.15)

  return height


def plotter_maker(df, peaks_=None) -> None:
    """
    :param df: датасет для построения графика
    :param peaks_: пики, если необходимо построить график для датасета в целом
    :return: None
    """

    if peaks_ is not None:
        plt.plot(df['Time'], df['Channel A'], label='Сигнал')
        plt.plot(df['Time'].iloc[peaks_], df['Channel A'].iloc[peaks_], 'x', label='Исходные пики',
                 color='red')
        plt.legend()
        plt.show()

        pass

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8))

    # Строим графики для каждой колонки в DataFrame
    axes[0, 0].plot(df['Time'], df['Channel A'], label='ColumnA')
    axes[0, 1].plot(df['Time'], df['Channel B'], label='ColumnB', color='orange')
    axes[1, 0].plot(df['Time'], df['Channel C'], label='ColumnC', color='green')
    axes[1, 1].plot(df['Time'], df['Channel D'], label='ColumnD', color='red')

    # Настроим подписи осей и легенду
    axes[0, 0].set_title('ColumnA')
    axes[0, 1].set_title('ColumnB')
    axes[1, 0].set_title('ColumnC')
    axes[1, 1].set_title('ColumnD')

    # Добавим общую легенду в правом верхнем углу
    axes[0, 0].legend(loc='upper right')
    axes[0, 1].legend(loc='upper right')
    axes[1, 0].legend(loc='upper right')
    axes[1, 1].legend(loc='upper right')

    # Добавим общую подпись осей y
    fig.text(0.02, 0.5, 'Значение', va='center', rotation='vertical', fontsize=12)

    # Отобразим графики
    plt.tight_layout()
    plt.show()


def dt_finder(dataframe):
    #max_values = dataframe.groupby('ID').max()
    #print(max_values.head(20))

    idshniki = dataframe['ID'].unique()
    max_values_df = pd.DataFrame(columns=dataframe.columns.difference(['Time']))

    for unique_id in idshniki:
        subset = dataframe[dataframe['ID'] == unique_id]
        max_values_row = subset.drop('Time', axis=1).max()
        max_values_df = pd.concat([max_values_df, max_values_row.to_frame().T], ignore_index=True)

    print(max_values_df)

    time_list_A = []
    time_list_B = []
    time_list_C = []
    time_list_D = []

    for index, row in max_values_df.iterrows():
        time_list_A.append(dataframe.loc[dataframe['ID'] == row['ID'], 'Time'].values[0])
        time_list_B.append(dataframe.loc[dataframe['ID'] == row['ID'], 'Time'].values[1])
        time_list_C.append(dataframe.loc[dataframe['ID'] == row['ID'], 'Time'].values[2])
        time_list_D.append(dataframe.loc[dataframe['ID'] == row['ID'], 'Time'].values[3])
    """
    dataframe['value1'] = dataframe['id'].map(lambda x: time_list_A[id_to_index[x]])
    dataframe['value2'] = dataframe['id'].map(lambda x: list2[id_to_index[x]])
    dataframe['value3'] = dataframe['id'].map(lambda x: list3[id_to_index[x]])
    dataframe['value4'] = dataframe['id'].map(lambda x: list4[id_to_index[x]])
    """
    dt1 = pd.DataFrame([a - b for a, b in zip(time_list_A, time_list_B)], columns=["dt1"])
    dt2 = pd.DataFrame([c - d for c, d in zip(time_list_C, time_list_D)], columns = ["dt2"])
    dt3 = pd.DataFrame([a - c for a, c in zip(time_list_A, time_list_C)], columns = ["dt3"])
    dt4 = pd.DataFrame([b - d for b, d in zip(time_list_B, time_list_D)], columns = ["dt4"])


    # для полного отображения дробных чисел
    pd.set_option('display.float_format', lambda x: '%.20f' % x)



    #print(dt1)
    #print([a - b for a, b in zip(time_list_A, time_list_B)])

    dataframe = pd.merge(dataframe, dt1, left_on="ID", right_index = True)
    dataframe = pd.merge(dataframe, dt2, left_on="ID", right_index=True)
    dataframe = pd.merge(dataframe, dt3, left_on="ID", right_index=True)
    dataframe = pd.merge(dataframe, dt4, left_on="ID", right_index=True)

    #dataframe = pd.concat([dataframe, dt1], ignore_index=True)
    #dataframe = pd.concat([dataframe, dt2], ignore_index=True)
    #dataframe = pd.concat([dataframe, dt3], ignore_index=True)
    #dataframe = pd.concat([dataframe, dt4], ignore_index=True)

    pd.set_option('display.max_columns', None)
    pd.set_option('display.expand_frame_repr', False)
    #

    print(dataframe)

    return dataframe










