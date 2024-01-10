import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt
import pandas as pd



#почекать катоф фрекьюнси
def apply_lowpass_filter(data, cutoff_freq=1.5, order=10, sample_rate=5):
    nyquist_freq = 0.5 * sample_rate
    normal_cutoff = cutoff_freq / nyquist_freq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data


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

    return train


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
        time_list_A.append(dataframe.loc[dataframe['Channel A'] == row['Channel A'], 'Time'].values[0])
        time_list_B.append(dataframe.loc[dataframe['Channel B'] == row['Channel B'], 'Time'].values[0])
        time_list_C.append(dataframe.loc[dataframe['Channel C'] == row['Channel C'], 'Time'].values[0])
        time_list_D.append(dataframe.loc[dataframe['Channel D'] == row['Channel D'], 'Time'].values[0])

    dt1 = pd.DataFrame([a - b for a, b in zip(time_list_A, time_list_B)], columns=["dtAB"])
    dt2 = pd.DataFrame([c - d for c, d in zip(time_list_C, time_list_D)], columns=["dtCD"])
    dt3 = pd.DataFrame([a - c for a, c in zip(time_list_A, time_list_C)], columns=["dtAC"])
    dt4 = pd.DataFrame([b - d for b, d in zip(time_list_B, time_list_D)], columns=["dtBD"])

    dt1['ID'] = idshniki
    dt2['ID'] = idshniki
    dt3['ID'] = idshniki
    dt4['ID'] = idshniki

    # для удобного отображения дробных чисел
    pd.set_option('display.float_format', lambda x: '%.10f' % x)

    dataframe1 = pd.merge(dt1, dt2, on="ID")
    dataframe1 = pd.merge(dataframe1, dt3, on="ID")
    dataframe1 = pd.merge(dataframe1, dt4, on="ID")

    new_order = ['ID', 'dtAB', 'dtCD', 'dtAC', 'dtBD']
    dataframe1 = dataframe1[new_order]
    pd.set_option('display.max_columns', None)
    pd.set_option('display.expand_frame_repr', False)

    return dataframe1


def raindrops_and_peaks(for_train, height_peak, window_size, butter=False, summa=0):

    peaks, i = find_peaks(for_train['Channel A'], height=height_peak, distance=window_size)

    print(peaks)

    print(str(len(peaks)) + " length of peaks\n")

    # храниние капель
    setup = np.empty((len(for_train),), dtype=object)
    print("Создано хранилище капель")

    # Объявляем глобальную переменную для учета индекса

    count = 0
    # поиск пиков и вырез окна
    for i, peak in enumerate(peaks):
        count += 1
        start = max(peak - window_size // 2, 0)
        end = min(peak + window_size // 2, len(for_train))
        # Вырезаем окно для каждого канала
        window = for_train.loc[start:end, ['Time', 'Channel A', 'Channel B', 'Channel C', 'Channel D']].copy()
        #print(window)

        # Добавляем окно в массив капелек
        if butter is True:


                for column in window.columns:

                    if column != "Time":
                        # train[column] = train[column].rolling(window=window_size).mean()
                        height_ = lambda a, b: a if a > b else b
                        window[column] = apply_lowpass_filter(window[column], height_(abs(window[column].max()), 0.000001), 1, 2)

        window['ID'] = i + summa

        setup[i] = window

    return setup







