import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt
import pandas as pd


def apply_lowpass_filter(data, cutoff_freq=1.5, order=10, sample_rate=5):
    nyquist_freq = 0.5 * sample_rate
    normal_cutoff = cutoff_freq / nyquist_freq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data


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


def plotter_maker(setup, setup2, col_name) -> None:
    """
    :param df: датасет для построения графика
    :param peaks_: пики, если необходимо построить график для датасета в целом
    :return: None
    """
    fig, axes = plt.subplots(4, 4, figsize=(20, 16))

    # Наложите каждый график на соответствующую область
    for i, ax in enumerate(axes.flatten()):
        # Вам нужно указать свои столбцы и параметры для графика в функции plot()

        ax.plot(setup2[i]['Time'], setup2[i][col_name], label='Not_Optimized', color='b')
        plt.grid(True)
        ax.plot(setup[i]['Time'], setup[i][col_name], label='Optimized', color='red')
        plt.grid(True)
        ax.legend()


    # Регулируем расположение графиков
    plt.tight_layout()
    plt.savefig(f'Figure{col_name[-1]}.png')



#todo на проверку смазываемости


def dt_finder(dataframe: pd.DataFrame) -> pd.DataFrame:

    idshniki = dataframe['ID'].unique()
    max_values_df = pd.DataFrame(columns=dataframe.columns.difference(['Time']))

    for unique_id in idshniki:
        subset = dataframe[dataframe['ID'] == unique_id]
        max_values_row = subset.drop('Time', axis=1).max()
        max_values_df = pd.concat([max_values_df, max_values_row.to_frame().T], ignore_index=True)

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


def raindrops_and_peaks(for_train, height_peak, window_size, butter=False):

    peaks, i = find_peaks(for_train['Channel A'], height=height_peak, distance=window_size)

    print(str(len(peaks)) + " length of peaks\n")

    # храниние капель
    setup = np.empty((len(for_train),), dtype=object)
    print("Создано хранилище капель")


    # поиск пиков и вырез окна
    for i, peak in enumerate(peaks):

        start = max(peak - window_size // 2, 0)
        end = min(peak + window_size // 2, len(for_train))

        # Вырезаем окно для каждого канала
        window = for_train.loc[start:end, ['Time', 'Channel A', 'Channel B', 'Channel C', 'Channel D']].copy()

        # Добавляем окно в массив капелек
        if butter is True:

                for column in window.columns:

                    if column != "Time":

                        height_ = lambda a, b: a if a > b else b
                        window[column] = apply_lowpass_filter(window[column], height_(abs(window[column].max()), 0.00000000000000001), 1, 2)

        window['ID'] = i

        setup[i] = window

    return setup


def find_subpeaks():
    ...




