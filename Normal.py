import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def optimization(train):

    for column in train.columns:
        train[column] = train[column].astype(str)
        train[column] = train[column].str.replace(',', '.')
        train[column] = pd.to_numeric(train[column], errors='coerce')
        print("!") #check

    return train


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

"""
def peak_marker(df, peaks):
   df['x'] = None
   for peak in peaks:
       df['x'] = 
"""
