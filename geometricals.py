import pandas as pd
import numpy as np
from typing import Union

#channel = df['channel']
#insert into main code:
"""
o(n^2) - bad bad bad. maybe loc into ID in df. if means o(1)
for df in setup:
    for channel in df:
        if channel != 'time' :
            f'semi_max_{channel[-1]}' = mean_median(channel)
"""


#done
def semi_max(channel) -> Union[int, float]:
    return channel.max() / 2


def subpeak(channel):
    ...


# time = df['time'] where df with ID (means raindrop)
def semi_width(time_values: pd.Series, channel_values: pd.Series) -> float:
    semi_max_value = semi_max(channel_values)
    above_semi_max = channel_values >= semi_max_value
    #under_semi_max = channel_values <= semi_max_value

    if above_semi_max.any():
        indices = above_semi_max.index[above_semi_max].tolist()
        print(indices)
        if len(indices) >= 2:
            left_index = indices[0]
            right_index = indices[-1]

            if left_index > 0 and right_index < len(time_values) - 1:
                print("Полупик существует с обеих сторон. Берем разность времени между правой и левой частью.")
                return time_values.iloc[right_index] - time_values.iloc[left_index]
            else:
                print("Полупик только с одной стороны. Берем расстояние от пика до полупика и умножаем на 2.")
                return (time_values.iloc[right_index] - time_values.iloc[left_index]) * 2
        else:
            print("С одной из сторон нет полупика. Возвращаем длину до пика * 2.")
            return (time_values.max() - time_values.min()) * 2
    else:
        print("Не удалось найти полупик.")
        return 0.0





def length_subpeak_peak(channel):
    ...


#done
def mean_median(channel) -> Union[int, float]:
    return channel.mean() - channel.median()


#done
def q10(channel) -> Union[int, float]:

    q10 = np.nanquantile(channel, 0.10)
    return q10


#done
def q25(channel) -> Union[int, float]:

    q25 = np.nanquantile(channel, 0.25)
    return q25


#done
def q75(channel) -> Union[int, float]:

    q75 = np.nanquantile(channel, 0.75)
    return q75



#done
def q90(channel) -> Union[int, float]:

    q90 = np.nanquantile(channel, 0.90)
    return q90


#done
def k_semi_max(channel) -> Union[int, float]:
    return channel.max() - semi_max(channel)



if __name__ == "__main__":
    ...
