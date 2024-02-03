from tsfresh.feature_extraction import feature_calculators
import pandas as pd
import numpy as np
import geometricals
import matplotlib.pyplot as plt


def feature_extractor(df, feature_df, col_name):
    unique_ids = df['ID'].unique()
    result_df = pd.DataFrame()
    # добавить time / loc po id
    for ID in unique_ids:
        channel_values = df.loc[df['ID'] == ID, col_name].reset_index(drop=True)
        time_values = df.loc[df['ID'] == ID, 'Time'].reset_index(drop=True)

        features = {
            'ID': ID,
            f'Minimum_{col_name[-1]}': channel_values.min(),
            f'Maximum_{col_name[-1]}': channel_values.max(),
            f'Variance_{col_name[-1]}': feature_calculators.variance(channel_values),
            f'Kurtosis_{col_name[-1]}': feature_calculators.kurtosis(channel_values),
            f'Skewness_{col_name[-1]}': feature_calculators.skewness(channel_values),
            f'Median_{col_name[-1]}': feature_calculators.median(channel_values),
            f'Mean_median_{col_name[-1]}': geometricals.mean_median(channel_values),
            f'Std_{col_name[-1]}': np.std(channel_values),
            f'Semimax_{col_name[-1]}': geometricals.semi_max(channel_values),
            f'K_{col_name[-1]}': geometricals.k_semi_max(channel_values, time_values),
            f'Q10_{col_name[-1]}': geometricals.q10(channel_values),
            f'Q25_{col_name[-1]}': geometricals.q25(channel_values),
            f'Q75_{col_name[-1]}': geometricals.q75(channel_values),
            f'Q90_{col_name[-1]}': geometricals.q90(channel_values),
            f'Subpeak_{col_name[-1]}': geometricals.subpeak(channel_values, time_values),
            f'Semiwidth_{col_name[-1]}': geometricals.semi_width(channel_values, time_values)[0]
            if isinstance(geometricals.semi_width(channel_values, time_values), tuple) else 0
        }



        result_df = result_df._append(features, ignore_index=True)

    feature_df = pd.merge(feature_df, result_df, on="ID", how="left")

    return feature_df


"""
            f'AbsEnergy_{col_name[-1]}': feature_calculators.abs_energy(channel_values),
            f'AbsoluteSumOfChanges_{col_name[-1]}': feature_calculators.absolute_sum_of_changes(channel_values),
            f'BenfordCorrelation_{col_name[-1]}': feature_calculators.benford_correlation(channel_values),
            f'CidCe_{col_name[-1]}': feature_calculators.cid_ce(channel_values, normalize=True),
            f'FirstLocMax_{col_name[-1]}': feature_calculators.first_location_of_maximum(channel_values),
            f'MeanChange_{col_name[-1]}': feature_calculators.mean_change(channel_values),
            f'MeanSecDerivative_{col_name[-1]}': feature_calculators.mean_second_derivative_central(channel_values),
            f'Median_{col_name[-1]}': feature_calculators.median(channel_values),
            f'SampleEntropy_{col_name[-1]}': feature_calculators.sample_entropy(channel_values),
            f'VarLargerThanStd_{col_name[-1]}': feature_calculators.variance_larger_than_standard_deviation(channel_values)
            
            
            
            f'Subpeak_{col_name[-1]}': geometricals.subpeak(channel_values),
            f'Mean_Mode_{col_name[-1]}': geometricals.mean_mode(channel_values),
            f'Semiwidth_{col_name[-1]}': geometricals.semi_width(channel_values),
            f'Length_subpeak_{col_name[-1]}': geometricals.length_subpeak_peak(channel_values),
            
"""

if __name__ == '__main__':
  ...

