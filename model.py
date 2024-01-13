from tsfresh.feature_extraction import feature_calculators
import pandas as pd


def feature_extractor(df, feature_df, col_name):
    unique_ids = df['ID'].unique()
    result_df = pd.DataFrame()

    for ID in unique_ids:
        channel_values = df.loc[df['ID'] == ID, col_name]

        features = {
            'ID': ID,
            f'Minimum_{col_name[-1]}': channel_values.min(),
            f'Maximum_{col_name[-1]}': channel_values.max(),
            f'Variance_{col_name[-1]}': feature_calculators.variance(channel_values),
            f'Skewness_{col_name[-1]}': feature_calculators.skewness(channel_values),
            f'Kurtosis_{col_name[-1]}': feature_calculators.kurtosis(channel_values),
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
"""



