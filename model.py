from tsfresh.feature_extraction import feature_calculators
import pandas as pd
import geometricals


def feature_extractor(df, feature_df, col_name):
    unique_ids = df['ID'].unique()
    result_df = pd.DataFrame()
    # добавить time / iloc po id
    for ID in unique_ids:
        channel_values = df.loc[df['ID'] == ID, col_name]
        time_values = df.loc[df['ID'] == ID, 'Time']
        #print(geometricals.mean_mode(channel_values))

        features = {
            'ID': ID,
            f'Minimum_{col_name[-1]}': channel_values.min(),
            f'Maximum_{col_name[-1]}': channel_values.max(),
            f'Variance_{col_name[-1]}': feature_calculators.variance(channel_values),
            f'Semimax_{col_name[-1]}': geometricals.semi_max(channel_values),
            f'K_{col_name[-1]}': geometricals.k_semi_max(channel_values),
            f'Q10_{col_name[-1]}': geometricals.q10(channel_values)
            #f'Semiwidth_{col_name[-1]}': geometricals.semi_width(time_values, channel_values)
        }
       # print(features)
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

