###############################################################################
#                                                                             #
#    This program is free software: you can redistribute it and/or modify     #
#    it under the terms of the GNU General Public License as published by     #
#    the Free Software Foundation, either version 3 of the License, or        #
#    (at your option) any later version.                                      #
#                                                                             #
#    This program is distributed in the hope that it will be useful,          #
#    but WITHOUT ANY WARRANTY; without even the implied warranty of           #
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the            #
#    GNU General Public License for more details.                             #
#                                                                             #
#    You should have received a copy of the GNU General Public License        #
#    along with this program. If not, see <http://www.gnu.org/licenses/>.     #
#                                                                             #
###############################################################################

"""
The aim here is to produce an 'integrated' statistic using the VAE output.

The aim is to do integrated stats on a single latent node, however, one could technically do it across nodes
building a linear model or performing an anova.

VAE stats takes a pretrained VAE from scivae and creates statistics on the latent space.

Statistics can either be performed on a single array (i.e. just getting the p-value for a point) or on all by
getting the p-value for each point in the latent space. In the latter case we do a correction for multiple testing.
"""
from scivae import VAE
from scipy import stats
from statsmodels.stats.multitest import multipletests
import pandas as pd
import json


class VAEStats:

    def __init__(self, df, sample_df, weight_file_path: str, optimizer_file_path: str, config_json: str,
                 feature_columns: list,
                 vae_name: str = "None"):
        """
        Initialize the VAEStats object.

        :param dataset: The dataset object to use.
        :param sample_df: The sample dataframe to use (contains information about the two conditions). i.e. case_id, condition, sample_id
        :param weight_file_path: The path to the weight file.
        :param optimizer_file_path: The path to the optimizer file.
        :param config_json: The path to the config json file.
        """
        self.df = df
        self.feature_columns = feature_columns
        self.encoded_df = pd.DataFrame()
        self.sample_df = sample_df
        with open(config_json, "r") as fp:
            self.config = json.load(fp)
        self.vae = VAE(df.values, df.values, ["None"] * len(df), self.config, vae_name)
        # Load pre-saved VAE
        self.vae.load(weight_file_path, optimizer_file_path, config_json)

    def test_for_normality(self, values, test_type: str = "shapiro"):
        """ Perform a test for normality."""
        k2, p = stats.normaltest(values)
        if p < 0.05:  # null hypothesis: x comes from a normal distribution
            print(f'NOT normally distributed')
            return False
        return True

    def peform_DVAE(self, test_type: str = "t-test"):
        # For each of the conditions we want to encode each of the points then perform a stats test between the two
        # conditions.
        # Get all the rows associated with this condition
        # There are three levels of information 1) condition, 2) feature, 3) case_id
        # Each case ID presents a unique training data point
        cond_1_sample_df = self.sample_df[self.sample_df['condition_id'] == 1]
        id_vals = self.df.index.values
        cond_1_encodings = {}
        for case in cond_1_sample_df['case_id'].unique():
            case_sample_df = cond_1_sample_df[cond_1_sample_df['case_id'] == case]
            # Need to think about this
            column_dict = dict(zip(case_sample_df.column_label, case_sample_df.column_id))
            # Now iterate through each of the columns and add those to the DF
            case_cond_df = pd.DataFrame(columns=self.feature_columns)
            for col in self.feature_columns:
                case_cond_df[col] = self.df[column_dict[col]].values # Get the column name from the case
            # Add this to the cond_1_sample_df
            cond_1_encodings[case] = self.vae.encode(case_cond_df.values)

        # Encode this value
        cond_0_sample_df = self.sample_df[self.sample_df['condition_id'] == 0]
        cond_0_encodings = {}
        for case in cond_0_sample_df['case_id'].unique():
            case_sample_df = cond_0_sample_df[cond_0_sample_df['case_id'] == case]
            # Need to think about this
            column_dict = dict(zip(case_sample_df.column_label, case_sample_df.column_id))
            case_cond_df = pd.DataFrame(columns=self.feature_columns)
            for col in self.feature_columns:
                case_cond_df[col] = self.df[column_dict[col]].values # Get the column name from the case
            cond_0_encodings[case] = self.vae.encode(case_cond_df.values)

        # Now we want to perform the differential test on the data between cond 1 - cond 0
        # If we have multiple samples we need to do this for each one
        if len(cond_0_encodings[0]) > 0:
            stat_vals = []
            p_vals = []
            # For each case in the encodings we want to collect the values
            for i in range(0, len(cond_0_encodings[0])):
                cases_0_vals = [c[i] for c in cond_0_encodings.values()]
                cases_1_vals = [c[i] for c in cond_1_encodings.values()]
                # potentially wrap a try catch if there are all even numbers
                t_stat, p_val = stats.mannwhitneyu(cases_1_vals, cases_0_vals)
                stat_vals.append(t_stat)
                p_vals.append(p_val)
            # Now we have the p-values we can perform the correction
            corrected_p_vals = multipletests(p_vals, method='fdr_bh')
            return pd.DataFrame(columns=['id', 'stat', 'padj'], data=[id_vals, stat_vals, corrected_p_vals])
        else:
            # Only one value so just do the test once.
            cases_0_vals = [c for c in cond_0_encodings.values()]
            cases_1_vals = [c for c in cond_1_encodings.values()]
            t_stat, p_val = stats.mannwhitneyu(cases_1_vals, cases_0_vals)
            return t_stat, p_val