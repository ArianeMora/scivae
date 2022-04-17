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

import os
import unittest
import pandas as pd

from scivae import VAE, VAEStats


class TestVAEStats(unittest.TestCase):

    def setUp(self):
        # Flag to set data to be local so we don't have to download them repeatedly. ToDo: Remove when publishing.
        self.local = True
        THIS_DIR = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(THIS_DIR, 'data/')
        #
        # if self.local:
        #     self.tmp_dir = os.path.join(THIS_DIR, 'data/tmp/')
        #     if os.path.exists(self.tmp_dir):
        #         shutil.rmtree(self.tmp_dir)
        #     os.mkdir(self.tmp_dir)
        # else:
        #     self.tmp_dir = tempfile.mkdtemp(prefix='EXAMPLE_PROJECT_tmp_')

    def tearDown(self):
        #shutil.rmtree(self.tmp_dir)
        print("Done")

    def test_train(self):
        data_dir = '../data/'
        config = {'loss': {'loss_type': 'mse', 'distance_metric': 'mmd', 'mmd_weight': 0.5},
                  'encoding': {'layers': [{'num_nodes': 4, 'activation_fn': 'selu'},
                                          {'num_nodes': 3, 'activation_fn': 'relu'}]},
                  'decoding': {'layers': [{'num_nodes': 2, 'activation_fn': 'relu'},
                                          {'num_nodes': 2, 'activation_fn': 'selu'}]},
                  'latent': {'num_nodes': 2},
                  'optimiser':  {'name': 'adam', 'params': {}}}
        weight_file_path = f'{data_dir}TvN_model_weights_VAE_MDS.h5'
        optimizer_file_path = f'{data_dir}TvN_model_optimiser_VAE_MDS.json'
        config_json = f'{data_dir}TvN_model_config_VAE_MDS.json'
        df = pd.read_csv(f'{data_dir}/MDS_data.csv', index_col=0)
        sample_df = pd.read_csv(f'{data_dir}/MDS_sample.csv')
        feature_columns = ['RNA-LogFC',
                           'Protein-LogFC',
                           'CpG-LogFC',
                           'RNA-Normal',
                           'RNA-Tumor',
                           'Protein-Normal',
                           'Protein-Tumor']
        train_df = pd.DataFrame(columns=feature_columns)
        matched_cases = []
        for case in sample_df['case_id'].unique():
            case_sample_df = sample_df[sample_df['case_id'] == case]
            if len(case_sample_df) > 8:
                # Need to think about this
                column_dict = dict(zip(case_sample_df.column_label, case_sample_df.column_id))
                # Now iterate through each of the columns and add those to the DF
                case_cond_df = pd.DataFrame()
                case_cond_df['id'] = list(df.index.values)
                for col in feature_columns:
                    case_cond_df[col] = df[column_dict[col]].values # Get the column name from the case
                # Add this to the cond_1_sample_df
                train_df = train_df.append(case_cond_df)
                matched_cases.append(case)
            else:
                print(case)
        case_sample_df = sample_df[sample_df['case_id'].isin(matched_cases)]
        case_sample_df.to_csv(f'{data_dir}/MDS_sample_matched.csv')
        train_df[['id'] + feature_columns].to_csv(f'{data_dir}/MDS_train_data.csv', index=False)
        vae_m = VAE(train_df[feature_columns].values, train_df[feature_columns].values,
                    list(train_df.id.values), config)

        vae_m.encode('default', epochs=200, batch_size=10)
        vae_m.save(weight_file_path=weight_file_path,
                   optimizer_file_path=optimizer_file_path,
                   config_json=config_json)  # save the VAE
        vae_m.u.dp(["Saved VAE to current directory."])

    def test_stats(self):
        data_dir = '../data/'
        df = pd.read_csv(f'{data_dir}/MDS_data.csv', index_col=0)
        sample_df = pd.read_csv(f'{data_dir}/MDS_sample_matched.csv')
        weight_file_path = f'{data_dir}TvN_model_weights_VAE_MDS.h5'
        optimizer_file_path = f'{data_dir}TvN_model_optimiser_VAE_MDS.json'
        config_json = f'{data_dir}TvN_model_config_VAE_MDS.json'
        feature_columns = ['RNA-LogFC',
                           'Protein-LogFC',
                           'CpG-LogFC',
                           'RNA-Normal',
                           'RNA-Tumor',
                           'Protein-Normal',
                           'Protein-Tumor']
        vs = VAEStats(df, sample_df, weight_file_path=weight_file_path,
                      optimizer_file_path=optimizer_file_path, config_json=config_json,
                      feature_columns=feature_columns,
                      )
        stats_vals = vs.peform_DVAE()
        stats_vals.to_csv(f'{data_dir}/MDS_stats.csv')
        print(stats_vals)
        print(len(stats_vals[stats_vals['padj'] < 0.05]))

    def test_train_multiloss(self):
        loss = {'loss_type': 'multi', 'distance_metric': 'mmd', 'mmd_weight': 0.1, 'multi_loss': ['mse', 'mse']}
        config = {"loss": loss,
                  "encoding": {"layers": [
                      [{"num_nodes": 2, "activation_fn": "relu"}, {"num_nodes": 2, "activation_fn": "relu"}]]},
                  "decoding": {"layers": [
                      [{"num_nodes": 2, "activation_fn": "relu"}, {"num_nodes": 2, "activation_fn": "relu"}]]},
                  "latent": {"num_nodes": 1},
                  "optimiser": {"params": {'learning_rate': 0.01}, "name": "adam"},
                  "input_size": [3, 4],
                  "output_size": [3, 4],
                  "epochs": 200,
                  "batch_size": 16
                  }
        data_dir = '../data/'
        weight_file_path = f'{data_dir}TvN_model_weights_VAE_MDS_multiloss.h5'
        optimizer_file_path = f'{data_dir}TvN_model_optimiser_VAE_MDS_multiloss.json'
        config_json = f'{data_dir}TvN_model_config_VAE_MDS_multiloss.json'
        feature_columns = ['RNA-LogFC',
                           'Protein-LogFC',
                           'CpG-LogFC',
                           'RNA-Normal',
                           'RNA-Tumor',
                           'Protein-Normal',
                           'Protein-Tumor']
        feature_column_position = {'RNA-LogFC': 0,
                                   'Protein-LogFC': 0,
                                   'CpG-LogFC' : 0,
                                   'RNA-Normal': 1,
                                   'RNA-Tumor': 1,
                                   'Protein-Normal': 1,
                                   'Protein-Tumor': 1}
        train_df = pd.read_csv(f'{data_dir}/MDS_train_data.csv')
        case_sample_df = pd.read_csv(f'{data_dir}/MDS_sample.csv')
        case_sample_df = case_sample_df[case_sample_df['column_label'].isin(feature_columns)]
        # Need to update with mutliloss position
        multi_loss = []
        for c in case_sample_df['column_label'].values:
            multi_loss.append(feature_column_position[c])
        case_sample_df['multi_loss'] = multi_loss
        case_sample_df.to_csv(f'{data_dir}/MDS_sample_matched_multiloss.csv')
        train_df[['id'] + feature_columns].to_csv(f'{data_dir}/MDS_train_data.csv', index=False)
        multi_loss = []
        # Need to put the columns in the correct multiloss order
        case_sample_df.sort_values(by=['multi_loss'], inplace=True)
        for c in case_sample_df['multi_loss'].unique():
            ml_col = case_sample_df[case_sample_df['multi_loss'] == c]
            cols = [c for c in feature_columns if c in list(ml_col.column_label.values)]
            multi_loss.append(train_df[cols].values)
        # Encode using multiloss
        print("training")
        vae_m = VAE(multi_loss, multi_loss, list(train_df.id.values), config)
        vae_m.encode('default', epochs=200, batch_size=10)
        vae_m.save(weight_file_path=weight_file_path,
                   optimizer_file_path=optimizer_file_path,
                   config_json=config_json)  # save the VAE
        vae_m.u.dp(["Saved VAE to current directory."])

    def test_stats_multiloss(self):
        data_dir = '../data/'
        df = pd.read_csv(f'{data_dir}/MDS_data.csv', index_col=0)
        weight_file_path = f'{data_dir}TvN_model_weights_VAE_MDS_multiloss.h5'
        optimizer_file_path = f'{data_dir}TvN_model_optimiser_VAE_MDS_multiloss.json'
        config_json = f'{data_dir}TvN_model_config_VAE_MDS_multiloss.json'
        feature_columns = ['RNA-LogFC',
                           'Protein-LogFC',
                           'CpG-LogFC',
                           'RNA-Normal',
                           'RNA-Tumor',
                           'Protein-Normal',
                           'Protein-Tumor']
        sample_df = pd.read_csv(f'{data_dir}/MDS_sample_matched_multiloss.csv')
        matched_cases = []
        for case in sample_df['case_id'].unique():
            case_sample_df = sample_df[sample_df['case_id'] == case]
            if len(case_sample_df) == 7:
                matched_cases.append(case)
            else:
                print(case)
        sample_df = sample_df[sample_df['case_id'].isin(matched_cases)]
        vs = VAEStats(df, sample_df, weight_file_path=weight_file_path,
                      optimizer_file_path=optimizer_file_path, config_json=config_json,
                      feature_columns=feature_columns,
                      )
        stats_vals = vs.peform_DVAE_multiloss()
        stats_vals.to_csv(f'{data_dir}/MDS_stats_multiloss.csv')
        print(stats_vals)
        print(len(stats_vals[stats_vals['padj'] < 0.05]))
