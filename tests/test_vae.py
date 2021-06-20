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
import shutil
import tempfile
import unittest
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import json
from sklearn.preprocessing import MinMaxScaler

from scivae import Optimiser, VAE, Validate, Vis
from sciviso import Scatterplot


class TestVAE(unittest.TestCase):

    def setUp(self):
        # Flag to set data to be local so we don't have to download them repeatedly. ToDo: Remove when publishing.
        self.local = True
        THIS_DIR = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = os.path.join(THIS_DIR, 'data/')

        if self.local:
            self.tmp_dir = os.path.join(THIS_DIR, 'data/tmp/')
            if os.path.exists(self.tmp_dir):
                shutil.rmtree(self.tmp_dir)
            os.mkdir(self.tmp_dir)
        else:
            self.tmp_dir = tempfile.mkdtemp(prefix='EXAMPLE_PROJECT_tmp_')

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_histone_data(self):
        df = pd.read_csv('data/mouse_HM_var500_data.csv')
        df = df.fillna(0)
        # Get out columns with HM values
        cols = [c for c in df.columns if '10' in c and 'brain' in c and 'signal' in c]  # i.e. only do brain at E10 samples
        # Make sure we log2 the values since they're too diffuse
        vae_df = pd.DataFrame()
        vae_df['external_gene_name'] = df['external_gene_name'].values
        new_cols = []
        for c in cols:
            new_name = ' '.join(c.split('_')[:-3]).replace('embryonic', '')
            new_cols.append(new_name)
            vae_df[new_name] = np.log2(df[c] + 1)

        dataset = vae_df[new_cols].values
        # Create and train VAE
        vae = VAE(dataset, dataset, ["None"] * len(dataset), 'data/histone_config.json', f'vae_rcm', config_as_str=True)
        vae.encode('default', epochs=10, batch_size=50, logging_dir=self.tmp_dir)
        vis = Vis(vae, vae.u, None)
        vis.plot_node_hists(show_plt=True, save_fig=False)
        vis.plot_node_feature_correlation(vae_df, 'external_gene_name', columns=new_cols, show_plt=True, save_fig=False)

        vis.plot_feature_scatters(vae_df, 'external_gene_name', columns=new_cols, show_plt=True, fig_type="png", save_fig=True,
                                  title="latent space", output_dir='test_figures/')

        cool_genes = [['Emx1', 'Eomes', 'Tbr1', 'Foxg1', 'Lhx6', 'Arx', 'Dlx1', 'Dlx2', 'Dlx5', 'Nr2e2', 'Otx2'],
                      ['Hoxd8', 'Hoxd9', 'Hoxd10', 'Hoxd11', 'Hoxd12', 'Hoxd13', 'Hoxa7', 'Hoxa9', 'Hoxa10', 'Hoxa11',
                      'Hoxa13',
                      'Hoxb9', 'Hoxb13', 'Hoxc8', 'Hoxc9', 'Hoxc10', 'Hoxc11', 'Hoxc12', 'Hoxc13'],
                      ['Ccna1', 'Ccna2', 'Ccnd1', 'Ccnd2', 'Ccnd3', 'Ccne1', 'Ccne2', 'Cdc25a',
                       'Cdc25b', 'Cdc25c', 'E2f1', 'E2f2', 'E2f3', 'Mcm10', 'Mcm5', 'Mcm3', 'Mcm2', 'Cip2a']
                      ]
        vis.plot_values_on_scatters(vae_df, "external_gene_name", ['Forebrain', 'Spinal cord', 'Pro. Prolif.'],
                                    cool_genes, show_plt=True, fig_type=".png",
                                    save_fig=False)


    def test_saving(self):
        vae_df = pd.read_csv(os.path.join(self.data_dir, 'vis.csv'))
        cols = ['sepal_length', 'sepal_width', 'petal_length']
        data_cols = [c for c in vae_df.columns if c != 'gene_name']
        dataset = vae_df[data_cols].values

        with open(os.path.join(self.data_dir, f'vis.json'), "r") as fp:
            config = json.load(fp)
            vae = VAE(dataset, dataset, ["None"] * len(dataset), os.path.join(self.data_dir, f'vis.json'), f'vae_rcm',
                      config_as_str=True)
            vae.encode('default', epochs=config['epochs'], batch_size=config['batch_size'], logging_dir=self.tmp_dir)
            vae.save()

            # Rebuild the model
            new_vae = VAE(None, None, None, config, f'vae_rcm', empty=True)
            new_vae.load()

            scaler = MinMaxScaler(copy=True)
            scaled_vals = scaler.fit_transform(dataset)
            data = new_vae.encode_new_data(scaled_vals, encoding_type="z")
            print(data)
            vis = Vis(new_vae.vae, new_vae.u, None)
            #  df: pd.DataFrame, row_id: str, vae_data=None, columns=None, output_dir="", fig_type='svg',
            #                               title="latent space", show_plt=False, save_fig=True, angle_plot=0
            vis_df = vae_df.copy()
            vis_df["vae_0"] = data[:, 0]
            vis_df["vae_1"] = data[:, 1]
            vis_df["vae_2"] = data[:, 2]
            vis.plot_top_values_by_rank(vis_df, ["vae_0", "vae_1", "vae_2"], cols, "gene_name", num_values=3,
                                        cluster_rows=False, show_plt=True, save_fig=False, title="Saved VAE")

            # Compare this to the old vae
            data = vae.encode_new_data(scaled_vals, encoding_type="z")
            #  df: pd.DataFrame, row_id: str, vae_data=None, columns=None, output_dir="", fig_type='svg',
            #                               title="latent space", show_plt=False, save_fig=True, angle_plot=0
            vis_df = vae_df.copy()
            vis_df["vae_0"] = data[:, 0]
            vis_df["vae_1"] = data[:, 1]
            vis_df["vae_2"] = data[:, 2]
            vis.plot_top_values_by_rank(vis_df, ["vae_0", "vae_1", "vae_2"], cols, "gene_name", num_values=3,
                                        cluster_rows=False, show_plt=True, save_fig=False, title="OG VAE")

            vae.set_seed()
            data = vae.encode_new_data(scaled_vals, encoding_type="z")
            print(data)
            #  df: pd.DataFrame, row_id: str, vae_data=None, columns=None, output_dir="", fig_type='svg',
            #                               title="latent space", show_plt=False, save_fig=True, angle_plot=0
            vis_df = vae_df.copy()
            vis_df["vae_0"] = data[:, 0]
            vis_df["vae_1"] = data[:, 1]
            vis_df["vae_2"] = data[:, 2]
            vis.plot_top_values_by_rank(vis_df, ["vae_0", "vae_1", "vae_2"], cols, "gene_name", num_values=3,
                                        cluster_rows=False, show_plt=True, save_fig=False, title="OG VAE")

    def test_vis(self):
        vae_df = pd.read_csv(os.path.join(self.data_dir, 'vis.csv'))
        cols = ['sepal_length', 'sepal_width', 'petal_length']
        with open(os.path.join(self.data_dir, f'vis.json'), "r") as fp:
            config = json.load(fp)
            data_cols = [c for c in vae_df.columns if c != 'gene_name']
            dataset = vae_df[data_cols].values

            # Create and train VAE
            vae = VAE(dataset, dataset, ["None"] * len(dataset), config, f'vae_rcm')
            vae.encode('default', epochs=config['epochs'], batch_size=config['batch_size'], logging_dir=self.tmp_dir)
            # Encode new data
            scaler = MinMaxScaler(copy=True)
            scaled_vals = scaler.fit_transform(dataset)
            data = vae.encode_new_data(scaled_vals, encoding_type="z")


            # Now we have 3 vae nodes so let's add them to the DF
            vis = Vis(vae, vae.u, None)
            ax = vis.plot_values_on_scatters(vae_df, "gene_name", ['Genes'],
                                       [['TUBB8P11', 'VWA1', 'ISG15', 'PTPN9']], show_plt=True, fig_type=".png",
                                             save_fig=False)

            vis_df = vae_df.copy()
            vis_df["vae_0"] = data[:, 0]
            vis_df["vae_1"] = data[:, 1]
            vis_df["vae_2"] = data[:, 2]
            vis.plot_top_values_by_rank(vis_df, ["vae_0", "vae_1", "vae_2"], cols, "gene_name", num_values=10,
                                        cluster_rows=False)
            vis.plot_feature_scatters(vae_df, 'gene_name', columns=cols, show_plt=True, fig_type=".png", save_fig=True,
                                      title="cX DepthshadeTrue latent space")
            vis.plot_node_hists(show_plt=True, save_fig=False)
            vis.plot_node_hists(show_plt=True, save_fig=False, method="z_mean")
            vis.plot_node_hists(show_plt=True, save_fig=False, method="z_log_var")

            vis.plot_node_feature_correlation(vae_df, 'gene_name', columns=cols, show_plt=True, save_fig=False)
            vis.plot_node_correlation(show_plt=True, save_fig=False)
            vis.plot_feature_correlation(vae_df, 'gene_name', columns=cols, show_plt=True, save_fig=False)
            vis.plot_input_distribution(vae_df, show_plt=True, save_fig=False)

    def test_vae(self):
        """
        Tested
        """
        for t in ['mmd']:
            loss = {'loss_type': 'mse', 'distance_metric': t, 'mmd_weight': 1.0, 'beta': 1.0, 'mmcd_method': 'k'}
            encoding = {'layers': []}#[{'num_nodes': 3, 'activation_fn': 'selu'}]} #, {'num_nodes': 3, 'activation_fn': 'relu'}]}
            decoding = {'layers': []} #[{'num_nodes': 3, 'activation_fn': 'selu'}]}
            latent = {'num_nodes': 2}
            optimisers = {'name': 'adam', 'params': {}}
            config = {'loss': loss, 'encoding': encoding, 'decoding': decoding, 'latent': latent, 'optimiser': optimisers}
            data = f'{self.data_dir}iris.csv'
            # Build a simple vae to learn the relations in the iris dataset
            df = pd.read_csv(data)
            value_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
            vae = VAE(df[value_cols].values, df[value_cols].values, df['label'].values, config, 'vae')
            vae.encode('default', batch_size=40, logging_dir=self.tmp_dir)

            # Lets have a look at a scatterplot version & apply the class colours to our plot
            encoding = vae.get_encoded_data()

            decoding = vae.decoder.predict(encoding)
            print(decoding)
            vis_df = pd.DataFrame()
            vis_df['latent_0'] = encoding[:, 0]
            vis_df['latent_1'] = encoding[:, 1]
            labels = df['label'].values
            lut = dict(zip(set(labels), sns.color_palette("coolwarm", len(set(labels)))))
            row_colors2 = pd.DataFrame(labels)[0].map(lut)
            vis_df['label'] = row_colors2

            vd = Validate(vae, labels)
            rf_acc = int(100 * vd.predict('rf', 'accuracy'))
            print(rf_acc)
            svm_acc = int(100 * vd.predict('svm', 'balanced_accuracy'))
            print(svm_acc)

            scatter = Scatterplot(vis_df, 'latent_0', 'latent_1', colour=row_colors2,
                                  title=f'Loss {loss.get("distance_metric")} acc svm:{svm_acc}% acc rf:{rf_acc}%',
                                  xlabel='')
            scatter.plot()
            plt.show()

    def test_multiloss(self):
        """
        Tests for using multiple loss functions on the output
        """
        loss = {'loss_type': 'multi', 'distance_metric': 'mmd', 'mmd_weight': 1, 'multi_loss': ['mse', 'mse']}
        encoding = {'layers': [[{'num_nodes': 2, 'activation_fn': 'selu'}, {'num_nodes': 1, 'activation_fn': 'selu'}]]}  #, {'num_nodes': 3, 'activation_fn': 'relu'}]}
        decoding = {'layers': [[{'num_nodes': 3, 'activation_fn': 'selu'}, {'num_nodes': 2, 'activation_fn': 'selu'}]]}
        latent = {'num_nodes': 2}
        optimisers = {'name': 'adam', 'params': {}}

        data = f'{self.data_dir}iris.csv'
        # Build a simple vae to learn the relations in the iris dataset
        df = pd.read_csv(data)
        value_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        input_values = df[value_cols].values
        config = {'loss': loss, 'encoding': encoding, 'decoding': decoding, 'latent': latent, 'optimiser': optimisers,
                  'input_size': [2, 2], 'output_size': [2, 2]}
        vae = VAE([input_values[:, :2], input_values[:, 2:]], [input_values[:, 2:], input_values[:, 2:]],
                  df['label'].values, config, 'vae')

        vae.encode('default', logging_dir=self.tmp_dir)

        # Lets have a look at a scatterplot version & apply the class colours to our plot
        encoding = vae.get_encoded_data()
        decoding = vae.decoder.predict(encoding)
        print(decoding)
        vis_df = pd.DataFrame()
        vis_df['latent_0'] = encoding[:, 0]
        vis_df['latent_1'] = encoding[:, 1]
        labels = df['label'].values
        lut = dict(zip(set(labels), sns.color_palette("coolwarm", len(set(labels)))))
        row_colors2 = pd.DataFrame(labels)[0].map(lut)
        vis_df['label'] = row_colors2
        scatter = Scatterplot(vis_df, 'latent_0', 'latent_1', colour=row_colors2, title='asd', xlabel='asd')
        scatter.plot()
        plt.show()
        vd = Validate(vae, labels)
        print(vd.predict('rf', 'accuracy'))
        print(vd.predict('svm', 'balanced_accuracy'))

    def test_optimiser(self):
        """
        Tested
        """
        bounds = {'latent_num_nodes': [1, 2, 3, 4],
                  'encoding': [[1, 2, 3, 4], [1, 2, 3, 4]],
                  'decoding': [[1, 2, 3, 4], [1, 2, 3, 4]],
                  'optimisers': ['adamax', 'adam']
                  }

        data = f'{self.data_dir}iris.csv'
        value_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        # Build a simple vae to learn the relations in the iris dataset
        df = pd.read_csv(data)
        optimiser = Optimiser(df[value_cols].values, df['label'].values, bounds)
        generations = 2  # Number of times to evole the population.
        population = 2  # Number of networks in each generation.
        vaes = optimiser.create_population(population)
        for i in range(generations):
            # Train
            optimiser.train_vaes(vaes)
            # Get accuracy
            avg_acc = optimiser.get_average_goodness(vaes)
            # evolve except last gen
            if i != generations - 1:
                # Do the evolution.
                vaes = optimiser.evolve(vaes)

        vaes = sorted(vaes, key=lambda x: x.get_goodness_score(), reverse=True)

        # Print out the top 5 networks.
        for vae in vaes:
            vae.print()

    def test_optimal_rf(self):
        """
        Tested
        """
        config = {'loss': {'loss_type': 'mse', 'distance_metric': 'mmd', 'mmd_weight': 0.5},
                  'encoding': {'layers': [{'num_nodes': 4, 'activation_fn': 'selu'},
                                          {'num_nodes': 3, 'activation_fn': 'relu'}]},
                  'decoding': {'layers': [{'num_nodes': 2, 'activation_fn': 'relu'},
                                          {'num_nodes': 2, 'activation_fn': 'selu'}]},
                  'latent': {'num_nodes': 2},
                  'optimiser':  {'name': 'adam', 'params': {}}}
        data = f'{self.data_dir}iris.csv'
        # Build a simple vae to learn the relations in the iris dataset
        df = pd.read_csv(data)
        value_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        vae = VAE(df[value_cols].values, df[value_cols].values, df['label'].values, config, 'vae')
        vae.encode('default', logging_dir=self.tmp_dir)

        # Lets have a look at a scatterplot version & apply the class colours to our plot
        encoding = vae.get_encoded_data()
        vis_df = pd.DataFrame()
        vis_df['latent_0'] = encoding[:, 0]
        vis_df['latent_1'] = encoding[:, 1]
        labels = df['label'].values
        lut = dict(zip(set(labels), sns.color_palette("coolwarm", len(set(labels)))))
        row_colors2 = pd.DataFrame(labels)[0].map(lut)
        vis_df['label'] = row_colors2
        scatter = Scatterplot(vis_df, 'latent_0', 'latent_1', colour=row_colors2, title='asd', xlabel='asd')
        scatter.plot()
        plt.show()
        vd = Validate(vae, labels)
        print(vd.predict('rf', 'balanced_accuracy'))
        print(vd.predict('svm', 'balanced_accuracy'))

    def test_optimal_svm(self):
        """
        Tested
        """
        config = {'loss': {'loss_type': 'mse', 'distance_metric': 'mmd', 'mmd_weight': 1},
         'encoding': {'layers': [{'num_nodes': 1, 'activation_fn': 'selu'}, {'num_nodes': 2, 'activation_fn': 'selu'}]},
         'decoding': {'layers': [{'num_nodes': 3, 'activation_fn': 'selu'}, {'num_nodes': 1, 'activation_fn': 'selu'}]},
         'latent': {'num_nodes': 2}, 'optimiser':  {'name': 'adam', 'params': {}}}

        data = f'{self.data_dir}iris.csv'
        # Build a simple vae to learn the relations in the iris dataset
        df = pd.read_csv(data)
        value_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        vae = VAE(df[value_cols].values, df[value_cols].values, df['label'].values, config, 'vae')
        vae.encode('default', logging_dir=self.tmp_dir)

        # Lets have a look at a scatterplot version & apply the class colours to our plot
        encoding = vae.get_encoded_data()
        vis_df = pd.DataFrame()
        vis_df['latent_0'] = encoding[:, 0]
        vis_df['latent_1'] = encoding[:, 1]
        labels = df['label'].values
        lut = dict(zip(set(labels), sns.color_palette("coolwarm", len(set(labels)))))
        row_colors2 = pd.DataFrame(labels)[0].map(lut)
        vis_df['label'] = row_colors2
        scatter = Scatterplot(vis_df, 'latent_0', 'latent_1', colour=row_colors2, title='asd', xlabel='asd')
        scatter.plot()
        plt.show()
        vd = Validate(vae, labels)
        print(vd.predict('rf', 'balanced_accuracy'))
        print(vd.predict('svm', 'balanced_accuracy'))

    def test_auto_optimise(self):
        """
        Tested
        """
        bounds = {'latent_num_nodes': [1, 2, 3, 4],
                  'encoding': [[1, 2, 3, 4], [1, 2, 3, 4]],
                  'decoding': [[1, 2, 3, 4], [1, 2, 3, 4]],
                  }

        data = f'{self.data_dir}iris.csv'
        value_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        # Build a simple vae to learn the relations in the iris dataset
        df = pd.read_csv(data)
        optimiser = Optimiser(df[value_cols].values, df['label'].values, bounds)
        generations = 2  # Number of times to evole the population.
        population = 2  # Number of networks in each generation.
        print(optimiser.optimise_architecture(num_generations=generations, population_size=population))

    def test_on_mnist(self):
        """
        Tested
        """
        data_dir = f'{self.data_dir}/mnist/'
        image_size = 28
        num_images = 6 # More images the better just takes alot longer
        training_f = open(f'{data_dir}train-images-idx3-ubyte', 'rb')
        training_f.read(16)
        buf = training_f.read(image_size * image_size * num_images)
        train_data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)

        train_data = train_data.reshape(num_images, image_size * image_size)

        f = open(f'{data_dir}train-labels-idx1-ubyte', 'rb')
        f.read(8)
        train_labels = []
        for i in range(0, len(train_data)):
            buf = f.read(1)
            labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
            train_labels.append(labels[0])
        test_f = open(f'{data_dir}t10k-images-idx3-ubyte', 'rb')
        test_f.read(16)

        buf = test_f.read(image_size * image_size * num_images)
        test_data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        #
        test_data = test_data.reshape(num_images, image_size * image_size)
        # image = np.asarray(test_data[2]).squeeze()

        f = open(f'{data_dir}t10k-labels-idx1-ubyte', 'rb')
        f.read(8)
        test_labels = []
        for i in range(0, len(test_data)):
            buf = f.read(1)
            labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
            test_labels.append(labels[0])

        bounds = {'latent_num_nodes': [1, 2, 4, 16],
                  'encoding': [[64, 128, 512, 1064], [64, 128, 512, 1064]],
                  'decoding': [[64, 128, 512, 1064], [64, 128, 512, 1064]],
                  }

        # Build a simple vae to learn the relations in the iris dataset
        # This will be terrible since we're only doing 2 epochs
        optimiser = Optimiser(train_data, train_labels, bounds,  validation_method="prediction",
                              epochs=1, batch_size=500)
        generations = 2  # Number of times to evole the population.
        population = 3   # Number of networks in each generation.
        encoder_configs_and_score = optimiser.optimise_architecture(num_generations=generations,
                                                                    population_size=population)
        # {'config': vaes[0], 'fitness': fitness}
        vae = encoder_configs_and_score[0]['config']
        # we also want to display the VAE coloured by the labels
        vae.print()

        plt.figure(figsize=(20, 2))
        n = num_images
        for i in range(n):
            ax = plt.subplot(1, n, i + 1)
            d = vae.vae.predict(train_data)
            plt.imshow(d[i].reshape(28, 28))

        plt.show()
        plt.figure(figsize=(20, 2))
        for i in range(n):
            ax = plt.subplot(1, n, i + 1)
            plt.imshow(train_data[i].reshape(28, 28))

        plt.show()
        return

    def test_vis_mnist(self):
        """
        Tested

        """
        config = {'loss': {'loss_type': 'mse', 'distance_metric': 'mmd', 'mmd_weight': 1},
                  'encoding': {'layers': [{'num_nodes': 203, 'activation_fn': 'selu'}]},
                  'decoding': {'layers': [{'num_nodes': 455, 'activation_fn': 'selu'}]},
                  'latent': {'num_nodes': 727}, 'optimiser': {'params': {}, 'name': 'adam'}}
        data_dir = f'{self.data_dir}/mnist/'
        image_size = 28
        # The more training the longer to run but the better your recon
        num_images = 6 # 50000
        test_f = open(f'{data_dir}train-images-idx3-ubyte', 'rb')
        test_f.read(16)

        buf = test_f.read(image_size * image_size * num_images)
        test_data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        test_data = test_data.reshape(num_images, image_size * image_size)
        # image = np.asarray(test_data[2]).squeeze()

        f = open(f'{data_dir}train-labels-idx1-ubyte', 'rb')
        f.read(8)
        test_labels = []
        for i in range(0, len(test_data)):
            buf = f.read(1)
            labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
            test_labels.append(labels[0])

        vae = VAE(test_data, test_data, test_labels, config, 'vae')
        vae.encode('default', epochs=2, batch_size=1000, logging_dir=self.tmp_dir)
        encoding = vae.get_encoded_data()
        plt.figure(figsize=(20, 2))
        n = num_images
        d = vae.vae.predict(test_data)
        for i in range(n):
            ax = plt.subplot(1, n, i + 1)
            plt.imshow(d[i,:].reshape(28, 28))

        plt.show()
        plt.figure(figsize=(20, 2))
        for i in range(n):
            ax = plt.subplot(1, n, i + 1)
            plt.imshow(test_data[i].reshape(28, 28))

        plt.show()

        # Let's have a look at how good the encoding was for prediction
        print(encoding)
        vis_df = pd.DataFrame()
        vis_df['latent_0'] = encoding[:, 0]
        vis_df['latent_1'] = encoding[:, 1]
        labels = test_labels
        lut = dict(zip(set(labels), sns.color_palette("coolwarm", len(set(labels)))))
        row_colors2 = pd.DataFrame(labels)[0].map(lut)
        vis_df['label'] = row_colors2
        scatter = Scatterplot(vis_df, 'latent_0', 'latent_1', colour=row_colors2, title='asd', xlabel='asd')
        scatter.plot()
        plt.show()
        # Have to make labels a 1 and 0 --> note this is just a tpy example to exn
        labels = np.array(labels)
        labels = 1 * (labels > 0)
        vd = Validate(vae, labels)
        print(vd.predict('svm', 'balanced_accuracy'))

        print(vd.predict('svm', 'accuracy'))



