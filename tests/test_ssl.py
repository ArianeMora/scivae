
import os
import shutil
import tempfile
import unittest
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from scivae import SupVAE, VAE, Validate
from sciviso import Scatterplot


class TestSupVAE(unittest.TestCase):

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


    def test_VAE(self):
        """
        Test using CE loss on the labels to get better separation in latent space
        """
        loss = {'loss_type': 'mse', 'distance_metric': 'mmd', 'mmd_weight': 0.2, 'loss_weightings': [0.5, 10.0]}
        encoding = {'layers': [{'num_nodes': 2, 'activation_fn': 'selu'},
                                {'num_nodes': 3, 'activation_fn': 'selu'}]} # , {'num_nodes': 3, 'activation_fn': 'relu'}]}
        decoding = {'layers': [{'num_nodes': 3, 'activation_fn': 'selu'},
                                {'num_nodes': 2, 'activation_fn': 'selu'}]}
        latent = {'num_nodes': 2}
        optimisers = {'name': 'adam', 'params': {}}

        data = f'{self.data_dir}iris.csv'
        # Build a simple vae to learn the relations in the iris dataset
        df = pd.read_csv(data)
        # Shuffle
        df = df.sample(frac=1)

        value_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        input_values = np.array(df[value_cols].values)

        config = {'loss': loss, 'encoding': encoding, 'decoding': decoding, 'latent': latent,
                  'optimiser': optimisers}

        labels = df['label'].values
        input_values = (input_values - np.min(input_values)) / (np.max(input_values) - np.min(input_values))

        vae = VAE(input_values, input_values, df['label'].values, config, 'vae')

        vae.encode('default', logging_dir=self.tmp_dir)

        # Lets have a look at a scatterplot version & apply the class colours to our plot
        encoding = vae.get_encoded_data()
        encoding = vae.encode_new_data(input_values, scale=False)

        decoding = vae.decoder.predict(encoding)
        print(decoding)
        vis_df = pd.DataFrame()
        vis_df['latent_0'] = encoding[:, 0]
        vis_df['latent_1'] = encoding[:, 1]
        labels = df['label'].values
        lut = dict(zip(set(labels), sns.color_palette("coolwarm", len(set(labels)))))
        row_colors2 = pd.DataFrame(labels)[0].map(lut)
        vis_df['label'] = row_colors2
        scatter = Scatterplot(vis_df, 'latent_0', 'latent_1', colour=row_colors2, title='multiloss', xlabel='latent')
        scatter.plot()
        plt.show()
        vd = Validate(vae.get_encoded_data(), labels)
        print(vd.predict('rf', 'accuracy'))
        print(vd.predict('svm', 'balanced_accuracy'))

    def test_sup_VAE(self):
        """
        Test using CE loss on the labels to get better separation in latent space
        """
        loss = {'loss_type': 'multi', 'distance_metric': 'mmd', 'mmd_weight': 0.2,
                'multi_loss': ['mse', 'ce'], 'loss_weightings': [0.5, 10.0], 'contrastive_params': {}}
        encoding = {'layers': [{'num_nodes': 2, 'activation_fn': 'selu'},
                                {'num_nodes': 3, 'activation_fn': 'selu'}]} # , {'num_nodes': 3, 'activation_fn': 'relu'}]}
        decoding = {'layers': [[{'num_nodes': 3, 'activation_fn': 'selu'},
                                {'num_nodes': 2, 'activation_fn': 'selu'}]]}
        latent = {'num_nodes': 2}
        optimisers = {'name': 'adam', 'params': {}}

        data = f'{self.data_dir}iris.csv'
        # Build a simple vae to learn the relations in the iris dataset
        df = pd.read_csv(data)
        # Shuffle
        df = df.sample(frac=1)

        value_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
        input_values = df[value_cols].values
        labels = []
        na_labels = []
        for c in df['label'].values:
            if c == 'Iris-setosa':
                labels.append(np.asarray([1, 0, 0]))
                na_labels.append(np.asarray([0, 0, 0]))
            elif c == 'Iris-virginica':
                labels.append(np.asarray([0, 1, 0]))
                na_labels.append(np.asarray([0, 0, 0]))
            else:
                labels.append(np.asarray([0, 0, 1]))
                na_labels.append(np.asarray([0, 0, 0]))

        print(set(df['label'].values))
        input_values = (input_values - np.min(input_values)) / (np.max(input_values) - np.min(input_values))
        config = {'loss': loss, 'encoding': encoding, 'decoding': decoding, 'latent': latent,
                  'optimiser': optimisers,
                  'input_size': [4, 3], 'output_size': [4, 3]}
        vae = SupVAE([input_values, np.asarray(labels)], [input_values, np.asarray(labels)],
                  df['label'].values, config, 'vae')

        vae.encode('default', logging_dir=self.tmp_dir)

        # Lets have a look at a scatterplot version & apply the class colours to our plot
        encoding = vae.get_encoded_data()
        encoding = vae.encode_new_data([input_values, np.asarray(labels)], scale=False)

        decoding = vae.decoder.predict(encoding)
        print(decoding)
        vis_df = pd.DataFrame()
        vis_df['latent_0'] = encoding[:, 0]
        vis_df['latent_1'] = encoding[:, 1]
        labels = df['label'].values
        lut = dict(zip(set(labels), sns.color_palette("coolwarm", len(set(labels)))))
        row_colors2 = pd.DataFrame(labels)[0].map(lut)
        vis_df['label'] = row_colors2
        scatter = Scatterplot(vis_df, 'latent_0', 'latent_1', colour=row_colors2, title='multiloss', xlabel='latent')
        scatter.plot()
        plt.show()
        vd = Validate(vae.get_encoded_data(), labels)
        print(vd.predict('rf', 'accuracy'))
        print(vd.predict('svm', 'balanced_accuracy'))