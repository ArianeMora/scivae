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

from scivae import Optimiser, ConvVAE, Validate, Vis
from sciviso import Scatterplot


class TestCVAE(unittest.TestCase):

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

    def test_conv_mnist(self):
        """
        Tested

        """
        config = {'scale_data': False,
                  'input_size': 28,
                 'loss': {'loss_type': 'mse', 'distance_metric': 'mmd', 'mmd_weight': 1},
                  'encoding': {'layers': [{'filters': 64, 'kernel_size': 3, 'strides': 2, 'padding':'same',
                                          'activation_fn': 'selu'},
                                          {'filters': 32, 'kernel_size': 3, 'strides': 2, 'padding': 'same',
                                           'activation_fn': 'selu'}
                                          ]},
                  'decoding': {'layers': [{'filters': 64, 'kernel_size': 3, 'strides': 2, 'padding':'same',
                                          'activation_fn': 'selu'},
                                          {'filters': 32, 'kernel_size': 3, 'strides': 2, 'padding': 'same',
                                           'activation_fn': 'selu'},
                                          {'filters': 1, 'kernel_size': 3, 'strides': 2, 'padding': 'same',
                                           'activation_fn': None}
                                          ]},

                  'latent': {'num_nodes': 2}, 'optimiser': {'params': {}, 'name': 'adam'}}
        data_dir = f'{self.data_dir}/mnist/'
        image_size = 28
        # The more training the longer to run but the better your recon
        num_images = 10  # 50000
        test_f = open(f'{data_dir}train-images-idx3-ubyte', 'rb')
        test_f.read(16)

        buf = test_f.read(image_size * image_size * num_images)
        test_data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)

        test_data = test_data.reshape((-1, 28, 28, 1)).astype('float32') #reshape(num_images, image_size * image_size)
        test_data /= 255

        f = open(f'{data_dir}train-labels-idx1-ubyte', 'rb')
        f.read(8)
        test_labels = []
        for i in range(0, len(test_data)):
            buf = f.read(1)
            labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
            test_labels.append(labels[0])

        vae = ConvVAE(test_data, test_data, test_labels, config, 'vae')
        vae.encode('default', epochs=100, batch_size=100, logging_dir=self.tmp_dir)
        encoding = vae.encode_new_data(test_data, scale=False)
        plt.figure(figsize=(20, 2))
        n = num_images
        for i in range(n):
            d = vae.decoder.predict(np.array([encoding[i]]))
            ax = plt.subplot(1, n, i + 1)
            plt.imshow(d.reshape(28, 28))
        plt.show()
        plt.figure(figsize=(20, 2))

        for i in range(n):
            ax = plt.subplot(1, n, i + 1)
            plt.imshow(test_data[i])

        plt.show()
