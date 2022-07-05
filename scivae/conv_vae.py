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

from tensorflow.keras.layers import Lambda, Input, Dense, BatchNormalization, Conv2D, Conv2DTranspose
import tensorflow as tf
import numpy as np
from scivae import VAE


class ConvVAE(VAE):

    """
    References:
            https://www.tensorflow.org/tutorials/generative/cvae

    """

    def __init__(self, input_data_np: np.array, output_data_np: np.array, labels: list, config, vae_label=None,
                 sciutil=None, config_as_str=False, empty=False):

        super().__init__(input_data_np, output_data_np, labels, config, vae_label, sciutil, config_as_str, empty)
        self.__last_encoding_shape = None

    def default_inputs(self):
        self.inputs_x = Input(shape=(self.input_size, self.input_size, 1), name='default_input')
        return self.inputs_x

    def build_encoder(self):
        # Check if multi - if so we need to concatenate our layers see:
        # https://github.com/CancerAI-CL/IntegrativeVAEs/blob/master/code/models/xvae.py
        layer_start_idx = 0
        self.encoding = self.inputs_x
        # Now for subsequent layers we run this
        for layer_idx in range(layer_start_idx, len(self.encoding_config['layers'])):
            layer = self.encoding_config['layers'][layer_idx]
            self.encoding = self.build_encoding_layer(layer, self.encoding, layer['activation_fn'])
        self.__last_encoding_shape = self.encoding.shape
        return self.encoding

    def build_decoder(self):
        # Generate input to the decoder which is based on random sampling (we then compare this to the reconstructed
        # values using the input as our output
        self.latent_inputs = Input(shape=(self.latent_config['num_nodes'],), name='z_sampling')
        self.decoding = self.latent_inputs
        # Here we also need to check for the multi output if so we don't do the last layer normally
        layer_end_idx = len(self.decoding_config['layers'])
        # Add in the first one which requires a reshape from the dense latent space
        s = self.__last_encoding_shape  # ToDo: make more general Only doing this for 2D conv
        self.decoding = Dense(units=s[1] * s[2] * s[3], activation=tf.nn.relu)(self.decoding)
        self.decoding = tf.keras.layers.Reshape(target_shape=(s[1], s[2], s[3]))(self.decoding)
        for layer_idx in range(1, layer_end_idx - 1):
            layer = self.decoding_config['layers'][layer_idx]
            self.decoding = self.build_decoding_layer(layer, self.decoding, layer['activation_fn'])

        # Build the last layer
        self.decoding = self.build_decoding_layer(self.decoding_config['layers'][-1], self.decoding,
                                                  self.output_activation_fn)
        return self.decoding

    def build_encoding_layer(self, layer, prev_layer, activation_fn='selu'):
        filters = layer.get('filters')
        kernel_size = layer.get('kernel_size')
        pooling = layer.get('pooling')
        strides = layer.get('strides')
        padding = layer.get('padding')
        x = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding, strides=strides,
                   activation=activation_fn)(prev_layer)
        #x = tf.keras.layers.MaxPool2D(pooling)(x)
        # Perform batch normalisation
        if self.batch_norm:
            x = BatchNormalization()(x)
        return x

    def build_decoding_layer(self, layer, prev_layer, activation_fn='selu'):
        filters = layer.get('filters')
        kernel_size = layer.get('kernel_size')
        pooling = layer.get('pooling')
        strides = layer.get('strides')
        padding = layer.get('padding')
        x = Conv2DTranspose(filters=filters, kernel_size=kernel_size, padding=padding, strides=strides,
                   activation=activation_fn)(prev_layer)
        #x = tf.keras.layers.MaxPool2D(pooling)(x)
        # Perform batch normalisation
        if self.batch_norm:
            x = BatchNormalization()(x)
        return x

    def build_embedding(self):
        # Flatten before encoding!
        self.encoding = tf.keras.layers.Flatten()(self.encoding)
        self.latent_z_mean = Dense(self.latent_config['num_nodes'], name='z_mean')(self.encoding)
        self.latent_z_log_sigma = Dense(self.latent_config['num_nodes'], name='z_log_sigma',
                                        kernel_initializer='zeros')(self.encoding)
        self.latent_z = Lambda(self.sample, output_shape=(self.latent_config['num_nodes'],), name='z')([
            self.latent_z_mean, self.latent_z_log_sigma])

