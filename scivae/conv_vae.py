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

from tensorflow.keras.layers import Lambda, Input, Dense, BatchNormalization, Conv2D, Conv2DTranspose, Concatenate
import tensorflow as tf
import numpy as np
from scivae import VAE


from tensorflow.keras.layers import Dropout

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
        if self.multi_output:
            # The first one always has to be 2D and second 1D
            self.inputs_x = [Input(shape=(self.input_size[0][0], self.input_size[0][1], 1), name='default_input_0'),
                             Input(shape=(self.input_size[1],), name='default_input_1')]
        else:
            self.inputs_x = Input(shape=(self.input_size[0], self.input_size[1], 1), name='default_input')
        return self.inputs_x

    def build_encoder(self):
        # Check if multi - if so we need to concatenate our layers see:
        # https://github.com/CancerAI-CL/IntegrativeVAEs/blob/master/code/models/xvae.py
        layer_start_idx = 0
        if self.multi_output:
            # If we have a fist layer if
            layer = self.encoding_config['layers'][0]
            # ToDo: refactor to handle arbitary number of inputs
            # For now we just need to ensure the first two layers are encoded separately
            # The first one is built using a conv layer
            encoding_0 = self.build_encoding_layer(layer, self.inputs_x[0], layer['activation_fn'])
            # We need to add a dense layer to combine these, we can't simply concatenate them because 1 is
            # a
            self.encoding = [encoding_0]
            layer_start_idx = 1  # Since we have already done the first
        else:
            self.encoding = self.inputs_x
        # Now for subsequent layers we run this
        for layer_idx in range(layer_start_idx, len(self.encoding_config['layers'])):
            layer = self.encoding_config['layers'][layer_idx]
            # Again here we're only encoding the input data rather than the labels as well
            if self.multi_output:
                self.encoding[0] = self.build_encoding_layer(layer, self.encoding[0], layer['activation_fn'])
            else:
                self.encoding = self.build_encoding_layer(layer, self.encoding, layer['activation_fn'])

        if self.multi_output:
            self.__last_encoding_shape = [self.encoding[0].shape]  # We just pass the label through
        else:
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
        if self.multi_output:
            self.decoding = [Dense(units=s[0][1] * s[0][2] * s[0][3], activation=tf.nn.relu)(self.decoding),
                             Dense(units=s[0][1] * s[0][2] * s[0][3], activation=tf.nn.relu)(self.decoding)]
            # Here we reshape one of them but keep the other as just normal output rather than CNN
            self.decoding[0] = tf.keras.layers.Reshape(target_shape=(s[0][1], s[0][2], s[0][3]))(self.decoding[0])
        else:
            self.decoding = Dense(units=s[1] * s[2] * s[3], activation=tf.nn.relu)(self.decoding)
            self.decoding = tf.keras.layers.Reshape(target_shape=(s[1], s[2], s[3]))(self.decoding)
        for layer_idx in range(1, layer_end_idx - 1):
            layer = self.decoding_config['layers'][layer_idx]
            if self.multi_output:
                self.decoding[0] = self.build_decoding_layer(layer[0], self.decoding[0], layer[0]['activation_fn'])
                self.decoding[1] = self.build_layer(layer[1]['num_nodes'], self.decoding[1], layer[1]['activation_fn'])
            else:
                self.decoding = self.build_decoding_layer(layer, self.decoding, layer['activation_fn'])

        # Build the last layer
        if self.multi_output:
            self.decoding = self.build_multi_output([self.decoding[0], self.decoding[1]])
        else:
            self.decoding = self.build_decoding_layer(self.decoding_config['layers'][-1], self.decoding,
                                                      self.output_activation_fn)
        return self.decoding

    def build_multi_output(self, decoding_layer):
        """ ToDo: Modularise this to have an arbitary number of layers.
        Note: the way this currently works, the labels have to be the second option and the data is the first! """
        # Add in the final layer
        decoder_0 = self.build_decoding_layer(self.decoding_config['layers'][-1][0], decoding_layer[0],
                                              self.output_activation_fn)

        # Add in final layer for decoder 2
        decoder_1 = self.build_layer(self.decoding_config['layers'][-1][1]['num_nodes'], decoding_layer[1],
                                     self.output_activation_fn)
        decoder_1 = self.build_layer(self.output_size[1], decoder_1, self.output_activation_fn)
        return [decoder_0, decoder_1]

    def build_encoding_layer(self, layer, prev_layer, activation_fn='selu'):
        filters = layer.get('filters')
        kernel_size = layer.get('kernel_size')
        pooling = layer.get('pooling')
        strides = layer.get('strides')
        padding = layer.get('padding')
        x = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding, strides=strides,
                   activation=activation_fn)(prev_layer)
        dropout = layer.get('dropout')
        if dropout:
            x = Dropout(dropout)(x)
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
        dropout = layer.get('dropout')
        if dropout:
            x = Dropout(dropout)(x)
        # Perform batch normalisation
        if self.batch_norm:
            x = BatchNormalization()(x)
        return x

    def build_embedding(self):
        # Flatten before encoding!
        # Check if it's the multi because we now combine it
        if self.multi_output:
            # Flatten the layers and concat our labels and the
            self.encoding = tf.keras.layers.Flatten()(self.encoding[0])
        else:
            self.encoding = tf.keras.layers.Flatten()(self.encoding)
        self.latent_z_mean = Dense(self.latent_config['num_nodes'], name='z_mean')(self.encoding)
        self.latent_z_log_sigma = Dense(self.latent_config['num_nodes'], name='z_log_sigma',
                                        kernel_initializer='zeros')(self.encoding)
        self.latent_z = Lambda(self.sample, output_shape=(self.latent_config['num_nodes'],), name='z')([
            self.latent_z_mean, self.latent_z_log_sigma])
