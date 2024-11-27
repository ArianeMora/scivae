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

from tensorflow.keras.layers import Input
import numpy as np
from scivae import VAE


class SupVAE(VAE):

    """
    References:
        https://github.com/pren1/keras-MMD-Variational-Autoencoder/blob/master/Keras_MMD_Variational_Autoencoder.ipynb
        https://github.com/s-omranpour/X-VAE-keras/blob/master/VAE/VAE_MMD.ipynb
        https://github.com/ShengjiaZhao/MMD-Variational-Autoencoder/blob/master/mmd_vae.py
        https://github.com/CancerAI-CL/IntegrativeVAEs/blob/master/code/models/mmvae.py
        https://arxiv.org/abs/1706.02262
        https://ermongroup.github.io/blog/a-tutorial-on-mmd-variational-autoencoders/
        https://github.com/tensorflow/tensorflow/issues/41053 for saving
        https://bjlkeng.github.io/posts/semi-supervised-learning-with-variational-autoencoders/ for semi supervised
    """
    def __init__(self, input_data_np: np.array, output_data_np: np.array, labels: list, config, vae_label=None,
                 sciutil=None, config_as_str=False, empty=False):

        super().__init__(input_data_np, output_data_np, labels, config, vae_label, sciutil, config_as_str, empty)
        self.__last_encoding_shape = None

    def build_encoder(self):
        # Check if multi - if so we need to concatenate our layers see:
        # https://github.com/CancerAI-CL/IntegrativeVAEs/blob/master/code/models/xvae.py
        layer_start_idx = 0
        layer = self.encoding_config['layers'][0]
        # ToDo: refactor to handle arbitary number of inputs
        # For now we just need to ensure the first two layers are encoded separately
        # The first one is built using a conv layer
        encoding_0 = self.build_layer(layer['num_nodes'], self.inputs_x[0], layer['activation_fn'])
        # We need to add a dense layer to combine these, we can't simply concatenate them because 1 is
        # a
        self.encoding = encoding_0
        layer_start_idx = 1  # Since we have already done the first

        # Now for subsequent layers we run this
        for layer_idx in range(layer_start_idx, len(self.encoding_config['layers'])):
            layer = self.encoding_config['layers'][layer_idx]
            # For now we just need to ensure the first two layers are encoded separately
            self.encoding = self.build_layer(layer['num_nodes'], self.encoding, layer['activation_fn'])

        return self.encoding

    def build_decoder(self):
        # Generate input to the decoder which is based on random sampling (we then compare this to the reconstructed
        # values using the input as our output
        self.latent_inputs = Input(shape=(self.latent_config['num_nodes'],), name='z_sampling')
        self.decoding = [self.latent_inputs, self.latent_inputs]
        # Here we also need to check for the multi output if so we don't do the last layer normally
        layer_end_idx = len(self.decoding_config['layers']) - 1

        for layer_idx in range(0, layer_end_idx):
            layer = self.decoding_config['layers'][layer_idx]
            self.decoding[0] = self.build_layer(layer[0]['num_nodes'], self.decoding[0], layer[0]['activation_fn'])
            self.decoding[1] = self.build_layer(layer[1]['num_nodes'], self.decoding[1], layer[1]['activation_fn'])

        # Add final layer to the decoder that matches the input size & use a sigmoid activation
        # If we have a multi layer output, i.e. a loss function that spans the two, we want x separate layers
        self.decoding = self.build_multi_output(self.decoding)

        return self.decoding

    def build_multi_output(self, decoding_layer):
        """ ToDo: Modularise this to have an arbitary number of layers. """
        # Add in the final layer
        decoder_0 = self.build_layer(self.decoding_config['layers'][-1][0]['num_nodes'], decoding_layer[0],
                                     self.output_activation_fn)
        decoder_0 = self.build_layer(self.output_size[0], decoder_0, self.output_activation_fn)

        # Add in final layer for decoder 2
        decoder_1 = self.build_layer(self.decoding_config['layers'][-1][1]['num_nodes'], decoding_layer[1],
                                     self.output_activation_fn)
        decoder_1 = self.build_layer(self.output_size[1], decoder_1, self.output_activation_fn)
        return [decoder_0, decoder_1]