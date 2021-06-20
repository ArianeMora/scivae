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

from keras.layers import Lambda, Input, Dense, BatchNormalization, Concatenate
from keras.models import Model
from keras import backend as K
from keras import optimizers
import tensorflow as tf
import numpy as np
from datetime import datetime
from keras.callbacks import CSVLogger
from numpy.random import seed
import pickle
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import TensorBoard
import json
import math
from sciutil import SciException, SciUtil
from scivae import Loss
import os


class VAEException(SciException):
    def __init__(self, message=''):
        Exception.__init__(self, message)


class VAE(object):

    """
    References:
        https://github.com/pren1/keras-MMD-Variational-Autoencoder/blob/master/Keras_MMD_Variational_Autoencoder.ipynb
        https://github.com/s-omranpour/X-VAE-keras/blob/master/VAE/VAE_MMD.ipynb
        https://github.com/ShengjiaZhao/MMD-Variational-Autoencoder/blob/master/mmd_vae.py
        https://github.com/CancerAI-CL/IntegrativeVAEs/blob/master/code/models/mmvae.py
        https://arxiv.org/abs/1706.02262
        https://ermongroup.github.io/blog/a-tutorial-on-mmd-variational-autoencoders/
        https://github.com/tensorflow/tensorflow/issues/41053 for saving
    """

    def __init__(self, input_data_np: np.array, output_data_np: np.array, labels: list, config, vae_label='',
                 sciutil=None, config_as_str=False, empty=False):
        self.u = sciutil if sciutil is not None else SciUtil()
        # Setting the other internal variables
        # Check if the config is a string object
        if config_as_str:
            with open(config, "r") as fp:
                config = json.load(fp)  # load from the FP
        self.encoder, self.decoder, self.encoded_data, self.decoded_data = None, None, None, None
        self.inputs_x, self.outputs_y = None, None
        self.latent_inputs = None
        self.vae_label = vae_label
        self.encoding, self.decoding, self.num_layers, self.vae = None, None, None, None
        self.latent_z, self.latent_z_mean, self.latent_z_log_sigma, self.latent_inputs = None, None, None, None
        self.training_input_np, self.test_input_np, self.test_labels, self.training_labels = None, None, None, None
        self.training_output_np, self.test_output_np = None, None
        self.goodness, self.validation_score, self.test_loss = None, None, None
        # Initialise using data and config
        self.input_data_np = input_data_np
        self.output_data_np = output_data_np
        self.labels = labels
        if not empty:
            print(config.get('input_size'))
            config['input_size'] = len(input_data_np[0]) if config.get('input_size') is None else config.get('input_size')
            config['output_size'] = len(output_data_np[0]) if config.get('output_size') is None else \
                config.get('output_size')
            self.config = config
            self.__init_from_config(config)

    def __init_from_config(self, config: dict):
        """ Enables initialisation from config dictionary, default behaviour & makes saving/reloading easier. """
        self.multi_output = True if config['loss'].get('multi_loss') is not None else False
        self.input_size = config['input_size']
        self.output_size = config['output_size']
        self.model_types = ['default', 'None']
        self.config = config
        self.model_type = None
        self.loss = Loss(config['loss']['loss_type'], config['loss']['distance_metric'],
                         config['loss']['mmd_weight'], config['loss'].get('multi_loss'),
                         beta=config['loss'].get('beta'), mmcd_method=config['loss'].get('mmcd_method'))
        self.encoding_config = config['encoding']
        self.decoding_config = config['decoding']
        self.latent_config = config['latent']
        self.output_activation_fn = 'sigmoid'
        if self.multi_output:
            if len(self.encoding_config['layers']) < 1:
                self.u.err_p(['ERR: VAE init: can not use a single layer VAE with mutliple inputs!'])
                return
        self.sample_method = config.get('sample_method') if config.get('sample_method') else "normal"  # Or negbinom
        self.seed = config.get('seed') if config.get('seed') else 17
        scale_data = config.get('scale_data') if config.get('scale_data') else True
        self.batch_norm = True if config.get('batch_norm') else False
        if scale_data and self.input_data_np is not None:
            self.scale_data()

    def scale_data(self):
        """ Scales data between 0 and 1 for better training. """
        scaler = MinMaxScaler(copy=True)
        if self.multi_output:
            # ToDo: Modularise!
            input_data_np = []
            output_data_np = []
            i = 0
            for c in self.config['loss'].get('multi_loss'):
                if c == 'ce':
                    input_data_np.append(self.input_data_np[i])
                    output_data_np.append(self.output_data_np[i])
                else:
                    # Transform the data if we are using a MAE or MSE or Corr loss
                    input_data_np.append(scaler.fit_transform(self.input_data_np[i]))
                    output_data_np.append(scaler.fit_transform(self.output_data_np[i]))
                i += 1
            self.output_data_np = output_data_np
            self.input_data_np = input_data_np
        else:
            # Don't transform if we don't have CE
            if self.config['loss']['loss_type'] != 'ce':
                self.input_data_np = scaler.fit_transform(self.input_data_np)
                self.output_data_np = scaler.fit_transform(self.output_data_np)

    def sample(self, args):
        """
        Reparameterization trick by sampling from an isotropic unit Gaussian.
            i.e. instead of sampling from Q(z|X), sample epsilon = N(0,I)
            z = z_mean + sqrt(var) * epsilon
        from https://github.com/s-omranpour/X-VAE-keras/blob/master/VAE/VAE_MMD.ipynb
            Arguments args (tensor): mean and log of variance of Q(z|X)
            Returns z (tensor): sampled latent vector
        """
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        zs = tf.zeros(K.shape(z_mean))
        # by default, random_normal has mean = 0 and std = 1.0
        # Now lets choose the sampling type we do
        epsilon = None
        if self.sample_method == 'normal':
            epsilon = K.random_normal(shape=(batch, dim), seed=self.seed)
            return z_mean + K.exp(0.5 * z_log_var) * epsilon
        elif self.sample_method == 'negbinom':
            epsilon = K.random_binomial(shape=(batch, dim), seed=self.seed)
            return z_mean + K.exp(0.5 * z_log_var) * epsilon
        elif self.sample_method == 'bimodal':
            # Make a mask for which values should go in which tensors based on whether they are < or > 0
            grp_1 = tf.cast(tf.math.greater(z_mean, zs), tf.float32)
            grp_2 = tf.cast(tf.math.greater_equal(zs, z_mean), tf.float32)
            epsilon_1 = K.random_normal(shape=(batch, dim), mean=1, stddev=1, seed=self.seed)
            epsilon_2 = K.random_normal(shape=(batch, dim), mean=-1, stddev=1, seed=self.seed)
            # Zero out where we don't have values matching (i.e. to make th epsilons correspond to the correct zmean
            epsilon_1 = epsilon_1*grp_1
            epsilon_2 = epsilon_2*grp_2
            epsilon = epsilon_1 + epsilon_2 #tf.keras.layers.Concatenate(axis=1)([epsilon_1, epsilon_2])
            return z_mean + K.exp(0.5 * z_log_var) * epsilon

    def set_seed(self):
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)

    def setup_test_train_data(self, train_percent):
        """  Split the data into training and test datasets. """
        n_data = len(self.input_data_np) if not self.multi_output else len(self.input_data_np[0])
        train_split = int(train_percent/100 * n_data)
        indices = np.random.permutation(n_data)
        training_idx, test_idx = indices[:train_split], indices[train_split:]
        if self.multi_output:
            # ToDo: Refactor to make it arbitary size.
            self.training_input_np = [self.input_data_np[0][training_idx], self.input_data_np[1][training_idx]]
            self.training_output_np = [self.output_data_np[0][training_idx], self.output_data_np[1][training_idx]]
            self.test_input_np = [self.input_data_np[0][test_idx], self.input_data_np[1][test_idx]]
            self.test_output_np = [self.output_data_np[0][test_idx], self.output_data_np[1][test_idx]]
        else:
            self.training_input_np = self.input_data_np[training_idx]
            self.training_output_np = self.output_data_np[training_idx]
            self.test_input_np = self.input_data_np[test_idx]
            self.test_output_np = self.output_data_np[test_idx]

    def get_inputs(self):
        if self.model_type == 'default' or self.model_type == 'None' or not self.model_type:
            return self.default_inputs()
        else:
            msg = self.u.msg.msg_arg_err("get_inputs", "model_type", self.model_type, self.model_types)
            self.u.err_p([msg])
            raise VAEException(msg)

    def default_inputs(self):
        if self.multi_output:
            self.inputs_x = [Input(shape=(self.input_size[0],), name='default_input_0'),
                             Input(shape=(self.input_size[1],), name='default_input_1')]
        else:
            self.inputs_x = Input(shape=(self.input_size,), name='default_input')
        return self.inputs_x

    def build_layer(self, num_nodes: int, prev_layer, activation_fn='selu'):
        x = Dense(num_nodes, activation=activation_fn)(prev_layer)
        # Perform batch normalisation
        if self.batch_norm:
            x = BatchNormalization()(x)
        return x

    def build_encoder(self):
        # Check if multi - if so we need to concatenate our layers see:
        # https://github.com/CancerAI-CL/IntegrativeVAEs/blob/master/code/models/xvae.py
        layer_start_idx = 0
        if self.multi_output:
            # If we have a fist layer if
            layer = self.encoding_config['layers'][0]
            # ToDo: refactor to handle arbitary number of inputs
            # For now we just need to ensure the first two layers are encoded separately
            encoding_0 = self.build_layer(layer[0]['num_nodes'], self.inputs_x[0], layer[0]['activation_fn'])
            encoding_1 = self.build_layer(layer[1]['num_nodes'], self.inputs_x[1], layer[1]['activation_fn'])
            self.encoding = Concatenate(axis=-1)([encoding_0, encoding_1])
            layer_start_idx = 1  # Since we have already done the first
        else:
            self.encoding = self.inputs_x
        # Now for subsequent layers we run this
        for layer_idx in range(layer_start_idx, len(self.encoding_config['layers'])):
            layer = self.encoding_config['layers'][layer_idx]
            self.encoding = self.build_layer(layer['num_nodes'], self.encoding, layer['activation_fn'])

        return self.encoding

    def build_decoder(self):
        # Generate input to the decoder which is based on random sampling (we then compare this to the reconstructed
        # values using the input as our output
        self.latent_inputs = Input(shape=(self.latent_config['num_nodes'],), name='z_sampling')
        self.decoding = self.latent_inputs
        # Here we also need to check for the multi output if so we don't do the last layer normally
        layer_end_idx = len(self.decoding_config['layers']) if not self.multi_output else \
            len(self.decoding_config['layers']) - 1
        for layer_idx in range(0, layer_end_idx):
            layer = self.decoding_config['layers'][layer_idx]
            self.decoding = self.build_layer(layer['num_nodes'], self.decoding, layer['activation_fn'])

        # Add final layer to the decoder that matches the input size & use a sigmoid activation
        # If we have a multi layer output, i.e. a loss function that spans the two, we want x separate layers
        if self.multi_output:
            self.decoding = self.build_multi_output(self.decoding)
        else:
            self.decoding = self.build_layer(self.output_size, self.decoding, self.output_activation_fn)
        return self.decoding

    def build_multi_output(self, decoding_layer):
        """ ToDo: Modularise this to have an arbitary number of layers. """
        # Add in the final layer
        decoder_0 = self.build_layer(self.decoding_config['layers'][-1][0]['num_nodes'], decoding_layer,
                                     self.output_activation_fn)
        decoder_0 = self.build_layer(self.output_size[0], decoder_0, self.output_activation_fn)

        # Add in final layer for decoder 2
        decoder_1 = self.build_layer(self.decoding_config['layers'][-1][1]['num_nodes'], decoding_layer,
                                     self.output_activation_fn)
        decoder_1 = self.build_layer(self.output_size[1], decoder_1, self.output_activation_fn)
        return [decoder_0, decoder_1]

    def build_embedding(self):
        # if self.loss.distance_metric == 'mmd':
        #     self.latent_z = Dense(self.latent_config['num_nodes'], name='z')(self.encoding)
        # else:
        self.latent_z_mean = Dense(self.latent_config['num_nodes'], name='z_mean')(self.encoding)
        self.latent_z_log_sigma = Dense(self.latent_config['num_nodes'], name='z_log_sigma',
                                        kernel_initializer='zeros')(self.encoding)
        self.latent_z = Lambda(self.sample, output_shape=(self.latent_config['num_nodes'],), name='z')([
            self.latent_z_mean, self.latent_z_log_sigma])

    def build_model(self):
        # Set random seed
        self.set_seed()

        # Convert the inputs into our first encoding layer
        self.get_inputs()

        # Build encoder based on the configs
        # ------------ Encoding Layer -----------------
        self.encoding = self.build_encoder()

        # ------------ Embedding Layer --------------
        self.build_embedding()

        # Initialise the encoder
        # if self.loss.distance_metric == 'mmd':
        #     self.encoder = Model(self.inputs_x, self.latent_z,
        #                          name='encoder')
        # else:
        self.encoder = Model(self.inputs_x, [self.latent_z_mean, self.latent_z_log_sigma, self.latent_z], name='encoder')
        self.encoder.summary()

        # Build the decoder network
        # ------------ Dense out -----------------
        self.decoding = self.build_decoder()
        self.decoder = Model(self.latent_inputs, self.decoding, name='decoder')
        self.decoder.summary()

        # ------------ Out -----------------------
        # if self.loss.distance_metric == 'mmd':
        #     self.outputs_y = self.decoder(self.encoder(self.inputs_x))
        # else:

        self.outputs_y = self.decoder(self.encoder(self.inputs_x)[2])
        self.vae = Model(self.inputs_x, self.outputs_y, name='VAE_' + self.vae_label + '_scivae')
        self.vae.add_loss(self.loss.get_loss(self.inputs_x, self.outputs_y, self.latent_z, self.latent_z_mean,
                                  self.latent_z_log_sigma))

    def compile(self, optimizer=None):
        optimizer = optimizer if optimizer is not None else self.optimiser(self.config['optimiser']['name'], self.config['optimiser']['params'])
        self.vae.compile(optimizer=optimizer)
        print(self.vae.summary())

    def encode(self, method='default', epochs=50, batch_size=50, train_percent=85.0, logging_dir=None, logfile=None):
        """
        Encodes the data based on a given method.

        Parameters
        ----------
        method: "default" is the default string. Will be extended to take more options.

        Returns
        -------

        """
        # setup the test and training data
        logging_dir = logging_dir if logging_dir else f'/tmp/SCIVAE-{str(datetime.now()).split(" ")[0]}_vae'
        logfile = f'{logging_dir}{logfile}' if logfile else f'{logging_dir}vae-{str(datetime.now()).replace(" ", "").replace(":", "").replace(".", "")}.csv'
        self.setup_test_train_data(train_percent)
        self.model_type = method

        if method == 'default':
            self._encode_default()
        elif method == 'optimise':
            self._encode_optimise()
        elif method != 'optimise' and method != 'default':
            msg = self.u.msg.msg_arg_err("encode", "method", method, ['default', 'optimise'])
            self.u.err_p([msg])
            raise VAEException(msg)

        csv_logger = CSVLogger(logfile, append=True, separator=',')
        # Otherwise fit the VAE with the training and test data
        self.vae.fit(self.training_input_np,
                     epochs=epochs,
                     batch_size=batch_size,
                     shuffle=True,
                     validation_data=(self.test_input_np, None),
                     callbacks=[TensorBoard(log_dir=logging_dir), csv_logger]
        )

    def _encode_default(self):
        """
        Builds a default (2 layer) AE. With 3 neurons.

        Returns
        -------
        encoded_data: the input data encoded into 3 latent variables.
        """
        self.build_model()
        self.compile()

    def _encode_optimise(self):
        """
        Optimise the encoding using pyswarm.

        Returns
        -------

        """
        return

    def get_encoded_data(self, encoding_type="z"):
        """
        Gets the encoded data, if no encoder has been run, runs the default AE.
        Encoding type refers to 2 = latent, 0 = mean, 1 = variance

        Returns
        -------

        """
        encoding_map = {"z": 2, "z_mean": 0, "z_log_var": 1}
        encoding_type = encoding_map.get(encoding_type)
        if self.encoded_data is not None:
            # if self.loss.distance_metric == 'mmd':
            #     return self.encoded_data
            if encoding_type is not None:
                return self.encoded_data[encoding_type]
            return self.encoded_data
        if self.encoder is not None:
            self.encoded_data = self.encoder.predict(self.input_data_np)
            # if self.loss.distance_metric == 'mmd':
            #     return self.encoded_data
            if encoding_type is not None:
                return self.encoded_data[encoding_type]
            return self.encoded_data
        if self.input_data_np is not None:
            self.encode(method="default")
            self.encoded_data = self.encoder.predict(self.input_data_np)
            # if self.loss.distance_metric == 'mmd':
            #     return self.encoded_data
            if encoding_type is not None:
                return self.encoded_data[encoding_type]
        self.u.warn_p(["WARN: get_encoded_data. \n \t No data or encoded data. Returning None."])

    def encode_new_data(self, data_np, encoding_type="z", scale=True):
        """
        Encodes new data using the existing encoder and decoder.

        Parameters
        ----------
        data_np

        Returns
        -------

        """
        encoding_map = {"z": 2, "z_mean": 0, "z_log_var": 1}
        encoding_type = encoding_map.get(encoding_type)
        if scale:
            scaler = MinMaxScaler(copy=True)
            data_np = scaler.fit_transform(data_np)
        if self.encoder is not None:
            if encoding_type is not None:
                return self.encoder.predict(data_np)[encoding_type]
            return self.encoder.predict(data_np)
        if data_np is not None:
            self.encode(method="default")
            if encoding_type is not None:
                return self.encoder.predict(data_np)[encoding_type]
            return self.encoder.predict(data_np)
        self.u.warn_p(["WARN: encode_data. \n \t No data or initial encoding data has been generated."
                       " Returning None."])

    def get_config(self) -> dict:
        return self.config

    def set_validation_score(self, validation_score: float):
        self.validation_score = validation_score

    def get_goodness_score(self, metric='reconstruction'):
        """
        Get the goodness. Metric can be: 'equal', 'reconstruction' or 'prediction'
        Parameters
        ----------
        metric

        Returns
        -------

        """
        if self.goodness is not None:
            return self.goodness
        if metric == 'combined':
            if self.validation_score is None:
                return None
            self.goodness = 0.5 * ((1 - self.validation_score) + self.get_reconstruction_loss())
            return self.goodness
        elif metric == 'reconstruction':
            self.goodness = self.get_reconstruction_loss()
            return self.goodness
        elif metric == 'prediction':
            if self.validation_score is None:
                return None
            self.goodness = 1 - self.validation_score
            return self.goodness
        msg = self.u.msg.msg_arg_err("get_goodness_score", "metric", metric, ['combined', 'reconstruction', 'prediction'])
        self.u.err_p([msg])
        raise VAEException(msg)

    def get_reconstruction_loss(self):
        """
        Return the loss of the VAE.

        Returns
        -------

        """
        if self.test_loss is not None:
            return self.test_loss
        self.test_loss = self.vae.evaluate(self.test_input_np, self.test_output_np)
        return self.test_loss

    def print(self) -> None:
        print("----------------------------")
        print(self.config)
        print(f'Validation score: \t \t{self.validation_score}')
        print(f'Reconstruction loss: \t \t{self.get_reconstruction_loss()}')
        print(f'Goodness score: \t \t{self.get_goodness_score()}')
        print("----------------------------")

    def optimiser(self, optimiser_name: str, params: dict):
        """
        gets a keras
        Parameters
        ----------
        optimiser_name
        params

        Returns
        -------

        """
        learning_rate = params.get('learning_rate') if params.get('learning_rate') else 0.01
        beta_1 = params.get('beta_1') if params.get('beta_1') else 0.9
        beta_2 = params.get('beta_2') if params.get('beta_2') else 0.999
        amsgrad = params.get('amsgrad') if params.get('amsgrad') else False
        rho = params.get('rho') if params.get('rho') else 0.95
        nesterov = params.get('nesterov') if params.get('nesterov') else False
        momentum = params.get('momentum') if params.get('momentum') else 0.9
        decay = params.get('decay') if params.get('decay') else 0.01
        epsilon = params.get('epsilon') if params.get('epsilon') else None
        if optimiser_name == 'adamax':
            return optimizers.Adamax(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2)

        elif optimiser_name == 'adam':
            return optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1, decay=decay,
                                   beta_2=beta_2, amsgrad=amsgrad)

        elif optimiser_name == 'adadelta':
            return optimizers.Adadelta(learning_rate=learning_rate, rho=rho)

        elif optimiser_name == 'adagrad':
            return optimizers.Adagrad(learning_rate=learning_rate)

        elif optimiser_name == 'rmsprop':
            return optimizers.RMSprop(learning_rate=learning_rate, rho=rho)

        elif optimiser_name == 'sgd':
            return optimizers.SGD(learning_rate=learning_rate, momentum=momentum, nesterov=nesterov)

        msg = self.u.msg.msg_arg_err("optimiser", "optimiser_name", optimiser_name, [
            'adamax', 'adam', 'adadelta', 'adagrad', 'rmsprop', 'sgd'
        ])
        self.u.err_p([msg])
        raise VAEException(msg)

    def save(self, weight_file_path='model_weights.h5', optimizer_file_path='model_optimiser.pkl',
             config_json='config.json', save_data=False, data_filename='data.csv'):
        """ Save data, design, and optimiser state for recreation of VAE. """
        self.vae.save_weights(weight_file_path)
        with open(optimizer_file_path, 'wb') as f:
            pickle.dump(self.vae.optimizer, f)
        # Save the user options as a json file.
        with open(config_json, 'w+') as f:
            json.dump(self.config, f)
        # Optionally save the training data
        if save_data:
            with open(f'input_{data_filename}', 'wb') as f:
                pickle.dump(self.input_data_np, f)
            with open(f'output_{data_filename}', 'wb') as f:
                pickle.dump(self.output_data_np, f)

    def load(self, weight_file_path='model_weights.h5', optimizer_file_path='model_optimiser.pkl',
             config_json='config.json', load_data=False, data_filename='data.csv'):
        """ Load previously saved config and data"""
        # Load the users config file
        with open(config_json, 'r+') as f:
            config = json.load(f)
        self.__init_from_config(config)
        # Load the training data
        if load_data:
            with open(f'input_{data_filename}', 'rb') as f:
                self.input_data_np = pickle.load(f)
            with open(f'output_{data_filename}', 'rb') as f:
                self.output_data_np = pickle.load(self.output_data_np)
        # Build the model
        self.build_model()
        # Load weights
        self.vae.load_weights(weight_file_path)
        # Load the optimiser
        with open(optimizer_file_path, 'rb') as f:
            self.compile(pickle.load(f))
