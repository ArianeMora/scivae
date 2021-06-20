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

from sciutil import SciUtil, SciException

from keras import backend as K
import tensorflow as tf
import tensorflow_probability as tfp
import math


class LossException(SciException):
    def __init__(self, message=''):
        Exception.__init__(self, message)


class Loss:

    """
    References:
        https://github.com/pren1/keras-MMD-Variational-Autoencoder/blob/master/Keras_MMD_Variational_Autoencoder.ipynb
        https://github.com/s-omranpour/X-VAE-keras/blob/master/VAE/VAE_MMD.ipynb
        https://github.com/ShengjiaZhao/MMD-Variational-Autoencoder/blob/master/mmd_vae.py
        https://github.com/CancerAI-CL/IntegrativeVAEs/blob/master/code/models/mmvae.py
        https://keras.io/guides/training_with_built_in_methods/
    """

    def __init__(self, loss_type: str, distance_metric: str, mmd_weight: float, multi_loss=None, sciutil=None,
                 mmcd_method='k',  beta=1.0):
        """

        Parameters
        ----------
        loss_type
        distance_metric
        learning_rate
        sciutil
        """
        self.u = sciutil if sciutil is not None else SciUtil()
        self.mmd_weight = mmd_weight

        self.loss_types = ['ce', 'mse', 'cor', 'cor-mse', 'multi', 'mae']
        if loss_type not in self.loss_types:
            msg = self.u.msg.msg_arg_err("Loss __init__", "loss", loss_type, self.loss_types)
            self.u.err_p([msg])
            raise LossException(msg)
        self.loss_type = loss_type
        self.distance_types = ['kl', 'mmd', 'mmcd', 'kl-mmd', 'bmmd']
        if self.loss_type == 'multi' and multi_loss is None:
            msg = self.u.msg.msg_arg_err("Loss __init__", "multi", multi_loss, 'Multi loss requires a dict of losss '
                                                                              'and corresponding idxs: {mse: [0,3]}')
            self.u.err_p([msg])
            raise LossException(msg)
        self.multi_loss_fn = multi_loss
        if distance_metric not in self.distance_types:
            msg = self.u.msg.msg_arg_err("Loss __init__", "distance_metric", distance_metric, self.distance_types)
            self.u.err_p([msg])
            raise LossException(msg)
        if distance_metric == 'mmd':
            self.mmd_weight = self.mmd_weight * 1.0
            # Check weight parameter is correct format otherwise throw error
            if not isinstance(self.mmd_weight, float) or mmd_weight < 0:
                msg = self.u.msg.msg_arg_err("Loss __init__", "mmd_weight", mmd_weight, ['>0', 'must be a float'])
                self.u.err_p([msg])
                raise LossException(msg)
        self.distance_metric = distance_metric
        self.beta = beta or 1.0
        self.mmcd_method = mmcd_method

    def get_loss(self, inputs_x, outputs_y, latent_z, latent_z_mean, latent_z_log_sigma) -> float:
        """
        Get the loss as defined by the configuration setup.
        Returns
        -------

        """
        distance = 1.0
        # Get the distance between the predicted distribution and the expected dist
        if self.distance_metric == 'kl':
            distance = self.get_kl_distance(latent_z_mean, latent_z_log_sigma)
        elif self.distance_metric == 'mmd':
            distance = self.get_mmd_distance(latent_z)
        elif self.distance_metric == 'bmmd':
            distance = self.get_bimodal_mmd_distance(latent_z)
        elif self.distance_metric == 'mmcd':
            istance = self.get_mmcd_distance(latent_z, latent_z_mean, latent_z_log_sigma, self.mmcd_method)
        elif self.distance_metric == 'kl-mmd':
            distance = self.get_mmd_distance(latent_z) + self.get_kl_distance(latent_z_mean, latent_z_log_sigma)
        else:
            msg = self.u.msg.msg_arg_err("Loss __init__", "distance_metric", self.distance_metric, self.distance_types)
            self.u.err_p([msg])
            raise LossException(msg)
        reconstruction_loss = 1000000000
        # Get reconstruction loss
        if self.loss_type == 'ce':
            reconstruction_loss = self.get_binary_crossentropy_loss(inputs_x, outputs_y)
        elif self.loss_type == 'mse':
            reconstruction_loss = self.get_mean_squared_error_loss(inputs_x, outputs_y)
        elif self.loss_type == 'cor':
            reconstruction_loss = self.get_correlation_loss(inputs_x, outputs_y)
        elif self.loss_type == 'cor-mse':
            reconstruction_loss = self.get_correlation_mse_loss(inputs_x, outputs_y)
        elif self.loss_type == 'mae':
            reconstruction_loss = self.get_mean_absolute_error_loss(inputs_x, outputs_y)
        elif self.loss_type == 'multi':
            reconstruction_loss = 0
            # ToDo: Modularise.
            loss_idx = 0
            for loss_method in self.multi_loss_fn:
                if loss_method == 'ce':
                    reconstruction_loss += self.get_binary_crossentropy_loss(inputs_x[loss_idx],
                                                                             outputs_y[loss_idx])
                elif loss_method == 'mse':
                    reconstruction_loss += self.get_mean_squared_error_loss(inputs_x[loss_idx],
                                                                            outputs_y[loss_idx])
                elif loss_method == 'cor':
                    reconstruction_loss += self.get_correlation_loss(inputs_x[loss_idx],
                                                                     outputs_y[loss_idx])
                elif loss_method == 'mae':
                    reconstruction_loss += self.get_mean_absolute_error_loss(inputs_x[loss_idx],
                                                                             outputs_y[loss_idx])
                loss_idx += 1
        else:
            msg = self.u.msg.msg_arg_err("Loss __init__", "loss", self.loss_type, self.loss_types)
            self.u.err_p([msg])
            raise LossException(msg)
        # Return the mean between the reconstruction loss and the learning rate * the distance between the distributions
        return K.mean(reconstruction_loss + self.mmd_weight * distance)

    @staticmethod
    def get_binary_crossentropy_loss(input_x, output_y):
        """
        From: https://github.com/CancerAI-CL/IntegrativeVAEs/blob/master/code/models/common.py
        Parameters
        ----------
        train_inputs
        train_reconstruction

        Returns
        -------

        """
        return K.sum(K.binary_crossentropy(input_x, output_y, from_logits=True), axis=1)

    @staticmethod
    def get_mean_squared_error_loss(input_x, output_y):
        """ https://github.com/CancerAI-CL/IntegrativeVAEs/blob/master/code/models/common.py """
        return K.sum(K.square(input_x - output_y), axis=1)

    @staticmethod
    def get_mean_absolute_error_loss(input_x, output_y):
        """ MAE """
        return K.sum(K.abs(input_x - output_y), axis=1)


    @staticmethod
    def get_correlation_loss(input_x, output_y):
        """ Check this function! WIP.
        https://stackoverflow.com/questions/46619869/how-to-specify-the-correlation-coefficient-as-the-loss-function-in-keras/46620771 """
        mx = K.mean(input_x)
        my = K.mean(output_y)
        xm, ym = input_x - mx, output_y - my
        r_num = K.sum(tf.multiply(xm, ym))
        r_den = K.sqrt(tf.multiply(K.sum(K.square(xm)), K.sum(K.square(ym))))
        r = r_num / r_den

        r = K.maximum(K.minimum(r, 1.0), -1.0)
        return 1 - r

    def get_correlation_mse_loss(self, input_x, output_y):
        """ WIP. """
        return 0.5 * (self.get_correlation_loss(input_x, output_y) + self.get_mean_squared_error_loss(input_x, output_y))

    @staticmethod
    def get_correlation_loss_per_dataype(input_x, output_y):
        """ WIP. """
        mx = K.mean(input_x)
        my = K.mean(output_y)
        xm, ym = input_x - mx, output_y - my
        r_num = K.sum(tf.multiply(xm, ym))
        r_den = K.sqrt(tf.multiply(K.sum(K.square(xm)), K.sum(K.square(ym))))
        r = r_num / r_den

        r = K.maximum(K.minimum(r, 1.0), -1.0)
        return 1 - r

    @staticmethod
    def get_kl_distance(z_mean, z_log_sigma):
        """ Resources:
        https://keras.io/examples/generative/vae/
        https://github.com/geyang/variational_autoencoder_pytorch
        """
        # KL regularizer. this is the KL of q(z|x) given that the target distribution is N(0,1)
        kl_loss = 1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5

        return kl_loss

    def get_mmcd_distance(self, latent_z, latent_z_mean, latent_z_log_sigma, method='k'):
        """
        WIP.
        https://wis.kuleuven.be/statdatascience/robust/software
        https://www.researchgate.net/publication/343471031_Outlier_detection_in_non-elliptical_data_by_kernel_MRCD
        https://wis.kuleuven.be/statdatascience/robust/papers/publications-2018/hubertdebruynerousseeuw-mcdext-wires-2018.pdf

        This code was translated from the paper: 10.1007/s11063-019-10090-0
        Maximum Mean and Covariance Discrepancy for Unsupervised Domain Adaptation
        by Wenju Zhang, Xiang Zhang, Long Lan & Zhigang Luo

        Code was in Matlab provided by Wenju Zhang accessed from github on 07/11/2020
        https://github.com/wj-zhang/McDA

        https://stats.stackexchange.com/questions/14673/measures-of-similarity-or-distance-between-two-covariance-matrices
        SchreursEtAl_KMRCD_arXiv_v1_2020.pdf

        """
        beta = self.beta or 1.0

        batch_size = K.shape(latent_z)[0]
        latent_dim = K.int_shape(latent_z)[1]
        loss_mmcd = 0  # If they don't set a proper param, then we just run with no MMCD i.e returning 0.

        # Randomly sample from a normal distribution
        true_samples = K.random_normal(shape=(batch_size, latent_dim), mean=0., stddev=1.)

        if method == 'k':  # If we're running mmcd on a kernal
            # Compute loss between the training latent space and normal data
            x_kernel = self.compute_kernel(true_samples, true_samples)
            y_kernel = self.compute_kernel(latent_z, latent_z)
            xy_kernel = self.compute_kernel(true_samples, latent_z)
            loss_mmd = K.mean(x_kernel) + K.mean(y_kernel) - 2 * K.mean(xy_kernel)
            cov_dif = K.sqrt(K.sum(K.square(tfp.stats.covariance(x_kernel) - tfp.stats.covariance(y_kernel))))
            loss_mmcd = K.sqrt(loss_mmd + (beta * cov_dif))  # Double check sqrt --> whether we need to square the mmd
            return loss_mmcd
        elif method == 'bk':  # Means we're running mmcd on the basic version with semi kernal
            # Since the covariance matrix will just be 1's on the diag (when we have an isotropic gaussian) we can just
            # use the identity matrix
            cov_yy = self.compute_kernel(tfp.stats.covariance(latent_z_log_sigma),
                                         tfp.stats.covariance(latent_z_log_sigma))
            cov_true = tfp.stats.covariance(true_samples)
            cov_ker = self.compute_kernel(cov_true, cov_true)
            cov_dif = K.sum(K.square(cov_ker - cov_yy))

            # Since the means of a standard normal are zeros we just calculate
            # the norm of the vector of our latent dataset
            x_kernel = self.compute_kernel(true_samples, true_samples)
            y_kernel = self.compute_kernel(latent_z, latent_z)
            means_dif = K.sum(K.square(K.mean(x_kernel) - K.mean(y_kernel)))
            loss_mmcd = K.sqrt(means_dif + (beta * cov_dif))
            return loss_mmcd
        elif method == 'b':
            # Without kernel below
            # means_dif = tf.square(tf.math.reduce_mean(latent_z_mean))
            # cov_dif = K.sqrt(K.square(-1 + K.min(latent_z_log_sigma)) + K.square(1 - K.max(latent_z_log_sigma)))/2
            # loss_mmcd = (means_dif + (beta * cov_dif))
            # means_dif = tf.square(tf.math.reduce_mean(latent_z_mean))
            means_dif = tf.square(true_samples - latent_z_mean)
            cov_dif = K.sqrt(K.square(1 + K.min(latent_z_log_sigma)) + K.square(1 - K.max(latent_z_log_sigma)))/2
            loss_mmcd = (means_dif + (beta * cov_dif))
        return loss_mmcd

    def get_mmd_distance(self, latent_z):
        """
        https://github.com/Saswatm123/MMD-VAE/blob/master/MMD_VAE.ipynb
        https://learning.mpi-sws.org/mlss2016/slides/cadiz16_2.pdf
        http://abdulfatir.com/Implicit-Reparameterization/
        Parameters
        ----------
        train_latent:               latent code
        train_reconstruction:       restructured data
        train_inputs:               training data

        Returns
        -------
        new loss
        """
        batch_size = K.shape(latent_z)[0]

        latent_dim = K.int_shape(latent_z)[1]

        # Randomly sample from a normal distribution
        true_samples = K.random_normal(shape=(batch_size, latent_dim), mean=0., stddev=1.)

        # Compute loss between the training latent space and normal data
        x_kernel = self.compute_kernel(true_samples, true_samples)
        y_kernel = self.compute_kernel(latent_z, latent_z)
        xy_kernel = self.compute_kernel(true_samples, latent_z)
        loss_mmd = K.mean(x_kernel) + K.mean(y_kernel) - 2 * K.mean(xy_kernel)

        # add mmd loss and true loss
        return loss_mmd

    def get_bimodal_mmd_distance(self, latent_z):
        """ Compute MMD twice, once with each different mean. """

        batch_size = K.shape(latent_z)[0]
        dim = K.int_shape(latent_z)[1]
        zs = tf.zeros(K.shape(latent_z))#+ tf.float32
        # # Make a mask for which values should go in which tensors based on whether they are < or > 0
        # # Zero out where we don't have values matching (i.e. to make th epsilons correspond to the correct zmean
        # grp_1 = tf.math.greater(latent_z, zs)
        # grp_2 = tf.math.greater_equal(zs, latent_z)
        #
        # # Randomly sample from a normal distribution
        # #slice_1 = tf.slice(latent_z, [0, 0], [batch_size, dim])
        #
        # true_samples = K.random_normal(shape=(batch_size, dim), mean=1., stddev=1.) # Seed?
        #
        # # Compute loss between the training latent space and normal data
        # true_grp_1 = tf.boolean_mask(true_samples, grp_1)
        # latent_z_1 = tf.boolean_mask(latent_z, grp_1)
        #
        # x_kernel = self.compute_kernel(true_grp_1, true_grp_1)
        # y_kernel = self.compute_kernel(latent_z_1, latent_z_1)
        # xy_kernel = self.compute_kernel(true_grp_1, latent_z_1)
        # loss_mmd = K.mean(x_kernel) + K.mean(y_kernel) - 2 * K.mean(xy_kernel)
        #
        # # Do the same for group 2
        # # true_samples = K.random_normal(shape=(batch_size, grp_2), mean=-1., stddev=1.)
        # # slice_2 = tf.slice(latent_z, [0, grp_1], [batch_size, grp_2])
        # # Compute loss between the training latent space and normal data
        # true_grp_2 = tf.boolean_mask(true_samples, grp_2)
        # latent_z_2 = tf.boolean_mask(latent_z, grp_2)
        # x_kernel = self.compute_kernel(true_grp_2, true_grp_2)
        # y_kernel = self.compute_kernel(latent_z_2, latent_z_2)
        # xy_kernel = self.compute_kernel(true_grp_2, latent_z_2)
        # loss_mmd += K.mean(x_kernel) + K.mean(y_kernel) - 2 * K.mean(xy_kernel)
        # New loss
        grp_1 = tf.cast(tf.math.greater(latent_z, zs), tf.float32)
        grp_2 = tf.cast(tf.math.greater_equal(zs, latent_z), tf.float32)
        epsilon_1 = K.random_normal(shape=(batch_size, dim), mean=1, stddev=1)
        epsilon_2 = K.random_normal(shape=(batch_size, dim), mean=-1, stddev=1)
        # Zero out where we don't have values matching (i.e. to make th epsilons correspond to the correct zmean
        epsilon_1 = epsilon_1 * grp_1
        epsilon_2 = epsilon_2 * grp_2
        epsilon = epsilon_1 + epsilon_2  # tf.keras.layers.Concatenate(axis=1)([epsilon_1, epsilon_2])
        tue_samples = 0 + K.exp(0.5) * epsilon
        x_kernel = self.compute_kernel(tue_samples, tue_samples)
        y_kernel = self.compute_kernel(latent_z, latent_z)
        xy_kernel = self.compute_kernel(tue_samples, latent_z)
        loss_mmd = K.mean(x_kernel) + K.mean(y_kernel) - 2 * K.mean(xy_kernel)
        # add mmd loss and true loss
        return loss_mmd

    @staticmethod
    def compute_kernel(x, y):
        """
        https://stats.stackexchange.com/questions/239008/rbf-kernel-algorithm-python
        Parameters
        ----------
        x
        y

        Returns
        -------

        """
        x_size = K.shape(x)[0]
        y_size = K.shape(y)[0]
        dim = K.shape(x)[1]
        tiled_x = K.tile(K.reshape(x, [x_size, 1, dim]), [1, y_size, 1])
        tiled_y = K.tile(K.reshape(y, [1, y_size, dim]), [x_size, 1, 1])
        return K.exp(-K.mean(K.square(tiled_x - tiled_y), axis=2) / K.cast(dim, 'float32'))
