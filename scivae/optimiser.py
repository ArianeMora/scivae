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

from functools import reduce
import numpy as np
from pyswarm import pso
from operator import add
import random

from scivae import VAE

"""
Code adapted from:
    - https://github.com/harvitronix/neural-network-genetic-algorithm/blob/master/optimizer.py
    - http://lethain.com/genetic-algorithms-cool-name-damn-simple/
    - http://francky.me/doc/mrf2011-HEC-ISIR-ENS_en.pdf
"""
# Using pyswarm for optimising the VAE: http://www.icst.pku.edu.cn/zlian/docs/20181023162743343059.pdf


class Optimiser(object):

    def __init__(self, data_np: np.array, labels: list, bounds: dict, validation_method='combined', retain=0.4,
                 random_select=0.1, mutate_chance=0.2, number_children=2, epochs=100, batch_size=5, limits=None,
                 validation_params=None):
        """

        Parameters
        ----------
        data_np
        labels
        bounds (dict):                  {'latent_num_nodes': [1, 2, .., n], 'encoding': [[1, 1, 2, ..., n], [], decoding: ...
        optimisation_method (string):   'combined', 'reconstruction', 'accuracy'
        retain
        random_select
        mutate_chance
        """
        self.mutate_chance = mutate_chance
        self.random_select = random_select
        self.number_children = number_children
        self.retain = retain
        self.data_np = data_np
        self.num_features = len(data_np[0])
        self.labels = labels
        self.bounds = bounds
        self.limits = limits
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_method = validation_method
        self.validation_params = validation_params if validation_params else \
            {'metric': 'combined', 'params': ('rf', 'accuracy', 10)}

    def pso_optimise_optimiser(self, config: dict, args):
        """
        Here we have chosen a configuration and we are now optimising the selected optimisation function
        of the VAE.

        Parameters
        ----------
        optimising_vars
        args

        Returns
        -------

        """
        config['optimiser']['params'] = args  # pass the new arguments to the optimiser function
        vae = VAE(self.data_np, self.labels, config, 'vae')
        return self.fitness(vae)

    def optimise_optimiser_params(self, upper_bounds: list, lower_bounds: list, optimal_config: dict):
        # Lower and upper bounds are usually 0 to 1,
        xopt, fopt = pso(self.pso_optimise_optimiser, np.array(lower_bounds), np.array(upper_bounds),
                         args=optimal_config)
        print(xopt, fopt)
        return xopt, fopt

    def optimise_with_bounds(self, num_generations: int, population_size: int, bounds: dict, print_out=False):
        """
        Optimise the vaes within certain boundries
                bounds = {'latent_num_nodes': [1, 2, 3, 4],
                  'encoding': [[1, 2, 3, 4], [1, 2, 3, 4]],
                  'decoding': [[1, 2, 3, 4], [1, 2, 3, 4]],
                  'optimisers': ['adamax', 'adam', 'adadelta', 'adagrad', 'rmsprop', 'sgd']
                  }

        Parameters
        ----------
        num_generations
        population_size
        bounds
        print_out

        Returns
        -------

        """
        self.bounds = bounds
        vaes = self.create_population(population_size)
        avg_goodness = []
        for i in range(num_generations):
            # Train
            self.train_vaes(vaes)
            # Get accuracy
            avg_goodness.append(self.get_average_goodness(vaes))
            # evolve except last gen
            if i != num_generations - 1:
                # Do the evolution.
                vaes = self.evolve(vaes)

        # Sort the VAEs
        vaes = sorted(vaes, key=lambda x: x.get_goodness_score(self.validation_method)) # for descending , reverse=True

        # Print out the top networks
        if print_out:
            for vae in vaes:
                vae.print()

        return vaes

    def optimise_architecture(self, num_generations: int, population_size: int, max_encoding_layers=5,
                              max_decoding_layers=5, equal_encoding_decoding=True, optimisers=None,
                              increment=1, print_out=False):
        """

        Parameters
        ----------
        num_generations
        population_size
        optimisers
        max_encoding_layers
        max_decoding_layers
        equal_encoding_decoding
        print_out

        Returns
        -------

        """
        # options: ['adamax', 'adam', 'adadelta', 'adagrad', 'rmsprop', 'sgd']
        default_optimisers = ['adamax', 'adadelta', 'sgd'] if optimisers is None else optimisers
        encoder_configs_and_score = []
        best_fitness = 1000000000000
        best_vae = None
        for i in range(0, max_encoding_layers):
            # optimise parameters
            if self.limits is not None:
                bounds = {'latent_num_nodes': range(self.limits['latent_num_nodes']['min'],
                                                    self.limits['latent_num_nodes']['max'], increment),
                          'encoding': [],
                          'decoding': [],
                          'optimisers': default_optimisers
                          }
            else:
                bounds = {'latent_num_nodes': range(1, self.num_features, increment),
                          'encoding': [],
                          'decoding': [],
                          'optimisers': default_optimisers
                          }
            if equal_encoding_decoding:
                for j in range(i):
                    if self.limits is not None:
                        bounds['encoding'].append(range(self.limits['encoding']['min'], self.limits['encoding']['max'],
                                                        increment))
                        bounds['decoding'].append(range(self.limits['decoding']['min'], self.limits['decoding']['max'],
                                                        increment))
                    else:
                        bounds['encoding'].append(range(1, self.num_features, increment))
                        bounds['decoding'].append(range(1, self.num_features, increment))

            # Lets now get the configuration and score for these
            vaes = self.optimise_with_bounds(num_generations, population_size, bounds, print_out)
            vaes = sorted(vaes, key=lambda x: x.get_goodness_score(self.validation_method)) # For descending :  reverse=True
            # Store the vae with the best config and keep track of the loss so we can make sure we stop
            # if we get a loss less than expected
            fitness = self.fitness(vaes[0])
            encoder_configs_and_score.append({'config': vaes[0], 'fitness': fitness})
            # Essentially we want to make sure that we keep the most simple VAE (even if it's just a linear version)
            if fitness > best_fitness:
                print("Reached turning point! Returning")
                return best_vae, best_fitness
            else:
                # i.e. we want the smallest re
                best_fitness = fitness
                best_vae = vaes[0]
        print("Continuilly improving.")
        return encoder_configs_and_score

    @staticmethod
    def generate_config_from_bounds(bounds=dict):
        """
        Generate a random configuration of the vae configuration using the supplied bounds.
        Returns
        -------

        """
        loss = {'loss_type': 'mse', 'distance_metric': 'mmd', 'mmd_weight': 1}
        config = {'loss': loss, 'encoding': {'layers': []}, 'decoding': {'layers': []},
                  'latent': {'num_nodes': random.choice(bounds['latent_num_nodes'])},
                  'optimiser': {'params': {}}}
        # For each of the number of layers specified lets choose a random configuration
        for b in bounds['encoding']:
            config['encoding']['layers'].append({'num_nodes': random.choice(b), 'activation_fn': 'selu'})

        for b in bounds['decoding']:
            config['decoding']['layers'].append({'num_nodes': random.choice(b), 'activation_fn': 'selu'})

        # Choose the optimiser
        config['optimiser']['name'] = random.choice(bounds['optimisers'])
        return config

    def create_population(self, count: int) -> list:
        """

        Parameters
        ----------
        count:       Number of networks to generate, aka the size of the population

        Returns
        -------
        Population of vae objects
        """
        pop = []
        for _ in range(0, count):
            # Create a random vae configuration.
            vae = VAE(self.data_np, self.data_np, self.labels, self.generate_config_from_bounds(self.bounds))

            # Add the network to our population.
            pop.append(vae)

        return pop

    def fitness(self, vae: VAE):
        """
        Returns the fitness parameter of our VAE. This is either dependent on reconstruction error or
        based on prediction error.

        Parameters
        ----------
        vae

        Returns
        -------

        """
        # Choose validation
        if self.validation_method == 'reconstruction':
            return vae.get_reconstruction_loss()
        elif self.validation_method == 'combined':
            if vae.get_goodness_score('combined') is not None:
                return vae.get_goodness_score('combined')
        elif self.validation_method == 'prediction':
            if vae.get_goodness_score('prediction') is not None:
                return vae.get_goodness_score('prediction')

        # Otherwise we need to calculate the validation error
        from scivae import Validate

        vd = Validate(vae, self.labels)
        score = vd.predict(*self.validation_params['params'])
        vae.set_validation_score(score)

        # Now we need to get the correct score
        if self.validation_method == 'combined':
            return vae.get_goodness_score('combined')

        return score

    def grade(self, population: list):
        """
        Calculate fitness for a given population
        Parameters
        ----------
        population:         list: population of networks

        Returns
        -------
        avg. accuracy or reconstruction loss of a given population
        """

        summed = reduce(add, (self.fitness(network) for network in population))
        return summed / float((len(population)))

    def generate_config_from_parents(self, parent_1: VAE, parent_2: VAE) -> dict:
        """
        Randomly choose a combination of the parent's configurations.
        Parameters
        ----------
        parent_1
        parent_2

        Returns
        -------

        """
        parent_1_config = parent_1.get_config()
        parent_2_config = parent_2.get_config()
        # We now want to choose just between the parameters of the parents randomly
        loss = {'loss_type': 'mse', 'distance_metric': 'mmd', 'mmd_weight': 1}
        config = {'loss': loss,
                  'encoding': {'layers': []},
                  'decoding': {'layers': []},
                  'latent': {'num_nodes': random.choice(self.bounds['latent_num_nodes'])},
                  'optimiser':  {'params': {}}
                  }
        # For each of the number of layers specified lets choose a random configuration
        i = 0
        for parent_1_layer in parent_1_config['encoding']['layers']:
            parent_2_num_nodes = parent_2_config['encoding']['layers'][i]['num_nodes']
            config['encoding']['layers'].append({'num_nodes':
                                                     random.choice([parent_1_layer['num_nodes'], parent_2_num_nodes]),
                                                 'activation_fn': 'selu'})
            i += 1
        i = 0
        for parent_1_layer in parent_1_config['decoding']['layers']:
            parent_2_num_nodes = parent_2_config['decoding']['layers'][i]['num_nodes']
            config['decoding']['layers'].append({'num_nodes':
                                                     random.choice([parent_1_layer['num_nodes'], parent_2_num_nodes]),
                                                 'activation_fn': 'selu'})
            i += 1
        config['optimiser']['name'] = random.choice([parent_1_config['optimiser']['name'],
                                                     parent_2_config['optimiser']['name']])

        return config

    def mutate(self, config: dict) -> dict:
        """
        Randomly mutate a field in the configuration.

        Parameters
        ----------
        config (dict):      the dictionary containing the parameters to mutate.

        Returns
        -------
        Dict with a parameter mutated.
        """
        # Choose a random field i.e. latent_nodes, encoding, decoding

        mutation_field = random.choice(['num_nodes', 'encoding', 'decoding', 'optimiser'])

        if mutation_field == 'num_nodes':
            # Randomly choose again from the number of nodes
            config[mutation_field] = random.choice(self.bounds['latent_num_nodes'])
        elif mutation_field == 'optimiser':
            config[mutation_field]['name'] = random.choice(self.bounds['optimisers'])
        else:
            # Otherwise choose the layer to mutate
            if len(config[mutation_field]['layers']) > 0:
                # Might be a linear VAE then we don't mutate the encoding and decoding
                layer_idx = random.choice(range(0, len(config[mutation_field]['layers']), 1))
                config[mutation_field]['layers'][layer_idx]['num_nodes'] = random.choice(
                    self.bounds[mutation_field][layer_idx])

        return config

    def breed(self, parent_1: VAE, parent_2: VAE) -> list:
        """
        Generate children based on two parents. Here we randomly choose between the parent configs
        and mutate at a given rate.

        Parameters
        ----------
        parent_1 (VAE):     VAE that we use for choosing the configurations
        parent_2 (VAE):     VAE as above.

        Returns
        -------
        list:               (vae_child_1, vae_child_2)
        """
        children = []
        for _ in range(2):

            # Loop through the parameters and pick params for the kid.
            child_config = self.generate_config_from_parents(parent_1, parent_2)

            # Randomly mutate some of the children (i.e. change one parameter).
            if self.mutate_chance > random.random():
                child_config = self.mutate(child_config)

            # Now create a network object.
            child_vae = VAE(self.data_np, self.data_np, self.labels, child_config)
            children.append(child_vae)

        return children

    def evolve(self, population):
        """
        Evolve a population of VAEs.

        Parameters
        ----------
        population

        Returns
        -------

        """
        # Get scores for each network.
        graded = [(self.fitness(vae), vae) for vae in population]

        # Sort on the scores.
        graded = [x[1] for x in sorted(graded, key=lambda x: x[0])] # , reverse=True For descending

        # Get the number we want to keep for the next gen.

        retain_length = int(len(graded) * self.retain)
        if retain_length < 2:
            # i.e. we have reached the end of our evolution
            return graded
        # The parents are every network we want to keep.
        parents = graded[:retain_length]

        # For those we aren't keeping, randomly keep some anyway.
        for individual in graded[retain_length:]:
            if self.random_select > random.random():
                parents.append(individual)

        # Now find out how many spots we have left to fill.
        parents_length = len(parents)
        desired_length = len(population) - parents_length
        children = []

        # Add children, which are bred from two remaining networks.
        while len(children) < desired_length:

            # Get a random mom and dad.
            parent_1 = random.randint(0, parents_length - 1)
            parent_2 = random.randint(0, parents_length - 1)

            # Assuming they aren't the same network...
            if parent_1 != parent_2:
                parent_1 = parents[parent_1]
                parent_2 = parents[parent_2]

                # Breed them.
                new_children = self.breed(parent_1, parent_2)

                # Add the children one at a time.
                for new_child in new_children:
                    # Don't grow larger than desired length.
                    if len(children) < desired_length:
                        children.append(new_child)

        parents.extend(children)

        return parents

    def train_vaes(self, vaes: list) -> None:
        """
        Train the VAEs.

        Parameters
        ----------
        vaes

        Returns
        -------

        """
        for vae in vaes:
            vae.encode(epochs=self.epochs, batch_size=self.batch_size, method='default')

    def get_average_goodness(self, vaes: list):
        """

        Parameters
        ----------
        vaes (list):        list of variational autoencoders.

        Returns
        -------
        average goodness for a given population.
        """
        total_accuracy = 0
        for vae in vaes:
            total_accuracy += self.fitness(vae)

        return total_accuracy / len(vaes)
