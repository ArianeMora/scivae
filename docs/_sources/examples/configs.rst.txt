Config examples
===============

Examples for input files & in depth description of parameters for configuring the VAE.

The primary input for the VAE is the configuration, which is in the form of either a json string, or as a json file. We
recommend using a json file, as it keeps things more organised but we allow for a json string anyway. Here we show
some examples for both cases.

Things to note: JSON format requires "double quotes" around strings!

Single layer VAE
----------------
Example for single layer VAE below. Here we have a VAE that uses MMD for the latent space  with mean square error loss
for the reconstruction loss. We have two latent nodes, and no internal layers (apart from the central one. We use adam
optimiser with the default parameters (learning rate:  0.01, beta 1: 0.9 and, beta 2: 0.999.

This VAE will work as so: input --> 2 nodes --> output size.

JSON file:

.. code-block:: json

   {"loss":
    {"loss_type": "mse",
      "distance_metric": "mmd",
      "mmd_weight": 1.0
    },
      "encoding": {
        "layers": []
      },
      "decoding": {
        "layers": []
      },
      "latent": {
        "num_nodes": 2
      },
      "optimiser": {
        "params": {}, "name": "adam"
      }
    }

JSON string:

.. code-block:: bash

    '{"loss": {"loss_type": "mse", "distance_metric": "mmd", "mmd_weight": 1.0}, "encoding": {"layers": []}, "decoding": {"layers": []}, "latent": {"num_nodes": 2}, "optimiser": {"params": {}, "name": "adam"}}'


Multi layer VAE
---------------
Example for mutli-layer VAE below. As above except with multiple layers (both in encoding and decoding, we have one
encoding layer, and two decoding layers, and 5 latent nodes. Here we use cross entropy as opposed to MSE and KL divergence
as opposed to MMD. We also set specific parameters for the optimiser.

For each layer you need to specify the number of node and the activation function (e.g. selu, relu, etc --> this should
work with any of the keras activation strings.

This VAE will work as so: input --> 64 nodes --> 5 nodes --> 32 nodes --> 64 nodes --> output size.

JSON file:

.. code-block:: json

   {"loss":
    {"loss_type": "ce",
      "distance_metric": "kl"
    },
      "encoding": {
        "layers": [
                    {"num_nodes": 64, "activation_fn": "selu"}
            ]
      },
      "decoding": {
        "layers": [
                    {"num_nodes": 32, "activation_fn": "relu"},
                    {"num_nodes": 64, "activation_fn": "selu"}
            ]
      },
      "latent": {
        "num_nodes": 5
      },
      "optimiser": {
        "params": {"learning_rate": 0.001, "beta_1": 0.8, "beta_2": 0.97},
        "name": "adamax"
      }
    }

JSON string:

.. code-block:: bash
   '{"loss": {"loss_type": "ce", "distance_metric": "kl"}, "encoding": {"layers": [{"num_nodes": 64, "activation_fn": "selu"}]}, "decoding": {"layers": [{"num_nodes": 32, "activation_fn": "relu"},{"num_nodes": 64, "activation_fn": "selu"}]}, "latent": {"num_nodes": 5}, "optimiser": {"params": {"learning_rate": 0.001, "beta_1": 0.8, "beta_2": 0.97}, "name": "adamax"}}'




Multi loss & multi layer VAE
----------------------------

Here we have multiple layers and also multiple loss functions (i.e. we want to integrate binary and continuous data).
FYI I have no idea what the mathematical implications are of joining these! Intuitively I assume it's not going to be
great so use at your own risk!!! If you just want to tie the different outputs (i.e. use MSE for both outputs then I
don't think there is too much of an issue **"multi_loss": ["mse", "mse"],**).

To enable use to use a multiple loss, we need to define which of the input columns will be tied to each loss. We only
allow the columns to be sequential (defined in **multi_sizes**), i.e. in this case we have in input of 60 columns, the first 40 will be "tied together"
in the last layer, and the last 20 will be "tied together". i.e. the VAE will look like:

.. code-block:: python
    40 ---> 10                  ---> 16 ---> 40
                ---> 5 ---> 10
    20 ---> 10                  ---> 16 ---> 20

JSON file:

.. code-block:: json

   {"loss":
    {"loss_type": "multi",
      "distance_metric": "mmd",
      "mmd_weight": 1.0,
      "multi_loss": ["mse", "mse"]
    },
      "encoding": {
        "layers": [
                    {"num_nodes": 10, "activation_fn": "selu"}
            ]
      },
      "decoding": {
        "layers": [
                    {"num_nodes": 10, "activation_fn": "relu"},
                    {"num_nodes": 16, "activation_fn": "selu"}
            ]
      },
      "latent": {
        "num_nodes": 5
      },
      "optimiser": {
        "params": {"learning_rate": 0.001, "beta_1": 0.8, "beta_2": 0.97},
        "name": "adamax"
      },
      "multi_sizes": [40, 20]
    }

.. code-block:: bash
   '{"loss": {"loss_type": "multi", "distance_metric": "mmd", "mmd_weight": 1.0, "multi_loss": ["mse", "mse"]}, "encoding": {"layers": [{"num_nodes": 10, "activation_fn": "selu"}]}, "decoding": {"layers": [{"num_nodes": 10, "activation_fn": "relu"},{"num_nodes": 16, "activation_fn": "selu"}]}, "latent": {"num_nodes": 5}, "optimiser": {"params": {"learning_rate": 0.001, "beta_1": 0.8, "beta_2": 0.97}, "name": "adamax"}, "multi_sizes": [40, 20]}'
