*******
sci-VAE
*******


Sci-VAE is an implementation of a variational autoencoder in Keras that was developed to use VAEs to integrate
biological data. The implementation allows for customisations to the VAE to be passed in via CLI (and a JSON file) or
in python and R scripts (see examples).

The VAE implementation expects a data matrix with features as columns (no headers) and rows as training data (no row IDs).
The first thing the VAE will do is transform your data between 0 and 1 so you don't need to do this prior to running the
VAE.

I show several examples, using MNIST, IRIS dataset and then also a publicly available histone modification
data from encode.

Saving has been implemented of the VAE state so that you can re-use your trained model on the same data and get
the same latent space (or use the trained VAE on new data).


There are also some useful visualisations that I was having to repeat often when inspecting the latent space so check
out the Vis functions if you're interested (these are also in the examples).

Lastly, there is a optimisation library that allows you to optimise the VAE architecture based on building a separable
latent space based on classification. If you choose to use this you'll also need to pass in *labels* into the VAE. Check
out some tests for how to run this - it uses an evolutionary algorithm.


Running sci-vae
===============

1. Install sci-vae (:ref:`Installing <installing>`)

2. View R examples in (:ref:`examples <examples/r_example>`)

3. Look at a python example notebook with output (:ref:`notebook <examples/python_example>`)

4. Details about json config (:ref:`cli <examples/configs>`)

Extending sci-vae
=================

1. Make a pull request on github


Citing sci-vae
===================
Sci-vae can be cited as in :ref:`references`, where we also provide citations for the used tools (e.g. numpy).

.. toctree::
   :caption: Getting started
   :maxdepth: 1

   about
   installing/index


.. toctree::
   :caption: Running sci-vae
   :maxdepth: 1

   examples/r_example
   examples/python_example
   examples/configs


.. toctree::
   :caption: About
   :maxdepth: 1

   faq
   changelog
   references
