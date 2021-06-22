# scivae

Check out our docs: https://arianemora.github.io/scivae/  

If you use this please cite: https://doi.org/10.1101/2021.06.22.449386

scivae is a wrapper around the keras AE that allows you to build/save/visualise with a variational autoencoder.

Blogs & notebooks used as references are noted in the code and a couple at the end of this README.

The primary difference between a VAE and a normal AE is in how the loss function is computed. Here the loss 
has been abstracted out to the loss class (in *loss.py*) where we can use a number of loss metrics MMD, KL and combine this with 
MSE or COR loss.

The VAE (in *vae.py*) class has the general VAE structure.

Saving has been implemented of the VAE state so that you can re-use your trained model on the same data and get 
the same latent space (or use the trained VAE on new data).  

Optimiser was a temporary deviation where we can pass in a VAE structure and using an evolutionary algorithm the 
optimisation class will try to get the best VAE structure. This will be returned.

Validate allows for running simple validations using scikitlearn i.e. if your primary interest is to get a meaningful
 latent space that captures the key features of the dataset, it can be good to compare how much "information" has 
 been captured between your classes. A good way of measuring this is by passing through the latent space and a set 
 of labels and seeing if a simple classifier can distingush your classes better than with the raw data.

## Users

Check out the install page and the documentation!

## Developers
Install required packages
```
pip install -r requirements.txt
```

It is very easy to call the basic VAE. Simply install the package (or raw code). Then you need to setup 
a config dictionary. This is pretty self explanatory. 

```
from scivae import *
```

    - loss: loss dictionary see Loss class for input details
    - encoding: a dictionary of encoding layers, number of nodes and activation function
    - decoding: same as above but for decoding layers
    - latent: configs for latent space. See (def optimiser(self, optimiser_name: str, params: dict):) in vae.py for details

```

config_mse = {'loss': {'loss_type': 'mse', 'distance_metric': 'mmd', 'mmd_weight': 1}, 
          'encoding': {'layers': [{'num_nodes': 32, 'activation_fn': 'selu'}, 
                                  {'num_nodes': 16, 'activation_fn': 'selu'}]}, 
          'decoding': {'layers': [{'num_nodes': 16, 'activation_fn': 'selu'}, 
                                  {'num_nodes': 32, 'activation_fn': 'selu'}]}, 
 'latent': {'num_nodes': 2}, 'optimiser': {'params': {}, 'name': 'adam'}}

```

Run the VAE. Numeric data is expected to follow an approximately normal distribution (each column).
It expects a numpy array with each row being a list of features corresponding to some label. Labels mean nothing - 
they just need to be a list of the same size - these are just used for downstream analyses (e.g. colouring).

Note for most configs we want input_data = output_data however I have left this modular so we can upgrade to having 
it be denoising etc in the future.
```
# vae_mse = VAE(input_data, output_data, labels, config_mse, 'vae_label')
vae_mse = VAE(numeric_data, numeric_data, labels, config_mse, 'vae_label')
# Set batch size and number of epochs
vae_mse.encode('default', epochs=500, bacth_size=50)
encoded_data_vae_mse = vae_mse.get_encoded_data()
``` 
The VAE can also be used to encode new data.
```
# Note the [0] on the end - the encoded data is just returned as the 0th element of a np array.
new_data_encoded = vae_mse.encode_new_data(some_new_np_array)[0]
```

Visualisation is the same as if we got back the PCs from PCA. i.e. the below code will plot a scatter plot of the first 
and second latent nodes.

```
plt.scatter(encoded_data_vae_mse[:,0], encoded_data_vae_mse[:,1])
```

## Tests
See tests for further examples.

## Creating a package for scivae

If you don't already have twine installed run the following:
`python -m pip install --user --upgrade twine`

## The following will create the package
```
python setup.py sdist bdist_wheel
twine check dist/PROJECT_NAME.tar.gz
```

## Install to python environment localling

`pip install PATH_TO_PROJECT/dist/PROJECT_NAME.tar.gz`
You should run this before uploading it and check all works as expected.

## The following will push the package to pip 
**Note you need to set up a pip account first**

```
twine upload dist/*
```

## Have a look at your projects page on pip

`https://pypi.org/project/PROJECT_NAME/`


## References
        https://github.com/pren1/keras-MMD-Variational-Autoencoder/blob/master/Keras_MMD_Variational_Autoencoder.ipynb
        https://github.com/s-omranpour/X-VAE-keras/blob/master/VAE/VAE_MMD.ipynb
        https://github.com/ShengjiaZhao/MMD-Variational-Autoencoder/blob/master/mmd_vae.py
        https://github.com/CancerAI-CL/IntegrativeVAEs/blob/master/code/models/mmvae.py