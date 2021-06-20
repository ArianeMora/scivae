.. _references:

References
==========

Code
----

sci-vae is designed for Python >=3.6 and requires the following libraries, which will be automatically installed:

.. list-table::
   :widths: 15 15 50
   :header-rows: 1

   * - Library
     - Version
     - Reference
   * - `keras <https://keras.io/>`_
     - >= 2.4.0
     - @misc{chollet2015keras, title={Keras}, author={Chollet, Fran\c{c}ois and others}, year={2015}, howpublished={\url{https://keras.io}},}
   * - `Tensorflow <https://www.tensorflow.org/>`_ and `tensorflow-probability <https://www.tensorflow.org/probability/>`_
     - >= 2.1.0
     - Abadi, M., Barham, P., Chen, J., Chen, Z., Davis, A., Dean, J., Devin, M., Ghemawat, S., Irving, G., Isard, M., Kudlur, M., Levenberg, J., Monga, R., Moore, S., Murray, D. G., Steiner, B., Tucker, P., Vasudevan, V., Warden, P., … Zheng, X. (2016). TensorFlow: A system for large-scale machine learning. 12th USENIX Symposium on Operating Systems Design and Implementation (OSDI 16), 265–283. https://www.usenix.org/system/files/conference/osdi16/osdi16-abadi.pdf
   * - `NumPy <https://numpy.org/>`_
     - >= 1.9.0
     - Harris, C.R., Millman, K.J., van der Walt, S.J. et al. Array programming with NumPy. Nature 585, 357–362 (2020). DOI: `0.1038/s41586-020-2649-2 <https://doi.org/10.1038/s41586-020-2649-2>`_
   * - `pandas <https://pandas.pydata.org/>`_
     - >= 1.0.3
     - .. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3715232.svg
          :target: https://doi.org/10.5281/zenodo.3715232
   * - `scikit-learn <https://scikit-learn.org/stable/about.html>`_
     - >= 0.22.2
     - Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, É. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12(85), 2825–2830.
   * - `pyswarm <https://pythonhosted.org/pyswarm/>`_
     - >= 0.6
     - https://github.com/tisimst/pyswarm
   * - `matplotlib <https://matplotlib.org/3.3.3/>`_
     - >= 3.3.3
     - .. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3264781.svg
          :target: https://doi.org/10.5281/zenodo.3264781
   * - `seaborn <https://seaborn.pydata.org/>`_
     - >= 0.11.1
     - .. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.4379347.svg
           :target: https://doi.org/10.5281/zenodo.4379347

**We strongly encourage you to cite the other python packages**. Note *pyswarm* and *scikit-learn* are only used when
using the *optimiser* component (not your standard VAE) so these can be omitted if you don't use them. Similarly,
matplotlib and seaborn may not necessarily be used, so it depends on what aspects of this project is used in your work.

Literature
----------
.. list-table::
   :widths: 15 50
   :header-rows: 1

   * - Label
     - Reference
   * - VAE paper
     - Kingma, Diederik P, and Max Welling. 2014. “Auto-Encoding Variational Bayes.” In International Conference on Learning Representations.
   * - Info VAE
     - Zhao, S., Song, J., & Ermon, S. (2018). InfoVAE: Information Maximizing Variational Autoencoders. ArXiv:1706.02262 [Cs, Stat]. http://arxiv.org/abs/1706.02262
   * - VAEs applied to cancer
     - Simidjievski, N., Bodnar, C., Tariq, I., Scherer, P., Andres Terre, H., Shams, Z., Jamnik, M., & Liò, P. (2019). Variational Autoencoders for Cancer Data Integration: Design Principles and Computational Practice. Frontiers in Genetics, 10. https://doi.org/10.3389/fgene.2019.01205


Blogs and Githubs used
----------------------

.. list-table::
   :widths: 100
   :header-rows: 1

   * - Link
   * - `MMD in Keras <https://github.com/pren1/keras-MMD-Variational-Autoencoder/blob/master/Keras_MMD_Variational_Autoencoder.ipynb>`_
   * - `X-VAE Keras <https://github.com/s-omranpour/X-VAE-keras/blob/master/VAE/VAE_MMD.ipynb>`_
   * - `MMD VAE in Tensorflow <https://github.com/ShengjiaZhao/MMD-Variational-Autoencoder>`_
   * - `X-VAE Keras <https://github.com/s-omranpour/X-VAE-keras/blob/master/VAE/VAE_MMD.ipynb>`_
   * - `VAE applied to cancer data <https://github.com/CancerAI-CL/IntegrativeVAEs/blob/master/code/models/mmvae.py>`_
   * - `Blog by Ermon group re. InfoVAE <https://ermongroup.github.io/blog/a-tutorial-on-mmd-variational-autoencoders/>`_


Data
----

.. list-table::
   :widths: 15 15 50
   :header-rows: 1

   * - Name
     - Link
     - Reference
   * - `Encode <https://www.encodeproject.org/>`_
     - https://www.encodeproject.org/help/citing-encode/
     - ENCODE Project Consortium. (2012). An integrated encyclopedia of DNA elements in the human genome. Nature, 489(7414), 57–74. https://doi.org/10.1038/nature11247
   * - `MNIST <http://yann.lecun.com/exdb/mnist/>`_
     - http://yann.lecun.com/exdb/mnist/
     - Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based learning applied to document recognition." Proceedings of the IEEE, 86(11):2278-2324, November 1998. [on-line version]
   * - `Iris dataset <https://en.wikipedia.org/wiki/Iris_flower_data_set>`_
     - https://archive.ics.uci.edu/ml/datasets/Iris/
     - Fisher,R.A. "The use of multiple measurements in taxonomic problems" Annual Eugenics, 7, Part II, 179-188 (1936); Edgar Anderson (1936). "The species problem in Iris". Annals of the Missouri Botanical Garden. 23 (3): 457–509. doi:10.2307/2394164


Environments
------------
This will be user dependent but don't forget to cite anaconda if you use it or `reticulate <https://rstudio.github.io/reticulate/>`_ if you use R.
