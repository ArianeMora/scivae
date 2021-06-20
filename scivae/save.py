def load_attributes_from_hdf5_group(group, name):
  """Loads attributes of the specified name from the HDF5 group.

  This method deals with an inherent problem
  of HDF5 file which is not able to store
  data larger than HDF5_OBJECT_HEADER_LIMIT bytes.

  Arguments:
      group: A pointer to a HDF5 group.
      name: A name of the attributes to load.

  Returns:
      data: Attributes data.
  """
  if name in group.attrs:
    data = [n.decode('utf8') for n in group.attrs[name]]
  else:
    data = []
    chunk_id = 0
    while '%s%d' % (name, chunk_id) in group.attrs:
      data.extend(
          [n.decode('utf8') for n in group.attrs['%s%d' % (name, chunk_id)]])
      chunk_id += 1
  return data


def _legacy_weights(layer):
  """DO NOT USE.

  For legacy reason, the layer.weights was in the order of
  [self.trainable_weights + self.non_trainable_weights], and this order was
  used for preserving the weights in h5 format. The new order of layer.weights
  are the same as layer.get_weights() which is more intuitive for user. To
  keep supporting the existing saved h5 file, this method should be used to
  save/load weights. In future version, we will delete this method and
  introduce a breaking change for h5 and stay with the new order for weights.

  Args:
    layer: a `tf.keras.Model` or `tf.keras.layers.Layer` instance.

  Returns:
    A list of variables with the order of trainable_weights, followed by
      non_trainable_weights.
  """
  weights = layer.trainable_weights + layer.non_trainable_weights
  if any(not isinstance(w, variables_module.Variable) for w in weights):
    raise NotImplementedError(
        'Save or restore weights that is not an instance of `tf.Variable` is '
        'not supported in h5, use `save_format=\'tf\'` instead. Got a model '
        'or layer {} with weights {}'.format(layer.__class__.__name__, weights))
  return weights


def load_weights_from_hdf5_group(f, layers):
  """Implements topological (order-based) weight loading.

  Arguments:
      f: A pointer to a HDF5 group.
      layers: a list of target layers.

  Raises:
      ValueError: in case of mismatch between provided layers
          and weights file.
  """
  if 'keras_version' in f.attrs:
    original_keras_version = f.attrs['keras_version'].decode('utf8')
  else:
    original_keras_version = '1'
  if 'backend' in f.attrs:
    original_backend = f.attrs['backend'].decode('utf8')
  else:
    original_backend = None

  filtered_layers = []
  for layer in layers:
    weights = _legacy_weights(layer)
    if weights:
      filtered_layers.append(layer)

  layer_names = load_attributes_from_hdf5_group(f, 'layer_names')
  filtered_layer_names = []
  for name in layer_names:
    g = f[name]
    weight_names = load_attributes_from_hdf5_group(g, 'weight_names')
    if weight_names:
      filtered_layer_names.append(name)
  layer_names = filtered_layer_names
  if len(layer_names) != len(filtered_layers):
    raise ValueError('You are trying to load a weight file '
                     'containing ' + str(len(layer_names)) +
                     ' layers into a model with ' + str(len(filtered_layers)) +
                     ' layers.')

  # We batch weight value assignments in a single backend call
  # which provides a speedup in TensorFlow.
  weight_value_tuples = []
  for k, name in enumerate(layer_names):
    g = f[name]
    weight_names = load_attributes_from_hdf5_group(g, 'weight_names')
    weight_values = [np.asarray(g[weight_name]) for weight_name in weight_names]
    layer = filtered_layers[k]
    symbolic_weights = _legacy_weights(layer)
    weight_values = preprocess_weights_for_loading(
        layer, weight_values, original_keras_version, original_backend)
    if len(weight_values) != len(symbolic_weights):
      raise ValueError('Layer #' + str(k) + ' (named "' + layer.name +
                       '" in the current model) was found to '
                       'correspond to layer ' + name + ' in the save file. '
                       'However the new layer ' + layer.name + ' expects ' +
                       str(len(symbolic_weights)) +
                       ' weights, but the saved weights have ' +
                       str(len(weight_values)) + ' elements.')
    weight_value_tuples += zip(symbolic_weights, weight_values)
  K.batch_set_value(weight_value_tuples)


def save_optimizer_weights_to_hdf5_group(hdf5_group, optimizer):
  """Saves optimizer weights of a optimizer to a HDF5 group.

  Arguments:
      hdf5_group: HDF5 group.
      optimizer: optimizer instance.
  """

  symbolic_weights = getattr(optimizer, 'weights')
  if symbolic_weights:
    weights_group = hdf5_group.create_group('optimizer_weights')
    weight_names = [str(w.name).encode('utf8') for w in symbolic_weights]
    save_attributes_to_hdf5_group(weights_group, 'weight_names', weight_names)
    weight_values = K.batch_get_value(symbolic_weights)
    for name, val in zip(weight_names, weight_values):
      param_dset = weights_group.create_dataset(
          name, val.shape, dtype=val.dtype)
      if not val.shape:
        # scalar
        param_dset[()] = val
      else:
        param_dset[:] = val


def load_optimizer_weights_from_hdf5_group(hdf5_group):
  """Load optimizer weights from a HDF5 group.

  Arguments:
      hdf5_group: A pointer to a HDF5 group.

  Returns:
      data: List of optimizer weight names.
  """
  weights_group = hdf5_group['optimizer_weights']
  optimizer_weight_names = load_attributes_from_hdf5_group(
      weights_group, 'weight_names')
  return [weights_group[weight_name] for weight_name in optimizer_weight_names]


def save_attributes_to_hdf5_group(group, name, data):
  """Saves attributes (data) of the specified name into the HDF5 group.

  This method deals with an inherent problem of HDF5 file which is not
  able to store data larger than HDF5_OBJECT_HEADER_LIMIT bytes.

  Arguments:
      group: A pointer to a HDF5 group.
      name: A name of the attributes to save.
      data: Attributes data to store.

  Raises:
    RuntimeError: If any single attribute is too large to be saved.
  """
  # Check that no item in `data` is larger than `HDF5_OBJECT_HEADER_LIMIT`
  # because in that case even chunking the array would not make the saving
  # possible.
  bad_attributes = [x for x in data if len(x) > HDF5_OBJECT_HEADER_LIMIT]

  # Expecting this to never be true.
  if bad_attributes:
    raise RuntimeError('The following attributes cannot be saved to HDF5 '
                       'file because they are larger than %d bytes: %s' %
                       (HDF5_OBJECT_HEADER_LIMIT, ', '.join(bad_attributes)))

  data_npy = np.asarray(data)

  num_chunks = 1
  chunked_data = np.array_split(data_npy, num_chunks)

  # This will never loop forever thanks to the test above.
  while any(x.nbytes > HDF5_OBJECT_HEADER_LIMIT for x in chunked_data):
    num_chunks += 1
    chunked_data = np.array_split(data_npy, num_chunks)

  if num_chunks > 1:
    for chunk_id, chunk_data in enumerate(chunked_data):
      group.attrs['%s%d' % (name, chunk_id)] = chunk_data
  else:
    group.attrs[name] = data

