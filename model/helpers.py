from keras import backend as K


# euclidean distance for contrastive loss
def euclidean_dist(vectors):
  x, y = vectors
  sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
  return K.sqrt(K.maximum(sum_square, K.epsilon()))

# returns output shape for euclidiean distance
def euclidean_dist_output_shape(shapes):
  shape1, _ = shapes
  return (shape1[0], 1)

# contrastive loss function for one-shot learning
def contrastive_loss(y_actual, y_pred):
  margin = 1
  square_pred = K.square(y_pred)
  margin_square = K.square(K.maximum(margin - y_pred, 0))
  return K.mean(y_actual * square_pred + (1 - y_actual) * margin_square)

# classification using contrastive loss iwth fixed threshold for distance
def accuracy(y_actual, y_pred):
  return K.mean(K.equal(y_actual, K.cast(y_pred < 0.5, y_actual.dtype)))
