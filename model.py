import tensorflow as tf
from tensorflow.keras.layers import Conv3DTranspose, Conv3D, BatchNormalization



def downsample(filters, size, apply_batchnorm=True):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())

  result.add(tf.keras.layers.LeakyReLU())

  return result

def upsample(filters, size, apply_dropout=False):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

  result.add(tf.keras.layers.BatchNormalization())

  if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.5))

  result.add(tf.keras.layers.ReLU())

  return result

def initializer():
  return tf.random_normal_initializer(0., 0.02)

def maxPooling2D(filters):
  model = tf.keras.Sequential()
  model.add(tf.keras.layers.MaxPooling2D(pool_size=(filters,filters),
   strides=(2, 2), padding='valid'))
  return model


def global_max_pooling():
  model = tf.keras.Sequential()
  model.add(tf.keras.layers.GlobalAveragePooling2D())
  return model


def dense_layer(size, flatten = False):  
  model = tf.keras.Sequential()
  if (flatten):
    model.add(tf.keras.layers.Flatten())
  model.add(tf.keras.layers.Dense(size, activation='relu'))
  return model




def Generator():
    inputs = tf.keras.layers.Input(shape=(128, 128, 1))

    #upsampling
    x = inputs

    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1',kernel_initializer=initializer())(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2',kernel_initializer=initializer())(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1',kernel_initializer=initializer())(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2',kernel_initializer=initializer())(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1',kernel_initializer=initializer())(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2',kernel_initializer=initializer())(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3',kernel_initializer=initializer())(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1',kernel_initializer=initializer())(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2',kernel_initializer=initializer())(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3',kernel_initializer=initializer())(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1',kernel_initializer=initializer())(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2',kernel_initializer=initializer())(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3',kernel_initializer=initializer())(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # Flatten and FC layers
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(8192, activation='relu', name='fc1')(x)
    x = tf.keras.layers.Dense(4096, activation='relu', name='fc2')(x)


    x = tf.keras.layers.Reshape((1, 1, 1, 4096))(x)
    x = tf.reshape(x, [-1, 1, 1, 1, 4096])

    # Decoder
    x = Conv3DTranspose(filters=64, kernel_size=(3, 3, 3), activation='sigmoid',strides=(2, 2, 2), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = Conv3DTranspose(filters=64, kernel_size=(3, 3, 3), activation='sigmoid',strides=(2, 2, 2), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = Conv3DTranspose(filters=64, kernel_size=(3, 3, 3), activation='sigmoid',strides=(2, 2, 2), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = Conv3DTranspose(filters=32, kernel_size=(3, 3, 3), activation='sigmoid',strides=(2, 2, 2), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = Conv3DTranspose(filters=32, kernel_size=(3, 3, 3), activation='sigmoid',strides=(2, 2, 2), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = Conv3DTranspose(filters=24, kernel_size=(3, 3, 3), activation='sigmoid',strides=(1, 1, 1), padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = Conv3D(filters=1, kernel_size=(3, 3, 3), strides=(1, 1, 1), padding='same')(x)


    return tf.keras.Model(inputs=inputs, outputs=x)