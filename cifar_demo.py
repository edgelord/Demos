### Mostly borrowed from tflearn's examples
import tensorflow as tf
import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import global_avg_pool
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from tflearn import batch_normalization
from tensorflow import nn
slim = tf.contrib.slim


# Data loading and preprocessing
from tflearn.datasets import cifar10
import tensorflow as tf
(X, Y), (X_test, Y_test) = cifar10.load_data()
X, Y = shuffle(X, Y)
Y = to_categorical(Y, 10)
Y_test = to_categorical(Y_test, 10)

# Real-time data preprocessing
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

# Real-time data augmentation
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
# img_aug.add_random_rotation(max_angle=25.)
img_aug.add_random_crop((32,32),4)
# Convolutional network building

# from fractal_block import tensor_shape, fractal_block
filter_size = 3
def make_network():
    net = input_data(shape=[None, 32, 32, 3],
                         data_preprocessing=img_prep,
                         data_augmentation=img_aug)

    with slim.arg_scope([slim.conv2d], padding='SAME', normalizer_fn = slim.batch_norm,
                        activation_fn=nn.relu, kernel_size=3,
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                        weights_regularizer=slim.l2_regularizer(0.0005)):

        for block, filters in enumerate([32,64,128]):
            net = slim.conv2d(net, filters)
            net = slim.conv2d(net, filters) + net
            net = slim.max_pool2d(net, 2)


        net = slim.conv2d(net, 256)
        net = slim.conv2d(net, 512)
        net = slim.conv2d(net, 10)
    net = global_avg_pool(net)
    return net

net = make_network()
# net = fully_connected(net, 10, activation='softmax')
net = regression(net, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)

# Train using classifier
      
model = tflearn.DNN(net, tensorboard_verbose=0,
                    checkpoint_path="models/baseline",
                    best_checkpoint_path="models/bestbaseline"
)
model.fit(X, Y, n_epoch=500, shuffle=True, validation_set=(X_test, Y_test),
          show_metric=True, batch_size=512, run_id='cifar10')
