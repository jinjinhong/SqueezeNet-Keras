import cv2
import numpy as np
from keras.models import Graph
from keras.optimizers import Adam
from sklearn.preprocessing import OneHotEncoder
from keras.layers import Activation, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten

# For those of you who just look for a Keras implementation of SqueezeNet,
# just drop the data preparation part, change the output dim of the model,
# customize your solver and train the model!

print "Preparing data"

# CIFAR-10 is used to train and test the model
# for details of the CIFAR-10 dataset, please refer to 
# http://www.cs.toronto.edu/~kriz/cifar.html

def unpickle(filename):
    import cPickle
    f = open(filename, 'rb')
    dict = cPickle.load(f)
    f.close()
    return dict

filenames = ['data_batch_' + str(i) for i in range(1, 6)]
batches = [unpickle(filename) for filename in filenames]

# Note that each data point in CIFAR-10 is a flattened vector of dim 3072;
# each image is first flattened in a single channel and then concatenated 
# to form a larger vector. However, Squeeze net requires that the input to 
# be of 3 x 227 x 227. Therefore, OpenCV function is invoked to resize each
# image, which is flattened in the same way as that in CIFAR-10 to preserve
# consistency.

temp = np.zeros((32, 32, 3), dtype = 'uint8')
temp_ = np.zeros((3, 227, 227), dtype = 'uint8')

x_train = np.zeros((50000, 3, 227, 227), dtype = 'uint8')
X = np.vstack([batch['data'] for batch in batches])
y_train = np.hstack([batch['labels'] for batch in batches])

# Build the training and test sets in the same way as CIFAR-10

for i in xrange(X.shape[0]):
    img = X[i, :]
    img = np.reshape(img, (3, 32, 32))
    temp[:, :, 0] = img[0, :, :]
    temp[:, :, 1] = img[1, :, :]
    temp[:, :, 2] = img[2, :, :]
    img = cv2.resize(temp, (227, 227))
    temp_[0, :, :] = img[:, :, 0]
    temp_[1, :, :] = img[:, :, 1]
    temp_[2, :, :] = img[:, :, 2]
    x_train[i, :, :, :] = temp_

batch = unpickle('test_batch')
X = batch['data']
y_test = batch['labels']
x_test = np.zeros((10000, 3, 227, 227), dtype = 'uint8')

for i in xrange(X.shape[0]):
    img = X[i, :]
    img = np.reshape(img, (3, 32, 32))
    temp[:, :, 0] = img[0, :, :]
    temp[:, :, 1] = img[1, :, :]
    temp[:, :, 2] = img[2, :, :]
    img = cv2.resize(temp, (227, 227))
    temp_[0, :, :] = img[:, :, 0]
    temp_[1, :, :] = img[:, :, 1]
    temp_[2, :, :] = img[:, :, 2]
    x_test[i, :, :, :] = temp_

# Note that y_test is a list in the original pickled file

y_test = np.array(y_test)

# To feed the Keras model, labels are encoded in one-hot representation

enc = OneHotEncoder()
enc.fit(y_train.reshape(-1, 1))
y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
y_test = enc.transform(y_test.reshape(-1, 1)).toarray()

print "Data prepared"

print "Building the model"

# Build the SqueezeNet model. Since Keras does not naively support filters
# of different shapes, Graph model, rather than Sequential model is used to
# get around this problem.
# The model architecture is detailed in arXiv 1602.07360
# http://arxiv.org/pdf/1602.07360v2.pdf

graph = Graph()
graph.add_input(name = 'input', input_shape = (3, 227, 227))
graph.add_node(
    Convolution2D(96, 7, 7, activation = 'relu', subsample = (2, 2)), 
    name = 'conv1', 
    input = 'input'
    )

graph.add_node(
    MaxPooling2D(pool_size = (3, 3), strides = (2, 2)), 
    name = 'maxpool1', 
    input = 'conv1'
    )

# The Fire module is implemented as follows. Please note the axis to be concatenated and the graph structure.

graph.add_node(
    Convolution2D(16, 1, 1, activation = 'relu'), 
    name = 'fire2_squeeze1x1', 
    input = 'maxpool1'
    )
graph.add_node(
    Convolution2D(64, 1, 1, activation = 'relu'), 
    name = 'fire2_expand1x1', 
    input = 'fire2_squeeze1x1'
    )
graph.add_node(
    ZeroPadding2D((1, 1)), 
    name = 'fire2_expand3x3_zeropad', 
    input = 'fire2_squeeze1x1'
    )
graph.add_node(
    Convolution2D(64, 3, 3, activation = 'relu'), 
    name = 'fire2_expand3x3', 
    input = 'fire2_expand3x3_zeropad'
    )

graph.add_node(
    Convolution2D(16, 1, 1, activation = 'relu'), 
    name = 'fire3_squeeze1x1', 
    inputs = ['fire2_expand1x1', 'fire2_expand3x3'], 
    merge_mode = 'concat', 
    concat_axis = 1
    )
graph.add_node(
    Convolution2D(64, 1, 1, activation = 'relu'), 
    name = 'fire3_expand1x1', 
    input = 'fire3_squeeze1x1'
    )
graph.add_node(
    ZeroPadding2D((1, 1)), 
    name = 'fire3_expand3x3_zeropad', 
    input = 'fire3_squeeze1x1'
    )
graph.add_node(
    Convolution2D(64, 3, 3, activation = 'relu'), 
    name = 'fire3_expand3x3', 
    input = 'fire3_expand3x3_zeropad'
    )

graph.add_node(
    Convolution2D(32, 1, 1, activation = 'relu'), 
    name = 'fire4_squeeze1x1', 
    inputs = ['fire3_expand1x1', 'fire3_expand3x3'], 
    merge_mode = 'concat', 
    concat_axis = 1
    )
graph.add_node(
    Convolution2D(128, 1, 1, activation = 'relu'), 
    name = 'fire4_expand1x1', 
    input = 'fire4_squeeze1x1'
    )
graph.add_node(
    ZeroPadding2D((1, 1)), 
    name = 'fire4_expand3x3_zeropad', 
    input = 'fire4_squeeze1x1'
    )
graph.add_node(Convolution2D(
    128, 3, 3, activation = 'relu'), 
    name = 'fire4_expand3x3', 
    input = 'fire4_expand3x3_zeropad'
    )

graph.add_node(
    MaxPooling2D(pool_size = (3, 3), strides = (2, 2)), 
    name = 'maxpool4', 
    inputs = ['fire4_expand1x1', 'fire4_expand3x3'], 
    merge_mode = 'concat', 
    concat_axis = 1
    )

graph.add_node(
    Convolution2D(32, 1, 1, activation = 'relu'), 
    name = 'fire5_squeeze1x1', 
    input = 'maxpool4'
    )
graph.add_node(
    Convolution2D(128, 1, 1, activation = 'relu'), 
    name = 'fire5_expand1x1', 
    input = 'fire5_squeeze1x1'
    )
graph.add_node(
    ZeroPadding2D((1, 1)), 
    name = 'fire5_expand3x3_zeropad', 
    input = 'fire5_squeeze1x1'
    )
graph.add_node(
    Convolution2D(128, 3, 3, activation = 'relu'), 
    name = 'fire5_expand3x3', 
    input = 'fire5_expand3x3_zeropad'
    )

graph.add_node(
    Convolution2D(48, 1, 1, activation = 'relu'), 
    name = 'fire6_squeeze1x1', 
    inputs = ['fire5_expand1x1', 'fire5_expand3x3'], 
    merge_mode = 'concat', 
    concat_axis = 1
    )
graph.add_node(
    Convolution2D(192, 1, 1, activation = 'relu'), 
    name = 'fire6_expand1x1', 
    input = 'fire6_squeeze1x1'
    )
graph.add_node(
    ZeroPadding2D((1, 1)), 
    name = 'fire6_expand3x3_zeropad', 
    input = 'fire6_squeeze1x1'
    )
graph.add_node(
    Convolution2D(192, 3, 3, activation = 'relu'), 
    name = 'fire6_expand3x3', 
    input = 'fire6_expand3x3_zeropad'
    )

graph.add_node(
    Convolution2D(48, 1, 1, activation = 'relu'), 
    name = 'fire7_squeeze1x1', 
    inputs = ['fire6_expand1x1', 'fire6_expand3x3'], 
    merge_mode = 'concat', 
    concat_axis = 1
    )
graph.add_node(
    Convolution2D(192, 1, 1, activation = 'relu'), 
    name = 'fire7_expand1x1', 
    input = 'fire7_squeeze1x1'
    )
graph.add_node(
    ZeroPadding2D((1, 1)), 
    name = 'fire7_expand3x3_zeropad', 
    input = 'fire7_squeeze1x1'
    )
graph.add_node(
    Convolution2D(192, 3, 3, activation = 'relu'), 
    name = 'fire7_expand3x3', 
    input = 'fire7_expand3x3_zeropad'
    )

graph.add_node(
    Convolution2D(64, 1, 1, activation = 'relu'), 
    name = 'fire8_squeeze1x1', 
    inputs = ['fire7_expand1x1', 'fire7_expand3x3'], 
    merge_mode = 'concat', 
    concat_axis = 1
    )
graph.add_node(
    Convolution2D(256, 1, 1, activation = 'relu'), 
    name = 'fire8_expand1x1', 
    input = 'fire8_squeeze1x1'
    )
graph.add_node(
    ZeroPadding2D((1, 1)), 
    name = 'fire8_expand3x3_zeropad', 
    input = 'fire8_squeeze1x1'
    )
graph.add_node(
    Convolution2D(256, 3, 3, activation = 'relu'), 
    name = 'fire8_expand3x3', 
    input = 'fire8_expand3x3_zeropad'
    )

graph.add_node(
    MaxPooling2D(pool_size = (3, 3), strides = (2, 2)), 
    name = 'maxpool8', 
    inputs = ['fire8_expand1x1', 'fire8_expand3x3'], 
    merge_mode = 'concat', 
    concat_axis = 1
    )

graph.add_node(
    Convolution2D(64, 1, 1, activation = 'relu'), 
    name = 'fire9_squeeze1x1', 
    input = 'maxpool8'
    )
graph.add_node(
    Convolution2D(256, 1, 1, activation = 'relu'), 
    name = 'fire9_expand1x1', 
    input = 'fire9_squeeze1x1'
    )
graph.add_node(
    ZeroPadding2D((1, 1)), 
    name = 'fire9_expand3x3_zeropad', 
    input = 'fire9_squeeze1x1'
    )
graph.add_node(
    Convolution2D(256, 3, 3, activation = 'relu'), 
    name = 'fire9_expand3x3', 
    input = 'fire9_expand3x3_zeropad'
    )

graph.add_node(
    Dropout(0.5), 
    name = 'dropout', 
    inputs = ['fire9_expand1x1', 'fire9_expand3x3'], 
    merge_mode = 'concat', 
    concat_axis = 1
    )

graph.add_node(
    Convolution2D(10, 1, 1, activation = 'relu'), 
    name = 'conv10', 
    input = 'dropout'
    )

graph.add_node(
    AveragePooling2D(pool_size = (13, 13)), 
    name = 'avgpool10', 
    input = 'conv10'
    )

graph.add_node(
    Flatten(), 
    name = 'flatten', 
    input = 'avgpool10'
    )

graph.add_node(
    Activation('softmax'), 
    name = 'softmax', 
    input = 'flatten'
    )

graph.add_output(name = 'output', input = 'softmax')

# Customize your solver here

adam = Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-08)
graph.compile(optimizer = adam, loss = {'output': 'categorical_crossentropy'})

print "Model built"

print "Training"

graph.fit({'input': x_train, 'output': y_train}, batch_size = 32, nb_epoch = 32, validation_split = 0.1, verbose = 1)
 
print "Model trained"

# Evaluate the performace

print "Evaluating"

score = graph.evaluate({'input': X_test, 'output': y_test}, batch_size = 32, verbose = 1)

print score
