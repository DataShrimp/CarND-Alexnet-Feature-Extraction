import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from alexnet import AlexNet

# TODO: Load traffic signs data.
training_file = "../CarND-Traffic-Sign-Classifier-Project/traffic-signs-data/train.p"
validation_file= "../CarND-Traffic-Sign-Classifier-Project/traffic-signs-data/valid.p"
testing_file = "../CarND-Traffic-Sign-Classifier-Project/traffic-signs-data/test.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

# TODO: Split data into training and validation sets.
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

# TODO: Define placeholders and resize operation.
nb_classes = 43
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int64, shape=(None))
y_onehot = tf.one_hot(y, nb_classes)
resize = tf.image.resize_images(x, [227, 227])

# TODO: pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(resize, feature_extract=True)
# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)

# TODO: Add the final layer for traffic sign classification.
shape = (fc7.get_shape().as_list()[-1], nb_classes)  # use this shape for the weight matrix
fc8W = tf.Variable(tf.truncated_normal(shape, stddev=1e-3))
fc8b = tf.Variable(tf.zeros(nb_classes))
logits = tf.matmul(fc7, fc8W) + fc8b
probs = tf.nn.softmax(logits)

# TODO: Define loss, training, accuracy operations.
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.
rate = 0.001
EPOCHS = 10
BATCH_SIZE = 128

cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
loss = tf.reduce_mean(cross_entropy)
opt = tf.train.AdamOptimizer(rate).minimize(loss, var_list=[fc8W, fc8b])

correct_prediction = tf.arg_max(logits, 1)
accuracy_operation = tf.reduce_mean(tf.cast(tf.equal(correct_prediction,y), tf.float32))

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

# TODO: Train and evaluate the feature extraction model.
from sklearn.utils import shuffle

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for i in range(EPOCHS):
        x_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, X_train.shape[0], BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = x_train[offset:end], y_train[offset:end]
            sess.run(opt, feed_dict={x: batch_x, y: batch_y})
            
        valid = evaluate(X_valid, y_valid)
        print("EPOCH {} ...".format(i))
        print("Valid Accuracy: {:.3f}".format(valid))
        print()

