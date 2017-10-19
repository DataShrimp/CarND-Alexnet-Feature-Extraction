import pickle
import time
import tensorflow as tf
from keras.datasets import cifar10
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from alexnet import AlexNet

# TODO: Load cifar10 data.
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
y_train = y_train.reshape(-1)
y_test = y_test.reshape(-1)


# TODO: Split data into training and validation sets.
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.33, random_state=42, stratify=y_train)

# TODO: Define placeholders and resize operation.
nb_classes = 10
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int64, shape=(None))
#y_onehot = tf.one_hot(y, nb_classes)
resize = tf.image.resize_images(x, [227, 227])

# TODO: pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(resize, feature_extract=True)
# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)

# TODO: Add the final layer for traffic sign classification.
shape = (fc7.get_shape().as_list()[-1], nb_classes)  # use this shape for the weight matrix
fc8W = tf.Variable(tf.truncated_normal(shape, stddev=1e-2))
fc8b = tf.Variable(tf.zeros(nb_classes))
logits = tf.matmul(fc7, fc8W) + fc8b

# TODO: Define loss, training, accuracy operations.
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.
rate = 0.001
EPOCHS = 10
BATCH_SIZE = 128

cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
loss_op = tf.reduce_mean(cross_entropy)
opt = tf.train.AdamOptimizer(rate)
train_op = opt.minimize(loss_op, var_list=[fc8W, fc8b])

correct_prediction = tf.arg_max(logits, 1)
accuracy_operation = tf.reduce_mean(tf.cast(tf.equal(correct_prediction,y), tf.float32))

def evaluate(X_data, y_data, sess):
    num_examples = len(X_data)
    total_accuracy = 0
    total_loss = 0
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        loss, accuracy = sess.run([loss_op, accuracy_operation], feed_dict={x: batch_x, y: batch_y})
        total_loss += (loss * len(batch_x))
        total_accuracy += (accuracy * len(batch_x))
    return total_loss / num_examples, total_accuracy / num_examples

# TODO: Train and evaluate the feature extraction model.
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        t0 = time.time()
        for offset in range(0, X_train.shape[0], BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(train_op, feed_dict={x: batch_x, y: batch_y})
            
        val_loss, val_acc = evaluate(X_valid, y_valid, sess)
        print("EPOCH {} ...".format(i))
        print("Time: {:.3f} seconds".format(time.time()-t0))
        print("Valid Loss: {:.3f}".format(val_loss))
        print("Valid Accuracy: {:.3f}".format(val_acc))
        print()

