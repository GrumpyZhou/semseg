import tensorflow as tf
import time

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Wrapper for layers and initial params
def conv2d(x, W, stride=1):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], 
                        padding='SAME')

def maxPool2d(x, stride=2):
    return tf.nn.max_pool(x, ksize=[1, stride, stride, 1],
                        strides=[1, stride, stride, 1], padding='SAME')

def weightVar(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1))
 
def biasVar(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))


def convNet(x, weight, bias, hiddenNeuron, keepProb=0.5):
    ''' 
    Convolution Neural Network Structure
        2 * (conv + relu + maxpooling)
        1 * (fc + relu + drop)
        1 * out
    '''
    
    # Reshape img
    x = tf.reshape(x, [-1, 28, 28,1])
    
    # ConvLayer 1 + ReLu + MaxPooling
    conv1 = maxPool2d(tf.nn.relu(conv2d(x, w['conv1']) + b['conv1']))
    
    # ConvLayer 2 + ReLu + MaxPooling
    conv2 = maxPool2d(tf.nn.relu(conv2d(conv1, w['conv2']) + b['conv2']))

    # FullyConnectedLayer 1 + Dropout, feature size 28/(2*2) = 7
    feats = tf.reshape(conv2, [-1, w['fc1'].get_shape().as_list()[0]])
    z = tf.nn.dropout(tf.nn.relu(tf.matmul(feats, w['fc1']) + b['fc1']), 
                      keepProb)
    # Output Layer
    y = tf.matmul(z, w['out']) + b['out']
    
    return y


# Hyper Parameters
learningRate = 0.001
iters = 500
batchSize = 128
hiddenNeuron = 1024
keepProb = 0.75

imgDim = mnist.train.images.shape[1]
classNum = mnist.train.labels.shape[1]

# Prepare placeholder for input
x = tf.placeholder(tf.float32, shape=[None, imgDim])
y = tf.placeholder(tf.float32, shape=[None, classNum])
pb = tf.placeholder(tf.float32)

# Initial w and b
w = {'conv1': weightVar([5, 5, 1, 32]),
     'conv2': weightVar([5, 5, 32, 64]),
     'fc1': weightVar([7 * 7 * 64, hiddenNeuron]),
     'out': weightVar([hiddenNeuron, classNum])}

tf.histogram_summary('cv1_w',w['conv1'])
tf.histogram_summary('cv2_w',w['conv2'])

b = {'conv1': biasVar([32]),
     'conv2': biasVar([64]),
     'fc1': biasVar([hiddenNeuron]),
     'out': biasVar([classNum])}

# Create and train a ConvNet
yPred = convNet(x, w, b, pb)

# Minimize softmax loss
with tf.name_scope("loss"):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(yPred, y))
    optimizer = tf.train.AdamOptimizer(learningRate).minimize(loss)
    tf.scalar_summary('loss', loss)

# Cal accuracy
with tf.name_scope("accuracy"):
    correct = tf.equal(tf.argmax(yPred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    tf.scalar_summary('acc', accuracy)

# Run the graph
with tf.Session() as sess:

    writer = tf.train.SummaryWriter("./logs", sess.graph)
    summary = tf.merge_all_summaries()
    sess.run(tf.initialize_all_variables())
    
    t1= time.time()
    print("Trainning:")
    for i in range(iters):
        batch_x, batch_y = mnist.train.next_batch(batchSize)
        
        optimizer.run(feed_dict={x:batch_x, y:batch_y, pb: keepProb})

        if i%10 == 0:
            accVal, lossVal = sess.run([accuracy, loss], feed_dict={x:batch_x, y:batch_y, pb: 1.})
            summaryStr = sess.run(summary, feed_dict={x:batch_x, y:batch_y, pb: 1.})
            writer.add_summary(summaryStr, i)
            writer.flush()
            print("step %d, accuracy %f, loss %f" %(i, accVal, lossVal))
    
    #Evaluation
    accVal = sess.run(accuracy,feed_dict={x: mnist.validation.images, y: mnist.validation.labels, pb: 1.})
    print("Validation accuracy %f" %accVal)

    #Test
    accTe = sess.run(accuracy,feed_dict={x: mnist.validation.images, y: mnist.validation.labels, pb: 1.})
    print("Testing accuracy %f" %accTe)
 
