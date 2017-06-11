#!/user/bin/env python
'''
structure_prediction.py: The neural network used to predict protein secondary
                        structures.

Authorship information:
    __author__ = "Mars Huang"
    __email__ = "marshuang80@gmai.com:
    __status__ = "debug"
'''
# Imports
import tensorflow as tf
import numpy as np
from read_data import *

# Neural network parameters
file_name = 'cullpdb+profile_6133.npy.gz'
batch_size = 128
epoches = 1000
learning_rate = 0.0005
num_input_variables = 22
num_filter = 16
num_rnn_cells = 256
sequence_length = 700
hidden_neurons_1 = 512
hidden_neurons_2 = 256
hidden_neurons_3= 256
num_output_class = 9
valid_ratio = 1
filter_sizes = [3,5,7]
filter_sizes_2 = [1,2,3]
random_seed = 7

# Lists to store outputs
losses = []
accs = []

# Input placeholders
x_input = tf.placeholder(np.float32, shape=[None, sequence_length, num_input_variables])
var_input = tf.placeholder(np.float32, shape=[None, sequence_length, 22])
y_label = tf.placeholder(np.float32, shape=[None, sequence_length, num_output_class])
alpha = tf.placeholder(tf.float32, shape = [])


def conv_aa_layers(x_input):
    '''The function generates a few convolutional network layers with 3,5,7
    filter lengths for the one-hot encoded amino acid sequence.

    Args:
        x_input (tf.placeholder): The batch of one hot encoded protein sequence

    Returns:
        conv_output (list): List of outputs from each CNN layer
     '''
    conv_outputs = []

    # Using 1d convolution to scan over the sequence with window of "size"
    for size in filter_sizes:

        with tf.name_scope("conv_%i"%(size)):
            # Define convolution weights
            conv_w = tf.get_variable("conv_%i"%size, shape = [size, num_input_variables,
               num_filter], initializer = tf.contrib.layers.xavier_initializer(seed = random_seed))

            # Get convolution output from filters
            conv_out_w = tf.nn.conv1d(x_input, conv_w, stride = 1,
                        padding = "SAME",)

            # Activation
            conv_output = tf.nn.relu(conv_out_w, name = "conv_%i_activation"%size)

            # Batch nomalization
            conv_output = tf.layers.batch_normalization(conv_output)

        conv_outputs.append(conv_output)

    return conv_outputs


def conv_features_layers(var_input):
    '''The function generates a few convolutional network layers with 3,5 and 7
    filter lengths on the input features.

    Args:
        var_input (tf.placeholder): The batch of protein features

    Returns:
        conv_features_output (list): List of outputs from each features
                                     convolutional neural network layer
     '''
    conv_features_outputs = []
    for size in filter_sizes:

        with tf.name_scope("conv_freaters_%i"%(size)):
            # Define convolution weights and biases
            conv_w = tf.get_variable("conv_features_%i"%size, shape = [size, num_input_variables,
               num_filter], initializer = tf.contrib.layers.xavier_initializer(seed = random_seed))
            # Get convolution output from filters
            conv_out_w = tf.nn.conv1d(var_input, conv_w, stride = 1,
                        padding = "SAME",)
            # Activation
            conv_output = tf.nn.relu(conv_out_w, name = "conv_%i_activation"%size)

            # Batch nomalization
            conv_output = tf.layers.batch_normalization(conv_output)

        conv_features_outputs.append(conv_output)

    return conv_features_outputs


def concat_layer(conv_outputs, conv_features_outputs, x_input, var_input):
    '''Concaternate and reshape each convolution outputs by horizontally stacking
    them together with the original inputs

    Args:
        conv_outputs (list): List of outputs from aa neural network layer 1
        conv_features_outputs (list): List of output from the features nn layer 1
        x_input (tensor): original one-hot encoded sequence
        var_input (tensor): original features vectors

    Returns:
        concat_aa (tensor): concatinated tensor from amino acid CNN layer 1
        concat_features (tensor): concatinated tensor from features CNN layer 1
    '''
    # Concat outputs from amino acid CNN layers
    to_concat = [layer for layer in conv_outputs]
    concat_outputs = tf.concat(to_concat, 2)

    # Concat with original data
    concat_aa = tf.concat([concat_outputs, x_input],2)

    # Concat outputs from features CNN layers
    to_concat_features  = [layers1 for layers1 in conv_features_outputs]
    concat_features_outputs = tf.concat(to_concat_features, 2)

    # Concat wuth original features
    concat_features = tf.concat([concat_features_outputs, var_input],2)

    return concat_aa, concat_features


def conv_aa_layers_2(concat_aa):
    '''The function generates the second convolutional network layers with 1,2,3
    filter lengths for the outputs from first amino acid cnn layer.

    Args:
        concat_aa (tf.placeholder): The batch of one hot encoded protein sequence

    Returns:
        conv_output (list): List of outputs from each CNN layer
     '''
    conv_outputs = []

    # Using 1d convolution to scan over the sequence with window of "size"
    for size in filter_sizes_2:

        with tf.name_scope("conv2_%i"%(size)):
            # Define convolution weights and biases
            conv_w = tf.get_variable("conv2_%i"%size, shape = [size, num_input_variables + 3*num_filter,
               num_filter], initializer = tf.contrib.layers.xavier_initializer(seed = random_seed))

            # Get convolution output from filters
            conv_out_w = tf.nn.conv1d(concat_aa, conv_w, stride = 1,
                        padding = "SAME",)

            # Activation
            conv_output = tf.nn.relu(conv_out_w, name = "conv2_%i_activation"%size)

            # Batch normailization
            conv_output = tf.layers.batch_normalization(conv_output)

        conv_outputs.append(conv_output)

    return conv_outputs


def conv_features_layers_2(concat_features):
    '''The function generates a few convolutional network layers with 1,2 and 3
    filter lengths for the outputs from the first features cnn layer

    Args:
        concat_features (tf.placeholder): The concaternated output from the first
                                          features CNN layer

    Returns:
        conv_output (list): List of outputs from each neural network
     '''
    conv_outputs = []

    # Using 1d convolution to scan over the sequence with window of "size"
    for size in filter_sizes_2:

        with tf.name_scope("conv2_features_%i"%(size)):
            # Define convolution weights and biases
            conv_w = tf.get_variable("conv2_features_%i"%size, shape = [size, num_input_variables + 3*num_filter,
               num_filter], initializer = tf.contrib.layers.xavier_initializer(seed = random_seed))

            # Get convolution output from filters
            conv_out_w = tf.nn.conv1d(concat_features, conv_w, stride = 1,
                        padding = "SAME",)

            # Activation
            conv_output = tf.nn.relu(conv_out_w, name = "conv_features_%i_activation"%size)

            conv_output = tf.layers.batch_normalization(conv_output)

        conv_outputs.append(conv_output)

    return conv_outputs


def concat_layer_2(conv_outputs, conv_features_outputs, concat_aa, concat_features):
    '''Concaternate and reshape each convolution outputs by horizontally stacking
    them together with the first CNN outputs. The concatinate resultst form AA
    and features together.

    Args:
        conv_outputs (list): List of outputs from aa neural network layer 1
        conv_features_outputs (list): List of output from the features nn layer 1
        concat_aa (tensor): outputs from first amino acid CNN layer
        concat_features (tensor): outputs from first features CNN layer

    Returns:
        concat_input (tensor): concatinated tensor from amino acid CNN layer 1
    '''
    # Concat layers trained on aa
    to_concat = [layer for layer in conv_outputs]
    concat_outputs = tf.concat(to_concat, 2)
    concat_outputs = tf.concat([concat_outputs,concat_aa],2)

    # Concat layers trained on freatures
    to_concat_features  = [layers1 for layers1 in conv_features_outputs]
    concat_features_outputs = tf.concat(to_concat_features, 2)
    concat_features_outputs = tf.concat([concat_features_outputs,concat_features], 2)

    # Concat both features and aa
    concat_input = tf.concat([concat_features_outputs,concat_outputs],2)

    # First fully connected layer with 200 neurons
    reshape_layer_1 = tf.reshape(concat_input, [batch_size * sequence_length, 2*(3*num_filter+3*num_filter+num_input_variables)])
    layer_1 = tf.layers.dense(reshape_layer_1, hidden_neurons_1, activation=tf.nn.relu, use_bias=True)
    concat_input =  tf.layers.batch_normalization(layer_1)

    return concat_input


def bi_nn_layer(rnn_input):
    '''Run a bidirectional recurrent neural network on input data

    Args:
        rnn_input (tensor): The data to run RNN on

    Retuns:
        rnn_output_dropout (tensor): The output from RNN
    '''
    # Split the input data
    rnn_inputs = tf.split(rnn_input, sequence_length, 0)

    # bidirectional rnn
    rnn_forward = tf.contrib.rnn.BasicLSTMCell(num_rnn_cells)
    rnn_backward = tf.contrib.rnn.BasicLSTMCell(num_rnn_cells)
    rnn_outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(rnn_forward,
                        rnn_backward,rnn_inputs,dtype = tf.float32) # dtype?

    # Reshape RNN output
    rnn_outputs_reshaped = tf.reshape(rnn_outputs, [batch_size*sequence_length, 2*num_rnn_cells])

    # Weight dropout for regularization
    rnn_outputs_dropout = tf.nn.dropout(rnn_outputs_reshaped, keep_prob = 0.5)

    return rnn_outputs_dropout


def dense_layers(rnn_outputs_dropout):
    '''3 fully connnect layers

    Args:
        rnn_outputs_dropout (tensor): The recurrent neural network output

    Return:
        dense_output (tensor): batch_size*700*9 tensor for confidence in the
                               different categories
    '''
    # Fully connected layer 1
    dense_layer = tf.layers.dense(rnn_outputs_dropout, hidden_neurons_2, activation=tf.nn.relu, use_bias=True)

    # Fully connected layer 2
    dense_layer1 = tf.layers.dense(dense_layer, hidden_neurons_3, activation=tf.nn.relu, use_bias=True)

    # Last fully connected layer
    dense_layer_2 = tf.layers.dense(dense_layer1, num_output_class,
                use_bias=True,
                bias_initializer = tf.random_normal_initializer(),
                kernel_initializer = tf.random_normal_initializer())

    # Reshape output
    dense_output = tf.reshape(dense_layer_2, [batch_size,sequence_length, num_output_class])

    return dense_output


def optimization(dense_output, y_input, alpha):
    '''Optimize model using AdamOptimizer

    Args:
        dense_output (tensor): The output from RNN
        y_input (tf.placeholder): y input labels
        alpha (float): learning rate

    Returns:
        optimizer (RMSPropOptimizer): optimizer used to train neural network
        loss (float): the training loss for each epoche
    '''
    # Calculate loss for gradient
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=dense_output, labels=y_input))

    # Optimizer to minize loss
    optimizer = tf.train.AdamOptimizer(learning_rate,).minimize(loss)
    # optimizer = tf.train.RMSPropOptimizer(alpha,).minimize(loss)

    return optimizer,loss


def get_accuracy(pred,y_input):
    '''Calcuate the prediction accuracy on the non-padded part of the sequence

    Args:
        pred (tensor): the predicted strucutres from the neural network model
        y_input (tensor): the original labels of structures of input data

    Returns:
        accuracy (float): the accuracy of the predictions
        seq_lens (list): the sequence lengths of each input sequence
    '''
    # Get actual labels
    labels = np.argmax(y_input,2)

    # Find the last aa in the squence
    last_aa = [(l == 8).sum() for l in labels]
    accs = []

    # Calculate accuracy based on actual amino acid sequence length
    for p,l,stop in zip(pred, labels, last_aa):
        accs.append(np.sum(np.equal(p[:-stop],l[:-stop]))/(sequence_length-stop))

    # Get lisit of sequence lengths
    seq_lens= [700-l for l in last_aa]

    return accs,seq_lens


if __name__ == "__main__":
    print("Reading input data...")
    # Get input data
    x,f,y = read_data(file_name)

    # Split training and testing data
    x,f,y,x_test,f_test,y_test = test_batch(x,f,y,5600)

    print("Building netowrk layers...")
    # First CNN layers
    conv_aa = conv_aa_layers(x_input)
    conv_features = conv_features_layers(var_input)
    concat_aa, concat_features = concat_layer(conv_aa, conv_features, x_input, var_input)

    # Second CNN layers
    conv_aa_2 = conv_aa_layers_2(concat_aa)
    conv_features_2 = conv_features_layers_2(concat_features)
    concat_outputs_2 = concat_layer_2(conv_aa_2, conv_features_2, concat_aa, concat_features)

    # Recurrent layers, dense layers, output and optimization
    rnn_output = bi_nn_layer(concat_outputs_2)
    dense_output = dense_layers(rnn_output)
    optimizer,loss = optimization(dense_output, y_label, alpha)
    #predictions = get_result(dense_output)

    print("Start training...")
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Training iterations
        for i in range(epoches):
            # Generate training and validation batches
            x_train, f_train, y_train, x_valid, f_valid, y_valid = valid_batch(x,f,y,batch_size,valid_ratio)

            # Optimize neural network
            dense_out,cost,_ = sess.run([dense_output,loss,optimizer], feed_dict={x_input: x_train,var_input:f_train ,y_label: y_train, alpha: learning_rate})

            # Loss and accuracy
            losses.append(cost)
            pred = tf.argmax(dense_out,2).eval()
            acc, _ = get_accuracy(pred, y_train)
            accs.append(sum(acc)/batch_size)
            print(sum(acc)/batch_size)

            # Print predictions every 5 epochs
            if i % 5 == 0:
                print(pred[0])
                print(np.argmax(y_train,2)[0])

        # Test on all data
        train_out = sess.run(dense, feed_dict={x_input: x, var_input: f, y_input: y})
        train_acc, train_length = get_accuracy(test_out,t_test)

        # Test on training data
        test_out = sess.run(dense, feed_dict={x_input: x_test, var_input: f_test, y_input: y_test})
        test_acc, test_length = get_accuracy(test_out,t_test)

        # Save model
        saver = tf.train.Saver()
        save_path = saver.save(sess, "./model.ckpt")
        print("Model saved in file: %s" % save_path)
