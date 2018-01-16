#!/user/bin/env python
'''
read_data.py: This file contains functions to parse data file and generate \
nerual network inputs

Authorship information:
    __author__ = "Mars Huang"
    __email__ = "marshuang80@gmail.com:
    __status__ = "debug"
'''

import numpy as np
import random

def read_data(file_name):
    '''Read the input file and split the data input features and labels

    Args:
        file_name (String): input file name

    Return:
        seq (np.array): lists of one-hot encoded amino acid sequence
        f (np.array): list of properties for the protein sequence
        y (np.array): one-hot encoded labels
    '''
    # Load input data
    input_data = np.load(file_name)
    input_data.shape = (6133,700,57)

    # Get one-hot encoded labels
    y = input_data[:,:,22:31].astype(np.float32)

    # One-hot encoded amino acid seqeuences
    seq = input_data[:,:,:22].astype(np.float32)

    # Other features: N & C terminals, solvent accessibility, sequence profile
    f = input_data[:,:,35:].astype(np.float32)

    return seq,f,y


def test_batch(x,f,y,split):
    '''Split the input data into train and test batches
    Args:
        x (list): aa features
        f (list): sequence profiles
        y (list): sequence labels
        ratio (float): ratio of training to testing set
    Return:
        x_train: batch of train aa
        f_train: batch of trian features
        y_train: batch of train label
        x_test: batch of test aa
        f_test: batch of test features
        y_test: batch of test label
    '''
    x_train, f_train, y_train = x[:split], f[:split], y[:split]
    x_test, f_test, y_test = x[split:], f[split:], y[split:]

    return x_train, f_train, y_train, x_test, f_test, y_test


def valid_batch(x,f,y,batch_size, ratio):
    '''Generate random batches of samples during training
    Args:
        x (list): input aa
        f (list): input features
        y (list): sequence labels
        batch_size (int): size of each training batch
        ratio (float): ratio of training to validation set
    Return:
        x_train: batch of aa
        f_train: batch of features
        y_train: batch of train label
        x_valid: batch of validation data
        f_valid: batch of features
        y_valid: batch of validation label
     '''
    # Sample data with batch_size
    idx = random.sample(range(len(x)),batch_size)
    xs, fs, ys = x[idx], f[idx], y[idx]

	# Generate batches
    split = int(len(xs) * ratio)
    x_train, f_train, y_train = xs[:split], fs[:split], ys[:split]
    x_valid, f_valid, y_valid = xs[split:], fs[split:], ys[split:]
    return x_train, f_train, y_train, x_valid, f_valid, y_valid

