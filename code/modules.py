# Copyright 2018 Stanford University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import print_function
"""This file contains some basic model components"""

import unittest
import numpy as np


import tensorflow as tf
from tensorflow.python.ops.rnn_cell import DropoutWrapper
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import rnn_cell


tf.app.flags.DEFINE_boolean(
    "is_test", False,
    "Whether or not this is running in a test. When running in test sets parameter weights to 1 to simplify calculations."
)
FLAGS = tf.app.flags.FLAGS


class CNNEncoder(object):
    """
    General-purpose module to encode a sequence using a RNN.
    It feeds the input through a RNN and returns all the hidden states.

    Note: In lecture 8, we talked about how you might use a RNN as an "encoder"
    to get a single, fixed size vector representation of a sequence
    (e.g. by taking element-wise max of hidden states).
    Here, we're using the RNN as an "encoder" but we're not taking max;
    we're just returning all the hidden states. The terminology "encoder"
    still applies because we're getting a different "encoding" of each
    position in the sequence, and we'll use the encodings downstream in the model.

    This code uses a bidirectional GRU, but you could experiment with other types of RNN.
    """

    def __init__(self, word_len, num_filters, window_size, char_embedding_size, keep_prob, initializer=tf.contrib.layers.xavier_initializer()):
        """
        Inputs:
          num_filters: Should equal word embedding size since the number of filters is what determines the len of the CNN output
          keep_prob: Tensor containing a single scalar that is the keep probability (for dropout)
        """
        self.word_len = word_len
        self.num_filters = num_filters
        self.window_size = window_size
        self.char_embedding_size = char_embedding_size
        self.keep_prob = keep_prob

        conv_filt_shape = [self.window_size, self.char_embedding_size, 1, self.num_filters]
        self.weights = tf.get_variable("conv_weights"+'_W', shape=conv_filt_shape, dtype=tf.float32, initializer=initializer)
        self.bias = tf.get_variable("conv_weights"+'_b', shape=[self.num_filters], dtype=tf.float32, initializer=initializer)

    def build_graph(self, inputs, masks):
        """
        Inputs:
          inputs: Tensor shape (batch_size, sentence_len, word_len, char_embedding_len)
          masks: Tensor shape (batch_size, sentence_len, word_len).
            Has 1s where there is real input, 0s where there's padding.
            This is used to make sure tf.nn.bidirectional_dynamic_rnn doesn't iterate through masked steps.

        Returns:
          out: Tensor shape (batch_size, seq_len, hidden_size*2).
            This is all hidden states (fw and bw hidden states are concatenated).
        """
        with vs.variable_scope("CNNEncoder"):
            # word_lens = tf.reduce_sum(
            #     masks, reduction_indices=2)  # shape (batch_size x sentence_len)
            sentence_len = tf.shape(inputs)[1]

            inputs = inputs*tf.cast(tf.expand_dims(masks, -1), tf.float32)

            inputs = tf.reshape(inputs, [-1, self.word_len, self.char_embedding_size])

            # input should be of shape [batch, in_height=WORD_SIZE, in_width=CHAR_EMBEDDING_SIZE, in_channels=1]
            # THEREFORE WANT MATRIX WHETERE EACH ROW IS A CHARACTER EMBEDDING

            # setup the filter input shape for tf.nn.conv_2d
            # Want filter to be of size 
            # filter_height = window_size
            # filter_width = char_embedding_size
            # input_channels = 1
            # num_filters should equal word_embedding_size so that after max pooling we end up with a vector of size word_embedding_size for each word


            # Setup the convolutional layer operation
            # stride = [1, 1, 1, 1]: how much to move filter over each of the four input dimensions
            conv_layer = tf.nn.conv2d(tf.expand_dims(inputs, -1), self.weights, [1, 1, 1, 1], padding='VALID')
            conv_layer += self.bias
            conv_layer = tf.nn.relu(conv_layer)

            # now perform max pooling

            # ksize = size of pooling window for each dimension in out_layer. 
            # Want to pool elementwise so that we end up with a vector of size filter_size.
            # After convolution we have matrix of size (WORD_LEN-window_wsize+1 x filter_size)
            # max pool each columns so that we end up with a single number for each filter
            ksize = [1, self.word_len - self.window_size + 1, 1, 1] 
            strides = [1, 1, 1, 1]
            out_layer = tf.nn.max_pool(conv_layer, ksize=ksize, strides=strides, 
                                       padding='VALID')
            out_layer = tf.reshape(tf.squeeze(out_layer), [-1, sentence_len, self.num_filters])
   
            # # Apply dropout
            # out = tf.nn.dropout(out, self.keep_prob)

            return conv_layer, out_layer


class RNNEncoder(object):
    """
    General-purpose module to encode a sequence using a RNN.
    It feeds the input through a RNN and returns all the hidden states.

    Note: In lecture 8, we talked about how you might use a RNN as an "encoder"
    to get a single, fixed size vector representation of a sequence
    (e.g. by taking element-wise max of hidden states).
    Here, we're using the RNN as an "encoder" but we're not taking max;
    we're just returning all the hidden states. The terminology "encoder"
    still applies because we're getting a different "encoding" of each
    position in the sequence, and we'll use the encodings downstream in the model.

    This code uses a bidirectional GRU, but you could experiment with other types of RNN.
    """

    def __init__(self, hidden_size, keep_prob):
        """
        Inputs:
          hidden_size: int. Hidden size of the RNN
          keep_prob: Tensor containing a single scalar that is the keep probability (for dropout)
        """
        self.hidden_size = hidden_size
        self.keep_prob = keep_prob
        self.rnn_cell_fw = rnn_cell.GRUCell(self.hidden_size)
        self.rnn_cell_fw = DropoutWrapper(
            self.rnn_cell_fw, input_keep_prob=self.keep_prob)
        self.rnn_cell_bw = rnn_cell.GRUCell(self.hidden_size)
        self.rnn_cell_bw = DropoutWrapper(
            self.rnn_cell_bw, input_keep_prob=self.keep_prob)

    def build_graph(self, inputs, masks):
        """
        Inputs:
          inputs: Tensor shape (batch_size, seq_len, input_size)
          masks: Tensor shape (batch_size, seq_len).
            Has 1s where there is real input, 0s where there's padding.
            This is used to make sure tf.nn.bidirectional_dynamic_rnn doesn't iterate through masked steps.

        Returns:
          out: Tensor shape (batch_size, seq_len, hidden_size*2).
            This is all hidden states (fw and bw hidden states are concatenated).
        """
        with vs.variable_scope("RNNEncoder"):
            input_lens = tf.reduce_sum(
                masks, reduction_indices=1)  # shape (batch_size)

            # Note: fw_out and bw_out are the hidden states for every timestep.
            # Each is shape (batch_size, seq_len, hidden_size).
            (fw_out, bw_out), _ = tf.nn.bidirectional_dynamic_rnn(
                self.rnn_cell_fw,
                self.rnn_cell_bw,
                inputs,
                input_lens,
                dtype=tf.float32)

            # Concatenate the forward and backward hidden states
            out = tf.concat([fw_out, bw_out], 2)

            # Apply dropout
            out = tf.nn.dropout(out, self.keep_prob)

            return out


class SimpleSoftmaxLayer(object):
    """
    Module to take set of hidden states, (e.g. one for each context location),
    and return probability distribution over those states.
    """

    def __init__(self):
        pass

    def build_graph(self, inputs, masks):
        """
        Applies one linear downprojection layer, then softmax.

        Inputs:
          inputs: Tensor shape (batch_size, seq_len, hidden_size)
          masks: Tensor shape (batch_size, seq_len)
            Has 1s where there is real input, 0s where there's padding.

        Outputs:
          logits: Tensor shape (batch_size, seq_len)
            logits is the result of the downprojection layer, but it has -1e30
            (i.e. very large negative number) in the padded locations
          prob_dist: Tensor shape (batch_size, seq_len)
            The result of taking softmax over logits.
            This should have 0 in the padded locations, and the rest should sum to 1.
        """
        with vs.variable_scope("SimpleSoftmaxLayer"):

            # Linear downprojection layer
            logits = tf.contrib.layers.fully_connected(
                inputs, num_outputs=1,
                activation_fn=None)  # shape (batch_size, seq_len, 1)
            logits = tf.squeeze(logits, axis=[2])  # shape (batch_size, seq_len)

            # Take softmax over sequence
            masked_logits, prob_dist = masked_softmax(logits, masks, 1)

            return masked_logits, prob_dist


class BiDAF(object):
    """Module for basic attention.

    Note: in this module we use the terminology of "keys" and "values" (see lectures).
    In the terminology of "X attends to Y", "keys attend to values".

    In the baseline model, the keys are the context hidden states
    and the values are the question hidden states.

    The keys_size is therefore n and the values_size is m using the definitions
    in the project description.

    We choose to use general terminology of keys and values in this module
    (rather than context and question) to avoid confusion if you reuse this
    module with other inputs.
    """

    def __init__(self, keep_prob):
        """
        Inputs:
          keep_prob: tensor containing a single scalar that is the keep probability (for dropout)
          key_vec_size: size of the key vectors. int
          question_vec_size: size of the question vectors. int
        """
        self.keep_prob = keep_prob

    def build_graph(self,
                    questions,
                    questions_mask,
                    contexts,
                    contexts_mask,
                    w_sim_initializer=tf.contrib.layers.xavier_initializer()):
        """
        Keys attend to values.
        For each key, return an attention distribution and an attention output vector.

        Inputs:
          questions: Tensor shape (batch_size, m, 2h).
          questions_mask: Tensor shape (batch_size, m).
            1s where there's real input, 0s where there's padding
          contexts: Tensor shape (batch_size, n, 2h)
            1s where there's real input, 0s where there's padding.
          contexts_mask: Tensor shape (batch_size, n)
          w_sim_initializer: Initializer for w_sim, the downprojection parameter used to construct the similarity matrix S. 
            Sij = wT sim[ci; qj ; ci * qj ] 

        Outputs:
          attn_dist: Tensor shape (batch_size, num_keys, num_values).
            For each key, the distribution should sum to 1,
            and should be 0 in the value locations that correspond to padding.
          output: Tensor shape (batch_size, num_keys, hidden_size).
            This is the attention output; the weighted sum of the values
            (using the attention distribution as weights).
        """
        with vs.variable_scope("BiDAF"):
            n = contexts.shape[1]
            m = questions.shape[1]
            assert contexts.shape[2] == questions.shape[2] 
            two_h = contexts.shape[2]
            batch_size = contexts.shape[0]

            # Add dimension of length 1 to contexts and questions so that when multiplied together the contexts/questions are broadcast
            # and every pair of contexts/questions is multiplied elementwise.
            contexts_reshaped = contexts[:, :, tf.newaxis, :]  # (batch_size, n, 1, 2h)
            questions_reshaped = questions[:, tf.newaxis, :, :]  # (batch_size, 1, m, 2h)
            elementwise_product = contexts_reshaped * questions_reshaped  # (batch_size, n, m, 2h) elementwise product of all c_i * q_j

            # Tile contexts and questions each to have shape (n, m, 2h) so that each context can be concatenated with each question.
            contexts_broadcast = tf.tile(contexts_reshaped, [1, 1, m, 1])
            questions_broadcast = tf.tile(questions_reshaped,[1, n, 1, 1])

            concatenated = tf.concat(
                [contexts_broadcast, questions_broadcast, elementwise_product],
                3)  # (batch_size, n, m, 6h)

            # Downproject S such that it is (n, m) instead of (n, m, 6h)
            S = tf.contrib.layers.fully_connected(
                concatenated,
                num_outputs=1,
                activation_fn=None,
                weights_initializer=w_sim_initializer
            )  # shape (batch_size, n, m, 1)
            S = tf.squeeze(S, axis=[3])  # shape (batch_size, n, m)

            # Reshape questions_mask from (batch_size, m) to (batch_size 1, m) so that it can be used to mask questions in each row of
            # S. Axis 1 of questions_mask will be broadcast to n and questions_mask will add -infinity to each column of S wherever
            # questions_mask has a 0.
            questions_mask = tf.expand_dims(questions_mask,
                                            1)  # shape (batch_size, 1, m)
            _, alpha = masked_softmax(S, questions_mask,2)  # shape (batch_size, n, m). take softmax over contexts
            a = tf.matmul(alpha, questions)  # shape (batch_size, n, 2h)
            a_dropout = tf.nn.dropout(a, self.keep_prob)

            # Take max of every row so that you end up with a vector of size n. Then take softmax of that vector.
            maxes = tf.reduce_max(S, reduction_indices=[2])  # (batch_size, n)
            _, beta = masked_softmax(maxes, contexts_mask, 1)  # (batch_size, n)
            beta = beta[:, tf.newaxis, :]  # (batch_size, 1, n)
            c_prime = tf.matmul(beta, contexts)  # shape (batch_size, 1, 2h)
            # TODO: Consider adding dropout to c_prime here (since you apply dropout to a).

            blended_reps = tf.concat(
                [contexts, a_dropout, contexts * a_dropout, contexts * c_prime], axis=2)  # (batch_size, context_len, hidden_size*8)

            # print_op = tf.print("attn_dist: ", attn_dist)
            # print_op_1 = tf.print("contexts_reshaped: ", contexts_reshaped, contexts_reshaped.shape)
            # print_op_2 = tf.print("questions_reshaped: ", questions_reshaped, questions_reshaped.shape)
            # print_op_3 = tf.print("elementwise_product: ", elementwise_product, elementwise_product.shape)

            # with tf.control_dependencies([print_op_1, print_op_2, print_op_3]):
            #   output = attn_dist

            # return attn_dist, output
            return blended_reps

class BasicAttn(object):
    """Module for basic attention.

    Note: in this module we use the terminology of "keys" and "values" (see lectures).
    In the terminology of "X attends to Y", "keys attend to values".

    In the baseline model, the keys are the context hidden states
    and the values are the question hidden states.

    We choose to use general terminology of keys and values in this module
    (rather than context and question) to avoid confusion if you reuse this
    module with other inputs.
    """

    def __init__(self, keep_prob, key_vec_size, value_vec_size):
        """
        Inputs:
          keep_prob: tensor containing a single scalar that is the keep probability (for dropout)
          key_vec_size: size of the key vectors. int
          value_vec_size: size of the value vectors. int
        """
        self.keep_prob = keep_prob
        self.key_vec_size = key_vec_size
        self.value_vec_size = value_vec_size

    def build_graph(self, values, values_mask, keys):
        """
        Keys attend to values.
        For each key, return an attention distribution and an attention output vector.

        Inputs:
          values: Tensor shape (batch_size, num_values, value_vec_size).
          values_mask: Tensor shape (batch_size, num_values).
            1s where there's real input, 0s where there's padding
          keys: Tensor shape (batch_size, num_keys, value_vec_size)

        Outputs:
          attn_dist: Tensor shape (batch_size, num_keys, num_values).
            For each key, the distribution should sum to 1,
            and should be 0 in the value locations that correspond to padding.
          output: Tensor shape (batch_size, num_keys, hidden_size).
            This is the attention output; the weighted sum of the values
            (using the attention distribution as weights).
        """
        with vs.variable_scope("BasicAttn"):

            # Calculate attention distribution
            values_t = tf.transpose(
                values, perm=[0, 2,
                              1])  # (batch_size, value_vec_size, num_values)
            attn_logits = tf.matmul(
                keys, values_t)  # shape (batch_size, num_keys, num_values)
            attn_logits_mask = tf.expand_dims(
                values_mask, 1)  # shape (batch_size, 1, num_values)
            _, attn_dist = masked_softmax(
                attn_logits, attn_logits_mask, 2
            )  # shape (batch_size, num_keys, num_values). take softmax over values

            # Use attention distribution to take weighted sum of values
            output = tf.matmul(
                attn_dist,
                values)  # shape (batch_size, num_keys, value_vec_size)

            # Apply dropout
            output = tf.nn.dropout(output, self.keep_prob)

            return attn_dist, output


def masked_softmax(logits, mask, dim):
    """
    Takes masked softmax over given dimension of logits.

    Inputs:
      logits: Numpy array. We want to take softmax over dimension dim.
      mask: Numpy array of same shape as logits.
        Has 1s where there's real data in logits, 0 where there's padding
      dim: int. dimension over which to take softmax

    Returns:
      masked_logits: Numpy array same shape as logits.
        This is the same as logits, but with 1e30 subtracted
        (i.e. very large negative number) in the padding locations.
      prob_dist: Numpy array same shape as logits.
        The result of taking softmax over masked_logits in given dimension.
        Should be 0 in padding locations.
        Should sum to 1 over given dimension.
    """
    exp_mask = (1 - tf.cast(mask, 'float')) * (
        -1e30)  # -large where there's padding, 0 elsewhere
    masked_logits = tf.add(
        logits, exp_mask)  # where there's padding, set logits to -large

    prob_dist = tf.nn.softmax(masked_logits, dim)
    return masked_logits, prob_dist


def simple_test(batch_size, expected_blended_tensor, qn_mask_tensor= [1, 1], context_mask_tensor=[1, 1, 1]):
    context_len = 3
    question_len = 2
    hidden_size = 2

    with tf.Graph().as_default():
        keep_prob = tf.placeholder_with_default(1.0, shape=())

        qn_mask_batch = [qn_mask_tensor for i in range(batch_size)]
        qn_mask = tf.placeholder(tf.int32, shape=[None, question_len])

        context_mask_batch = [context_mask_tensor for i in range(batch_size)]
        context_mask = tf.placeholder(tf.int32, shape=[None, context_len])

        context_tensor = [[1., 2., 3., 4.], [5., 6., 7., 8.],
                           [9., 10., 11., 12.]]
        context_tensor_batch = [context_tensor for i in range(batch_size)]                  
        context_hiddens = tf.placeholder(
            tf.float32, shape=[None, context_len, hidden_size * 2])

        question_tensor = [[1., 1., 2., 2.],
                            [3., 3., 4., 4.]]
        question_tensor_batch = [question_tensor for i in range(batch_size)]                  

        question_hiddens = tf.placeholder(
            tf.float32, shape=[None, question_len, hidden_size * 2])

        attn_layer = BiDAF(keep_prob)

        blended_reps = attn_layer.build_graph(
            question_hiddens, qn_mask, context_hiddens, context_mask,
            tf.initializers.ones())  # attn_output is

        expected_blended_tensor = [expected_blended_tensor for i in range(batch_size)]   
        test_1 = tf.assert_equal(blended_reps, expected_blended_tensor)

        init = tf.global_variables_initializer()
        with tf.Session() as session:
            session.run(init)
            session.run(
                [test_1],
                feed_dict={
                    context_hiddens: context_tensor_batch,
                    question_hiddens: question_tensor_batch,
                    qn_mask: qn_mask_batch,
                    context_mask: context_mask_batch
                })

def run_tests():
    FLAGS.is_test = True
    # Test that it works on batch size of 1.
    simple_test(1, [[  1.,   2.,   3.,   4.,   3.,   3.,   4.,   4.,   3.,   6.,
              12.,  16.,   9.,  20.,  33.,  48.],
            [  5.,   6.,   7.,   8.,   3.,   3.,   4.,   4.,  15.,  18.,
              28.,  32.,  45.,  60.,  77.,  96.],
            [  9.,  10.,  11.,  12.,   3.,   3.,   4.,   4.,  27.,  30.,
              44.,  48.,  81., 100., 121., 144.]])

    # Test that it works on batch size of 2.
    simple_test(2, [[  1.,   2.,   3.,   4.,   3.,   3.,   4.,   4.,   3.,   6.,
              12.,  16.,   9.,  20.,  33.,  48.],
            [  5.,   6.,   7.,   8.,   3.,   3.,   4.,   4.,  15.,  18.,
              28.,  32.,  45.,  60.,  77.,  96.],
            [  9.,  10.,  11.,  12.,   3.,   3.,   4.,   4.,  27.,  30.,
              44.,  48.,  81., 100., 121., 144.]])

    # Test that masking the questions and contexts works.
    simple_test(1, [[  1.,   2.,   3.,   4.,   1.,   1.,   2.,   2.,   1.,   2.,
              6.,  8.,   5.,  12.,  21.,  32.],
            [  5.,   6.,   7.,   8.,   1.,   1.,   2.,   2.,  5.,  6.,
              14.,  16.,  25.,  36.,  49.,  64.],
            [  9.,  10.,  11.,  12.,   1.,   1.,   2.,   2.,  9.,  10.,
              22.,  24.,  45., 60., 77., 96.]], [1, 0], [1, 1, 0])




class TestModules(unittest.TestCase):

    def test_cnn_encoder(self):
        """
        Test CNNEncoder on batch of size two. Uses padding chars which must be masked.
        """
        with tf.Graph().as_default():
            batch_size = 2
            qn_len = 3                
            word_len = 7

            num_filters=3
            window_size=2
            char_embedding_size=4
            keep_prob = 1

            qn_char_embs = tf.constant([[[[ 4.,  5.,  6.,  7.], # O
                                     [ 4.,  5.,  6.,  7.], # R
                                     [ 8.,  9., 10., 11.], # A - > 2
                                     [ 4.,  5.,  6.,  7.], # N
                                     [ 4.,  5.,  6.,  7.], # G
                                     [ 4.,  5.,  6.,  7.], # E
                                     [ 0.,  1.,  2.,  3.]], # PAD

                                    [[ 0.,  1.,  2.,  3.], # PAD
                                     [ 0.,  1.,  2.,  3.], # PAD
                                     [ 0.,  1.,  2.,  3.], # PAD
                                     [ 0.,  1.,  2.,  3.], # PAD
                                     [ 0.,  1.,  2.,  3.], # PAD
                                     [ 0.,  1.,  2.,  3.], # PAD
                                     [ 0.,  1.,  2.,  3.]],

                                    [[ 0.,  1.,  2.,  3.], # PAD
                                     [ 0.,  1.,  2.,  3.], # PAD
                                     [ 0.,  1.,  2.,  3.], # PAD
                                     [ 0.,  1.,  2.,  3.], # PAD
                                     [ 0.,  1.,  2.,  3.], # PAD
                                     [ 0.,  1.,  2.,  3.], # PAD
                                     [ 0.,  1.,  2.,  3.]]], #PAD

                                    [[[ 4.,  5.,  6.,  7.], # O
                                     [ 4.,  5.,  6.,  7.], # R
                                     [ 8.,  9., 10., 11.], # A - > 2
                                     [ 4.,  5.,  6.,  7.], # N
                                     [ 4.,  5.,  6.,  7.], # G
                                     [ 4.,  5.,  6.,  7.], # E
                                     [ 0.,  1.,  2.,  3.]], # PAD

                                    [[ 4.,  5.,  6.,  7.], # U
                                     [ 4.,  5.,  6.,  7.], # N
                                     [ 4.,  5.,  6.,  7.], # K
                                     [ 4.,  5.,  6.,  7.], # N
                                     [ 4.,  5.,  6.,  7.], # O
                                     [ 4.,  5.,  6.,  7.], # W
                                     [ 4.,  5.,  6.,  7.]], # N

                                    [[ 0.,  1.,  2.,  3.], # PAD
                                     [ 0.,  1.,  2.,  3.], # PAD
                                     [ 0.,  1.,  2.,  3.], # PAD
                                     [ 0.,  1.,  2.,  3.], # PAD
                                     [ 0.,  1.,  2.,  3.], # PAD
                                     [ 0.,  1.,  2.,  3.], # PAD
                                     [ 0.,  1.,  2.,  3.]]]], dtype=tf.float32) #PAD 
            qn_char_masks = np.array([[[1, 1, 1, 1, 1, 1, 0], 
                                [0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0]],

                                [[1, 1, 1, 1, 1, 1, 0], 
                                [1, 1, 1, 1, 1, 1, 1],
                                [0, 0, 0, 0, 0, 0, 0]]])
            expected_conv_output = [[[[45., 45., 45.]],
                                      [[61., 61., 61.]],
                                      [[61., 61., 61.]],
                                      [[45., 45., 45.]],
                                      [[45., 45., 45.]],
                                      [[23., 23., 23.]]],

                                     [[[1., 1., 1.]],
                                      [[1., 1., 1.]],
                                      [[1., 1., 1.]],
                                      [[1., 1., 1.]],
                                      [[1., 1., 1.]],
                                      [[1., 1., 1.]]],

                                     [[[1., 1., 1.]],
                                      [[1., 1., 1.]],
                                      [[1., 1., 1.]],
                                      [[1., 1., 1.]],
                                      [[1., 1., 1.]],
                                      [[1., 1., 1.]]],


                                     [[[45., 45., 45.]],
                                      [[61., 61., 61.]],
                                      [[61., 61., 61.]],
                                      [[45., 45., 45.]],
                                      [[45., 45., 45.]],
                                      [[23., 23., 23.]]],

                                     [[[45., 45., 45.]],
                                      [[45., 45., 45.]],
                                      [[45., 45., 45.]],
                                      [[45., 45., 45.]],
                                      [[45., 45., 45.]],
                                      [[45., 45., 45.]]],

                                     [[[1., 1., 1.]],
                                      [[1., 1., 1.]],
                                      [[1., 1., 1.]],
                                      [[1., 1., 1.]],
                                      [[1., 1., 1.]],
                                      [[1., 1., 1.]]]]

            expected_max_pool_out = [[[61., 61., 61.],
                                      [1., 1., 1.],
                                      [1., 1., 1.]],

                                     [[61., 61., 61.],
                                      [45., 45., 45.],
                                      [1., 1., 1.]]]

            encoder = CNNEncoder(word_len, num_filters,window_size, char_embedding_size,keep_prob, initializer=tf.initializers.ones())
            conv_out, max_pool_out = encoder.build_graph(qn_char_embs, qn_char_masks)

            conv_test = tf.assert_equal(conv_out, expected_conv_output)
            max_pool_test = tf.assert_equal(max_pool_out, expected_max_pool_out)

            init = tf.global_variables_initializer()
            with tf.Session() as sess:
                sess.run(init)
                sess.run([conv_test, max_pool_test])

        # Output after convolution should be (batch_size*qn_len = 6, word_len-filter_size+1 = 6, num_filters = 3)
        # Output after pooling should be of shape (batch_size = 2, qn_len = 3, num_filters = 3)



if __name__ == "__main__":
    unittest.main()