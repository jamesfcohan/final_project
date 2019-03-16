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

"""This file contains a function to read the GloVe vectors from file,
and return them as an embedding matrix"""

from __future__ import absolute_import
from __future__ import division

from tqdm import tqdm
import numpy as np
import unittest

import os

_PAD = b"<pad>"
_UNK = b"<unk>"
_START_VOCAB = [_PAD, _UNK]
PAD_ID = 0
UNK_ID = 1

MAIN_DIR = os.path.relpath(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # relative path of the main directory
DEFAULT_DATA_DIR = os.path.join(MAIN_DIR, "data") # relative path of data dir


def get_glove(glove_path, glove_dim):
    """Reads from original GloVe .txt file and returns embedding matrix and
    mappings from words to word ids.

    Input:
      glove_path: path to glove.6B.{glove_dim}d.txt
      glove_dim: integer; needs to match the dimension in glove_path

    Returns:
      emb_matrix: Numpy array shape (400002, glove_dim) containing glove embeddings
        (plus PAD and UNK embeddings in first two rows).
        The rows of emb_matrix correspond to the word ids given in word2id and id2word
      word2id: dictionary mapping word (string) to word id (int)
      id2word: dictionary mapping word id (int) to word (string)
    """

    print "Loading GLoVE vectors from file: %s" % glove_path
    vocab_size = int(4e5) # this is the vocab size of the corpus we've downloaded

    emb_matrix = np.zeros((vocab_size + len(_START_VOCAB), glove_dim))
    word2id = {}
    id2word = {}

    random_init = True
    # randomly initialize the special tokens
    if random_init:
        emb_matrix[:len(_START_VOCAB), :] = np.random.randn(len(_START_VOCAB), glove_dim)

    # put start tokens in the dictionaries
    idx = 0
    for word in _START_VOCAB:
        word2id[word] = idx
        id2word[idx] = word
        idx += 1

    # go through glove vecs
    with open(glove_path, 'r') as fh:
        for line in tqdm(fh, total=vocab_size):
            line = line.lstrip().rstrip().split(" ")
            word = line[0]
            vector = list(map(float, line[1:]))
            if glove_dim != len(vector):
                raise Exception("You set --glove_path=%s but --embedding_size=%i. If you set --glove_path yourself then make sure that --embedding_size matches!" % (glove_path, glove_dim))
            emb_matrix[idx, :] = vector
            word2id[word] = idx
            id2word[idx] = word
            idx += 1

    final_vocab_size = vocab_size + len(_START_VOCAB)
    assert len(word2id) == final_vocab_size
    assert len(id2word) == final_vocab_size
    assert idx == final_vocab_size

    return emb_matrix, word2id, id2word

def get_chars(vocab_path):
    char2id = {}
    id2char = {}
    idx = 0

    for char in _START_VOCAB:
        char2id[char] = idx
        id2char[idx] = char
        idx += 1

    chars = []
    with open(vocab_path, 'r') as f:
        for line in f:
            c = line.strip()
            assert len(c) == 1
            chars.append(c)

    for char in chars:
        char2id[char] = idx
        id2char[idx] = char
        idx += 1

    return char2id, id2char


class TestGetChars(unittest.TestCase):

    def test_get_chars(self):
        expected_char2id = {
            '<pad>': 0,
            '<unk>': 1,
            '!': 2,
            '"': 3,
            '#': 4,
            '$': 5,
            '%': 6,
            '&': 7,
            "'": 8,
            '(': 9,
            ')': 10,
            '*': 11,
            '+': 12,
            ',': 13,
            '-': 14,
            '.': 15,
            '/': 16,
            '0': 17,
            '1': 18,
            '2': 19,
            '3': 20,
            '4': 21,
            '5': 22,
            '6': 23,
            '7': 24,
            '8': 25,
            '9': 26,
            ':': 27,
            ';': 28,
            '<': 29,
            '=': 30,
            '>': 31,
            '?': 32,
            '@': 33,
            '[': 34,
            ']': 35,
            '^': 36,
            '_': 37,
            '`': 38,
            'a': 39,
            'b': 40,
            'c': 41,
            'e': 43,
            'd': 42,
            'f': 44,
            'g': 45,
            'h': 46,
            'i': 47,
            'j': 48,
            'k': 49,
            'l': 50,
            'm': 51,
            'n': 52,
            'o': 53,
            'p': 54,
            'q': 55,
            'r': 56,
            's': 57,
            't': 58,
            'u': 59,
            'v': 60,
            'w': 61,
            'x': 62,
            'y': 63,
            'z': 64,
            '{': 65,
            '|': 66,
            '}': 67,
            '~': 68,
        }
        expected_id2char = dict((v, k) for k, v in expected_char2id.iteritems())

        file_path = os.path.join(DEFAULT_DATA_DIR, "char_vocabulary.txt")
        char2id, id2char = get_chars(file_path)

        self.assertEqual(char2id, expected_char2id)
        self.assertEqual(id2char, expected_id2char)   

if __name__ == "__main__":
    unittest.main()
