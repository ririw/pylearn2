"""
Read a text dataset from a file given as the first
argument to this script. Then map over it and produce 
a sparse csr matricx, which is written to the file 
specified as the second argument.

Uses the treebanks tokenizer, because it's nice and
quick. We also filter out punctuation and so on, leaving
only words.

Note: for performance reasons, we often will chunk the
file in very large chunks, of 100K characters or so. This
will affect the windows, but not in any meaningful way.

Finally, we set the number of words in a window and the
vocabulary size at the top of this file.
"""

import nltk.tokenize.treebank
import nltk.tokenize.simple
import sys
from collections import Counter
import scipy.sparse
import numpy as np
import logging
import gzip
import cPickle as pickle
import re
import h5py

logging.basicConfig(level=logging.DEBUG)


class SimpleSplittingTokenizer(object):
	def tokenize(self, input):
		return re.split('\W+', input)


# tokenizer = nltk.tokenize.treebank.TreebankWordTokenizer()
tokenizer = SimpleSplittingTokenizer()

vocab_size = int(1e3)
window_size = 5
chunk_size = int(1e4)


def iterate_file_chunks(input_file_handle):
	done = False
	while not done:
		input_chunk = input_file_handle.read(chunk_size)
		input_char = input_file_handle.read(1)
		while input_char != ' ' and input_char != '':
			input_chunk += input_char
			input_char = input_file_handle.read(1)
		done = input_char == ''
		yield input_chunk
	logging.info('Finished all chunks')


def get_vocab(input_file_handle):
	vocab = Counter()
	for input_chunk in iterate_file_chunks(input_file_handle):
		tokens = tokenizer.tokenize(input_chunk)
		vocab.update(tokens)
	return vocab


def make_windows(input_file_handle, vocab):
	input_file_handle.seek(0, 2)
	file_length = input_file_handle.tell()
	input_file_handle.seek(0, 0)

	for input_chunk in iterate_file_chunks(input_file_handle):
		logging.info('Processed %f%% of file' % (100 * float(input_file_handle.tell()) / file_length))
		tokens = filter(lambda w: w.isalpha() and w in vocab, tokenizer.tokenize(input_chunk))
		for start in range(len(tokens) - window_size + 1):
			window = tokens[start:start + window_size]
			yield window

def write_matrix(dataset, window_vectors):
	logging.info('Writing...')
	data_matrix = scipy.sparse.vstack(window_vectors).todense()
	dataset_len = dataset.shape[0]
	dataset.resize((dataset_len + data_matrix.shape[0], vector_length))
	dataset[dataset_len:dataset_len+data_matrix.shape[0], :] = data_matrix

if __name__ == '__main__':
	in_filename = sys.argv[1]

	logging.info('Starting!')
	with open(in_filename) as f:
		vocab = get_vocab(f)
		limited_vocab = vocab.most_common(vocab_size)
	vocab_index = {word: ix for ix, (word, _) in enumerate(limited_vocab)}
	logging.info('Generated index...')
	vector_length = vocab_size*window_size
	with open(in_filename) as f, h5py.File(sys.argv[2], 'w') as out:
		dataset = out.create_dataset('dataset',
								   (0, vector_length),
								   maxshape=(None, vector_length),
								   compression='gzip',
								   chunks=(10000, vector_length))
		data_ix = 1

		window_vectors = []
		for window in make_windows(f, vocab_index):
			window_vector = scipy.sparse.lil_matrix((1, vector_length))
			for ix, word in enumerate(window):
				window_vector[0, vocab_size * ix + vocab_index[word]] = 1.0
			window_vectors.append(window_vector.tocsr())

			if data_ix % chunk_size == 0:
				write_matrix(dataset, window_vectors)
				window_vectors = []
			data_ix += 1
		write_matrix(dataset, window_vectors)

