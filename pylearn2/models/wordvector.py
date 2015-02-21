import numpy
import theano
from theano.printing import Print
from theano import tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from theano.compat.six.moves import zip as izip, reduce
from pylearn2.utils.rng import make_np_rng, make_theano_rng
from pylearn2.utils import sharedX
from pylearn2.space import VectorSpace
from pylearn2.models import Model

class SubVectorModel(Model):
	def __init__(self,
	             input_width, 
	             output_width, 
	             num_subvectors, 
	             seed=123):
		super(SubVectorModel, self).__init__()
		self.input_width = input_width
		self.output_width = output_width
		self.num_subvectors = num_subvectors
        	self.act_enc = T.tanh

		self.input_space = VectorSpace(input_width*num_subvectors)
		self.output_space = VectorSpace(output_width*num_subvectors)
        	rng = make_np_rng(seed, which_method="randn")
		self.transform = sharedX(
			rng.rand(input_width*num_subvectors, output_width), 
			name='transform', 
			borrow=True)
		self.inverse_transform = self.transform.T
		self.index_stream = RandomStreams(seed=234)
		self._params = [self.transform]

	def get_input_space(self):
		return self.input_space

	def get_output_space(self):
		return self.output_space

	def reconstruct(self, X):
		randomized = self.randomize(X)
		return self.decode(self.encode(randomized))

	def randomize(self, X):
		if isinstance(X, T.Variable):
			return self._randomize_single(X)
		else:
			return [self.randomize(v) for v in X]

	def _randomize_single(self, X):
		randomized_vector = X.copy()
		n = self.index_stream.random_integers((1,), 0, self.input_width)
		offset = (self.num_subvectors / 2) * self.input_width
		zeroed = T.set_subtensor(
			randomized_vector[0:-1, offset:offset+self.input_width], 
			0.0,
			inplace=False)
		inced = T.inc_subtensor(
			zeroed[0:-1, offset+n], 
			1.0,
			inplace=False) # TODO: why doesn't inplace work here?
		return inced

	def encode(self, X):
		if isinstance(X, T.Variable):
			return self._encode_single(X)
		else:
			return [self.encode(v) for v in X]

	def _encode_single(self, X):
		sub_vectors = []
		for v in range (0, self.num_subvectors):
			subvec = X[v*self.input_width:(v+1)*self.input_width]
			subvec_transform = self.act_enc(T.dot(subvec, self.transform))
			sub_vectors.append(subvec_transform)
		return T.concatenate(sub_vectors)

	def decode(self, hiddens):
		if isinstance(hiddens, T.Variable):
			return self._decode_single(hiddens)
		else:
			return [self.decode(v) for v in hiddens]

	def _decode_single(self, hiddens):
		sub_vectors = []
		for v in range (0, self.num_subvectors):
			subvec = hiddens[v*self.output_width:(v+1)*self.output_width]
			subvec_transform = self.act_enc(T.dot(subvec, self.inverse_transform))
			sub_vectors.append(subvec_transform)
		return T.concatenate(sub_vectors)

	def get_weights(self, borrow=False):
		return self.transform.get_value(borrow=borrow)

	def get_weights_format(self):
		return ['v', 'h']

	def get_monitoring_channels(self, X):
		reconstruction = self.reconstruct(X)
		clamped = T.round(reconstruction)
		anded = T.and_(X, clamped)
		err = T.sum(T.sum(anded))
		return {'err': err}

	def __call__(self, X):
		return self.encode(X)

if __name__=='__main__':
	import numpy as np
	from theano import function
	def simple_setup():
		X = np.zeros((500))
		X[0] = 1.
		X[101] = 1.
		X[202] = 1.
		X[303] = 1.
		X[404] = 1.
		XT = T.as_tensor(X)
		return XT

	def basic_randomize_test():
		XT = simple_setup()
		f = function([], model.randomize(XT))
		print f()
		assert f()[202] != 1.0, 'probabalistic failure, try again (1/100 chance)'

	def basic_encode_test():
		XT = simple_setup()
		f = function([], model.encode(XT))
		print f()
		print f().shape

	def basic_decode_test():
		XT = simple_setup()
		e = model.encode(XT)
		d = model.decode(e)
		f = function([], d)
		print f()

	model = SubVectorModel(100, 10, 5)
	basic_encode_test()
	basic_decode_test()
