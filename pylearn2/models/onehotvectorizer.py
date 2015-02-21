import logging
import numpy as np
from theano import tensor
import theano
from pylearn2.blocks import Block
from pylearn2.models import Model
from pylearn2.models.autoencoder import Autoencoder
from pylearn2.space import VectorSpace
from pylearn2.utils import sharedX


class OneHotVectorizerTransform(Model, Block):
	def __init__(self, onehot_vector_length, output_vector_width):
		super(OneHotVectorizerTransform, self).__init__()
		self.onehot_vector_length = onehot_vector_length
		self.output_vector_width = output_vector_width
		self.data_vec = sharedX(
			np.random.normal(size=(self.onehot_vector_length, self.output_vector_width)),
			'data_vec')
		self._params = [
			self.data_vec
		]

	def get_input_space(self):
		return VectorSpace(self.onehot_vector_length)

	def get_output_space(self):
		return VectorSpace(self.output_vector_width)

	def __call__(self, inputs):
		if isinstance(inputs, tensor.Variable):
			return tensor.dot(inputs, self.data_vec)
		else:
			return [self(input) for input in inputs]

class MultiAppliedModel(Model, Block):
	def __init__(self, model, n_applications, call_target):
		super(MultiAppliedModel, self).__init__()
		self.n_applications = n_applications
		self.model = model
		self.call_target = call_target
		self._params = [param for param in model._params]

	def get_input_space(self):
		vs = self.model.get_input_space()
		assert isinstance(vs, VectorSpace)
		total_size = vs.get_total_dimension() * self.n_applications
		return VectorSpace(total_size, vs.sparse, vs.dtype)

	def get_output_space(self):
		vs = self.model.get_output_space()
		assert isinstance(vs, VectorSpace)
		total_size = vs.get_total_dimension() * self.n_applications
		return VectorSpace(total_size, vs.sparse, vs.dtype)

	def __call__(self, inputs):
		if isinstance(inputs, tensor.Variable):
			slice_width = self.model.get_input_space().get_total_dimension()
			results = []
			for slice_ix in range(self.n_applications):
				slice_start = slice_ix * slice_width
				slice_end = slice_start + slice_width
				data_slice = inputs[:, slice_start:slice_end]
				result = getattr(self.model, self.call_target)(data_slice)
				results.append(result.T)
			return tensor.concatenate(results).T
		else:
			return [self(input) for input in inputs]

class SimpleLayerTransform(Model, Block):
	def __init__(self, n_visible, n_hidden, function):
		super(SimpleLayerTransform, self).__init__()
		self.n_hidden = n_hidden
		self.n_visible = n_visible
		self.function = getattr(tensor, function)
		self.W = sharedX(
			np.random.normal(size=(self.n_visible, self.n_hidden)),
			'W')
		self.b = sharedX(np.random.normal(self.n_hidden), 'b')
		self._params = [self.W, self.b]

	def get_input_space(self):
		return VectorSpace(self.n_visible)

	def get_output_dim(self):
		return VectorSpace(self.n_hidden)

	def __call__(self, inputs):
		if isinstance(inputs, tensor.Variable):
			return self.function(tensor.dot(inputs, self.W) + self.b)
		else:
			return [self(input) for input in inputs]


if __name__ == '__main__':
	def test_oh():
		vectorizer = OneHotVectorizerTransform(100, 10)
		input = tensor.as_tensor(np.random.random((100,)))
		tf = vectorizer(input)
		f = theano.function([], tf)

		assert f().shape == (10,), "Size was " + str(f().shape)

	def test_many_oh():
		vectorizer = OneHotVectorizerTransform(100, 10)
		multi = MultiAppliedModel(vectorizer, 5, '__call__')
		input = tensor.as_tensor(np.random.random((500,)))
		tf = multi(input)
		f = theano.function([], tf)
		assert f().shape == (50,), "Size was " + str(f().shape)

	def test_many_oh_ae():
		ae = Autoencoder(100, 10, 'sigmoid', 'tanh')
		multi = MultiAppliedModel(ae, 15, 'encode')
		input = tensor.as_tensor(np.random.random((1500,)))
		tf = multi(input)
		f = theano.function([], tf)
		assert f().shape == (150,), "Size was " + str(f().shape)

	test_oh()
	test_many_oh()
	test_many_oh_ae()
