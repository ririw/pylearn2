from theano import tensor
import theano.sparse
import numpy as np
from pylearn2 import corruption
from pylearn2.costs.cost import Cost, DefaultDataSpecsMixin
from theano.tensor.shared_randomstreams import RandomStreams
from pylearn2.utils import sharedX


class CorruptingWordVecCost(DefaultDataSpecsMixin, Cost):
	supervised = False

	def __init__(self, n_hidden, n_visible, n_windows):
		self.U = sharedX(np.random.normal(size=(n_hidden)), 'U')
		corruption_start = n_visible * (n_windows / 2)
		corruption_end = n_visible * (n_windows / 2 + 1)
		self.corruptor = corruption.SubwindowCorruptor(corruption.SaltPepperCorruptor(5), corruption_start, corruption_end)


	def expr(self, model, data, *args, **kwargs):
		space, source = self.get_data_specs(model)
		space.validate(data)

		inputs = data
		outputs = model(inputs)[-1]
		uncorrupted = tensor.dot(outputs, self.U)

		corrupted_inputs = self.corruptor(data)
		corrupted_outputs = model(corrupted_inputs)[-1]
		corrupted = tensor.dot(corrupted_outputs, self.U)

		zeros = tensor.zeros_like(uncorrupted)
		loss = tensor.sum(tensor.largest(0, 1 - uncorrupted + corrupted))
		#loss = tensor.sum(inputs - corrupted_inputs)

		return loss

