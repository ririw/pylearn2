"""
A module defining the Dataset class.
"""


class Dataset(object):

    """
    Abstract interface for Datasets.
    """

    def __iter__(self):
        """
        .. todo::

            WRITEME
        """
        return self.iterator()

    def iterator(self, mode=None, batch_size=None, num_batches=None,
                 rng=None, data_specs=None, return_tuple=False):
        """
        Return an iterator for this dataset with the specified
        behaviour. Unspecified values are filled-in by the default.

        Parameters
        ----------
        mode : str or object, optional
            One of 'sequential', 'random_slice', or 'random_uniform',
            *or* a class that instantiates an iterator that returns
            slices or index sequences on every call to next().
        batch_size : int, optional
            The size of an individual batch. Optional if `mode` is
            'sequential' and `num_batches` is specified (batch size
            will be calculated based on full dataset size).
        num_batches : int, optional
            The total number of batches. Unnecessary if `mode` is
            'sequential' and `batch_size` is specified (number of
            batches will be calculated based on full dataset size).
        rng : int, object or array_like, optional
            Either an instance of `numpy.random.RandomState` (or
            something with a compatible interface), or a seed value
            to be passed to the constructor to create a `RandomState`.
            See the docstring for `numpy.random.RandomState` for
            details on the accepted seed formats. If unspecified,
            defaults to using the dataset's own internal random
            number generator, which persists across iterations
            through the dataset and may potentially be shared by
            multiple iterator objects simultaneously (see "Notes"
            below).
        data_specs : (space, source) pair, optional
            `space` must be an instance of `Space` and `source` must be
            a string or tuple of string names such as 'features' or
            'targets'. The source names specify where the data will come
            from and the Space specifies its format.
            When source is a tuple, there are some additional requirements:

            * `space` must be a `CompositeSpace`, with one sub-space
              corresponding to each source name. i.e., the specification
              must be flat.
            * None of the components of `space` may be a `CompositeSpace`.
            * Each corresponding (sub-space, source name) pair must be
              unique, but the same source name may be mapped to many
              sub-spaces (for example if one part of the model is fully
              connected and expects a `VectorSpace`, while another part is
              convolutional and expects a `Conv2DSpace`).

            If `data_specs` is not provided, the behaviour (which
            sources will be present, in which order and space, or
            whether an Exception will be raised) is not defined and may
            depend on the implementation of each `Dataset`.
        return_tuple : bool, optional
            In case `data_specs` consists of a single space and source,
            if `return_tuple` is True, the returned iterator will return
            a tuple of length 1 containing the minibatch of the data
            at each iteration. If False, it will return the minibatch
            itself. This flag has no effect if data_specs is composite.
            Default: False.

        Returns
        -------
        iter_obj : object
            An iterator object implementing the standard Python
            iterator protocol (i.e. it has an `__iter__` method that
            return the object itself, and a `next()` method that
            returns results until it raises `StopIteration`).
            The `next()` method returns a batch containing data for
            each of the sources required in `data_specs`, in the requested
            `Space`.

        Notes
        -----
        Arguments are passed as instantiation parameters to classes
        that derive from `pylearn2.utils.iteration.SubsetIterator`.

        Iterating simultaneously with multiple iterator objects
        sharing the same random number generator could lead to
        difficult-to-reproduce behaviour during training. It is
        therefore *strongly recommended* that each iterator be given
        its own random number generator with the `rng` parameter
        in such situations.

        When it is valid to call the `iterator` method with the default
        value for all arguments, it makes it possible to use the `Dataset`
        itself as an Python iterator, with the default implementation of
        `Dataset.__iter__`. For instance, `DenseDesignMatrix` supports a
        value of `None` for `data_specs`.
        """
        # TODO: See how much of the logic from DenseDesignMatrix.iterator
        # can be handled here.
        raise NotImplementedError()

    def adjust_for_viewer(self, X):
        """
        Shift and scale a tensor, mapping its data range to [-1, 1].

        It makes it possible for the transformed tensor to be displayed
        with `pylearn2.gui.patch_viewer` tools.
        Default is to do nothing.

        Parameters
        ----------
        X: `numpy.ndarray`
            a tensor in the same space as the data

        Returns
        -------
        `numpy.ndarray`
            X shifted and scaled by a transformation that maps the data
            range to [-1, 1].

        Notes
        -----
        For example, for MNIST X will lie in [0,1] and the return value
        should be X*2-1
        """
        return X

    def has_targets(self):
        """ Returns true if the dataset includes targets """

        raise NotImplementedError()

    def get_topo_batch_axis(self):
        """
        Returns the index of the axis that corresponds to different examples
        in a batch when using topological_view.
        """

        # Subclasses that support topological view must implement this to
        # specify how their data is formatted.
        raise NotImplementedError()

    def get_batch_design(self, batch_size, include_labels=False):
        """
        Returns a randomly chosen batch of data formatted as a design
        matrix.

        This method is not guaranteed to have any particular properties
        like not repeating examples, etc. It is mostly useful for getting
        a single batch of data for a unit test or a quick-and-dirty
        visualization. Using this method for serious learning code is
        strongly discouraged. All code that depends on any particular
        example sampling properties should use Dataset.iterator.

        .. todo::

            Refactor to use `include_targets` rather than `include_labels`,
            to make the terminology more consistent with the rest of the
            library.

        Parameters
        ----------
        batch_size : int
            The number of examples to include in the batch.
        include_labels : bool
            If True, returns the targets for the batch, as well as the
            features.

        Returns
        -------
        batch : member of feature space, or member of (feature, target) space.
            Either numpy value of the features, or a (features, targets) tuple
            of numpy values, depending on the value of `include_labels`.
        """
        raise NotImplementedError(str(type(self)) + " does not implement "
                                  "get_batch_design.")

    def get_batch_topo(self, batch_size, include_labels=False):
        """
        Returns a topology-preserving batch of data.

        This method is not guaranteed to have any particular properties
        like not repeating examples, etc. It is mostly useful for getting
        a single batch of data for a unit test or a quick-and-dirty
        visualization. Using this method for serious learning code is
        strongly discouraged. All code that depends on any particular
        example sampling properties should use Dataset.iterator.

        .. todo::

            Refactor to use `include_targets` rather than `include_labels`,
            to make the terminology more consistent with the rest of the
            library.

        Parameters
        ----------
        batch_size : int
            The number of examples to include in the batch.
        include_labels : bool
            If True, returns the targets for the batch, as well as the
            features.

        Returns
        -------
        batch : member of feature space, or member of (feature, target) space.
            Either numpy value of the features, or a (features, targets) tuple
            of numpy values, depending on the value of `include_labels`.
        """
        raise NotImplementedError()

    def get_num_examples(self):
        """
        Returns the number of examples in the dataset

        Notes
        -----
        Infinite datasets have float('inf') examples.
        """
        raise NotImplementedError()
