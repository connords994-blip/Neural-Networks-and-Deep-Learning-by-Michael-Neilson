"""
mnist_loader
~~~~~~~~~~~~

A library to load the MNIST image data.  For details of the data
structures that are returned, see the doc strings for ``load_data``
and ``load_data_wrapper``.  In practice, ``load_data_wrapper`` is the
function usually called by our neural network code.
"""

#### Libraries
# Standard library
import pickle
import gzip

# Third-party libraries
import numpy as np

def _load_streamed(f, header):
    """Read the streamed (chunked) layout written by ``expand_mnist.py``.

    ``f`` is a gzip file object positioned just after ``header``.  The
    training data is read block by block into a single preallocated
    array, so peak memory is the final array plus one block rather than
    a Python list of every image.  Returns data in the same
    ``(images, labels)`` shape as the legacy format.
    """
    validation_data = tuple(pickle.load(f, encoding='latin1'))
    test_data = tuple(pickle.load(f, encoding='latin1'))
    n, d = header["n_train"], header["n_features"]
    train_x = np.empty((n, d), dtype=np.float32)
    train_y = np.empty((n,), dtype=np.int64)
    offset = 0
    while offset < n:
        block_x, block_y = pickle.load(f, encoding='latin1')
        k = len(block_x)
        train_x[offset:offset+k] = block_x
        train_y[offset:offset+k] = block_y
        offset += k
    return (train_x, train_y), validation_data, test_data

def load_data(filename='../data/mnist.pkl.gz'):
    """Return the MNIST data as a tuple containing the training data,
    the validation data, and the test data.

    The ``training_data`` is returned as a tuple with two entries.
    The first entry contains the actual training images.  This is a
    numpy ndarray with 50,000 entries.  Each entry is, in turn, a
    numpy ndarray with 784 values, representing the 28 * 28 = 784
    pixels in a single MNIST image.

    The second entry in the ``training_data`` tuple is a numpy ndarray
    containing 50,000 entries.  Those entries are just the digit
    values (0...9) for the corresponding images contained in the first
    entry of the tuple.

    The ``validation_data`` and ``test_data`` are similar, except
    each contains only 10,000 images.

    ``filename`` may point either at the standard single-pickle file or
    at the streamed/chunked file produced by ``expand_mnist.py``; the
    format is auto-detected, so the expanded training set can be loaded
    the same way.

    This is a nice data format, but for use in neural networks it's
    helpful to modify the format of the ``training_data`` a little.
    That's done in the wrapper function ``load_data_wrapper()``, see
    below.
    """
    f = gzip.open(filename, 'rb')
    first = pickle.load(f, encoding='latin1')
    if isinstance(first, dict) and first.get("format") == "expanded-chunked-v1":
        data = _load_streamed(f, first)
    else:
        # Legacy single-pickle format: the first object is the full tuple.
        data = first
    f.close()
    return data

def load_data_wrapper(filename='../data/mnist.pkl.gz'):
    """Return a tuple containing ``(training_data, validation_data,
    test_data)``. Based on ``load_data``, but the format is more
    convenient for use in our implementation of neural networks.

    In particular, ``training_data`` is a list containing 50,000
    2-tuples ``(x, y)``.  ``x`` is a 784-dimensional numpy.ndarray
    containing the input image.  ``y`` is a 10-dimensional
    numpy.ndarray representing the unit vector corresponding to the
    correct digit for ``x``.

    ``validation_data`` and ``test_data`` are lists containing 10,000
    2-tuples ``(x, y)``.  In each case, ``x`` is a 784-dimensional
    numpy.ndarry containing the input image, and ``y`` is the
    corresponding classification, i.e., the digit values (integers)
    corresponding to ``x``.

    Obviously, this means we're using slightly different formats for
    the training data and the validation / test data.  These formats
    turn out to be the most convenient for use in our neural network
    code."""
    tr_d, va_d, te_d = load_data(filename)
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = list(zip(training_inputs, training_results))
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = list(zip(validation_inputs, va_d[1]))
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = list(zip(test_inputs, te_d[1]))
    return (training_data, validation_data, test_data)

def load_data_matrices(filename='../data/mnist.pkl.gz'):
    """Return ``(training_data, validation_data, test_data)`` with each
    set as a tuple ``(X, y)`` of *whole-dataset* matrices, suited to the
    vectorized networks (network4, network5).

    ``X`` has shape ``(784, n)`` -- one image per column -- as float32.
    ``y`` has shape ``(n,)`` and holds the integer digit labels.  Convert
    ``y`` to one-hot targets with ``one_hot`` when a training target
    matrix is needed.

    ``filename`` is forwarded to ``load_data``, so the expanded training
    set works here too."""
    tr_d, va_d, te_d = load_data(filename)
    def to_matrix(data):
        X = np.asarray(data[0], dtype=np.float32).T
        y = np.asarray(data[1]).astype(np.int64)
        return (X, y)
    return to_matrix(tr_d), to_matrix(va_d), to_matrix(te_d)

def one_hot(y, num_classes=10):
    """Return a ``(num_classes, len(y))`` one-hot matrix for the integer
    label vector ``y`` (one column per example)."""
    y = np.asarray(y)
    out = np.zeros((num_classes, y.shape[0]), dtype=np.float32)
    out[y, np.arange(y.shape[0])] = 1.0
    return out

def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
