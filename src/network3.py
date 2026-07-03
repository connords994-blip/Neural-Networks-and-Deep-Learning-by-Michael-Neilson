"""network3.py
~~~~~~~~~~~~~~

A PyTorch-based program for training and running simple neural
networks.

Supports several layer types (fully connected, convolutional, max
pooling, softmax), and activation functions (sigmoid, tanh, and
rectified linear units, with more easily added).

This is a port of the original Theano-based network3.py.  The public
API (the ``Network`` class, the layer classes, and ``load_data_shared``)
has been kept as close as possible to the original so that the
experiments in conv.py continue to work.  The program runs on a CPU.

Because the code is now based on PyTorch, the internals differ from the
original Theano version: instead of building a symbolic graph and
compiling ``theano.function``s, each layer is a ``torch.nn.Module`` and
training is an ordinary optimizer loop.  Note that I have focused on
making the code simple, easily readable, and easily modifiable.  It is
not optimized, and omits many desirable features.

"""

#### Libraries
# Standard library
import pickle
import gzip

# Third-party libraries
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Activation functions for neurons
def linear(z): return z
def ReLU(z): return F.relu(z)
sigmoid = torch.sigmoid
tanh = torch.tanh


#### Constants
# This version runs on a CPU.  (PyTorch can run on a GPU via ``.to("cuda")``,
# but GPU support has intentionally been left out to keep the code simple.)
print("Running network3.py on a CPU under PyTorch.")

#### Load the MNIST data
def _load_streamed(f, header):
    """Read the streamed (chunked) expanded-data layout written by
    ``expand_mnist.py``.  ``f`` is positioned just after ``header``.

    The training data is read block by block into a single preallocated
    array, so peak memory is the final array plus one block rather than a
    Python list of every image.
    """
    validation_data = pickle.load(f, encoding='latin1')
    test_data = pickle.load(f, encoding='latin1')
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
    return [train_x, train_y], validation_data, test_data

def load_data_shared(filename="../data/mnist.pkl.gz"):
    f = gzip.open(filename, 'rb')
    first = pickle.load(f, encoding='latin1')
    if isinstance(first, dict) and first.get("format") == "expanded-chunked-v1":
        training_data, validation_data, test_data = _load_streamed(f, first)
    else:
        # Legacy single-pickle format: the first object is the full tuple.
        training_data, validation_data, test_data = first
    f.close()
    def shared(data):
        """Place the data into tensors.

        """
        shared_x = torch.tensor(np.asarray(data[0]), dtype=torch.float32)
        shared_y = torch.tensor(np.asarray(data[1]), dtype=torch.long)
        return shared_x, shared_y
    return [shared(training_data), shared(validation_data), shared(test_data)]

#### Main class used to construct and train networks
class Network(nn.Module):

    def __init__(self, layers, mini_batch_size):
        """Takes a list of `layers`, describing the network architecture, and
        a value for the `mini_batch_size` to be used during training
        by stochastic gradient descent.

        """
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.mini_batch_size = mini_batch_size

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            validation_data, test_data, lmbda=0.0):
        """Train the network using mini-batch stochastic gradient descent."""
        training_x, training_y = training_data
        validation_x, validation_y = validation_data
        test_x, test_y = test_data

        # compute number of minibatches for training, validation and testing
        num_training_batches = size(training_data)//mini_batch_size
        num_validation_batches = size(validation_data)//mini_batch_size
        num_test_batches = size(test_data)//mini_batch_size

        # the (regularized) cost is the final layer's cost plus an L2 penalty
        optimizer = torch.optim.SGD(self.parameters(), lr=eta)

        # Do the actual training
        best_validation_accuracy = 0.0
        best_iteration = 0
        test_accuracy = 0.0
        for epoch in range(epochs):
            for minibatch_index in range(num_training_batches):
                iteration = num_training_batches*epoch+minibatch_index
                if iteration % 1000 == 0:
                    print("Training mini-batch number {0}".format(iteration))
                self.train()
                start = minibatch_index*mini_batch_size
                end = start+mini_batch_size
                x_mb, y_mb = training_x[start:end], training_y[start:end]
                optimizer.zero_grad()
                output = self(x_mb)
                l2_norm_squared = sum((layer.w**2).sum() for layer in self.layers)
                cost = self.layers[-1].cost(output, y_mb) \
                    + 0.5*lmbda*l2_norm_squared/num_training_batches
                cost.backward()
                optimizer.step()
                if (iteration+1) % num_training_batches == 0:
                    validation_accuracy = np.mean(
                        [self._accuracy(validation_x, validation_y, j, mini_batch_size)
                         for j in range(num_validation_batches)])
                    print("Epoch {0}: validation accuracy {1:.2%}".format(
                        epoch, validation_accuracy))
                    if validation_accuracy >= best_validation_accuracy:
                        print("This is the best validation accuracy to date.")
                        best_validation_accuracy = validation_accuracy
                        best_iteration = iteration
                        if test_data:
                            test_accuracy = np.mean(
                                [self._accuracy(test_x, test_y, j, mini_batch_size)
                                 for j in range(num_test_batches)])
                            print('The corresponding test accuracy is {0:.2%}'.format(
                                test_accuracy))
        print("Finished training network.")
        print("Best validation accuracy of {0:.2%} obtained at iteration {1}".format(
            best_validation_accuracy, best_iteration))
        print("Corresponding test accuracy of {0:.2%}".format(test_accuracy))

    def _accuracy(self, x, y, minibatch_index, mini_batch_size):
        """Return the accuracy on a single mini-batch, evaluated with the
        network in evaluation mode (so dropout is disabled)."""
        self.eval()
        start = minibatch_index*mini_batch_size
        end = start+mini_batch_size
        with torch.no_grad():
            output = self(x[start:end])
        return self.layers[-1].accuracy(output, y[start:end])

    def predict(self, x):
        """Return the predicted class for each input in the batch ``x``,
        with the network in evaluation mode."""
        self.eval()
        with torch.no_grad():
            output = self(x)
        return self.layers[-1].y_out(output)

#### Define layer types

class ConvPoolLayer(nn.Module):
    """Used to create a combination of a convolutional and a max-pooling
    layer.  A more sophisticated implementation would separate the
    two, but for our purposes we'll always use them together, and it
    simplifies the code, so it makes sense to combine them.

    """

    def __init__(self, filter_shape, image_shape, poolsize=(2, 2),
                 activation_fn=sigmoid):
        """`filter_shape` is a tuple of length 4, whose entries are the number
        of filters, the number of input feature maps, the filter height, and the
        filter width.

        `image_shape` is a tuple of length 4, whose entries are the
        mini-batch size, the number of input feature maps, the image
        height, and the image width.

        `poolsize` is a tuple of length 2, whose entries are the y and
        x pooling sizes.

        """
        super().__init__()
        self.filter_shape = filter_shape
        self.image_shape = image_shape
        self.poolsize = poolsize
        self.activation_fn = activation_fn
        # initialize weights and biases
        n_out = (filter_shape[0]*np.prod(filter_shape[2:])//np.prod(poolsize))
        self.w = nn.Parameter(torch.tensor(
            np.random.normal(loc=0, scale=np.sqrt(1.0/n_out), size=filter_shape),
            dtype=torch.float32))
        self.b = nn.Parameter(torch.tensor(
            np.random.normal(loc=0, scale=1.0, size=(filter_shape[0],)),
            dtype=torch.float32))

    def forward(self, inpt):
        inpt = inpt.view(-1, self.image_shape[1],
                         self.image_shape[2], self.image_shape[3])
        conv_out = F.conv2d(inpt, self.w)
        pooled_out = F.max_pool2d(conv_out, self.poolsize)
        return self.activation_fn(pooled_out + self.b.view(1, -1, 1, 1))

class FullyConnectedLayer(nn.Module):

    def __init__(self, n_in, n_out, activation_fn=sigmoid, p_dropout=0.0):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.activation_fn = activation_fn
        self.p_dropout = p_dropout
        # Initialize weights and biases
        self.w = nn.Parameter(torch.tensor(
            np.random.normal(loc=0.0, scale=np.sqrt(1.0/n_out), size=(n_in, n_out)),
            dtype=torch.float32))
        self.b = nn.Parameter(torch.tensor(
            np.random.normal(loc=0.0, scale=1.0, size=(n_out,)),
            dtype=torch.float32))

    def forward(self, inpt):
        inpt = inpt.reshape(-1, self.n_in)
        inpt = F.dropout(inpt, p=self.p_dropout, training=self.training)
        return self.activation_fn(torch.matmul(inpt, self.w) + self.b)

    def y_out(self, output):
        "Return the predicted class for each row of ``output``."
        return torch.argmax(output, dim=1)

    def accuracy(self, output, y):
        "Return the accuracy for the mini-batch."
        return torch.mean((self.y_out(output) == y).float()).item()

class SoftmaxLayer(nn.Module):

    def __init__(self, n_in, n_out, p_dropout=0.0):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.p_dropout = p_dropout
        # Initialize weights and biases
        self.w = nn.Parameter(torch.zeros((n_in, n_out), dtype=torch.float32))
        self.b = nn.Parameter(torch.zeros((n_out,), dtype=torch.float32))

    def forward(self, inpt):
        inpt = inpt.reshape(-1, self.n_in)
        inpt = F.dropout(inpt, p=self.p_dropout, training=self.training)
        # Return the (pre-softmax) logits; softmax is applied inside ``cost``
        # (via cross_entropy) and the class prediction is the argmax, which is
        # unchanged by the softmax.
        return torch.matmul(inpt, self.w) + self.b

    def cost(self, output, y):
        "Return the log-likelihood cost."
        return F.cross_entropy(output, y)

    def y_out(self, output):
        "Return the predicted class for each row of ``output``."
        return torch.argmax(output, dim=1)

    def accuracy(self, output, y):
        "Return the accuracy for the mini-batch."
        return torch.mean((self.y_out(output) == y).float()).item()


#### Miscellanea
def size(data):
    "Return the size of the dataset `data`."
    return data[0].shape[0]
