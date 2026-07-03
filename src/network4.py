"""network4.py
~~~~~~~~~~~~~~

A fully *vectorized* feedforward network.  It keeps the same learning
ingredients as ``network2.py`` -- momentum-based stochastic gradient
descent with L2 regularization -- but instead of looping over the
examples in a mini-batch one at a time (as ``network.py`` and
``network2.py`` do in ``update_mini_batch``/``backprop``), it processes
the whole mini-batch as a single matrix.  Each activation is a matrix of
shape ``(units, batch_size)`` -- one example per column -- so a forward
or backward pass is a handful of ``numpy`` matrix products.

Compared to the earlier networks it also adds:

  * ReLU hidden units with a softmax + cross-entropy output layer
    (He-style weight initialization),
  * a **dropout schedule** and a **learning-rate schedule**, each of
    which may be given as a constant, a per-epoch list, or a callable
    ``f(epoch) -> value``.

Data is consumed in the matrix format produced by
``mnist_loader.load_data_matrices`` -- a tuple ``(X, y)`` where ``X`` is
``(784, n)`` and ``y`` is an integer label vector of length ``n``.

"""

#### Libraries
# Standard library
import json
import sys

# Third-party libraries
import numpy as np


def make_schedule(spec):
    """Normalize a schedule ``spec`` into a callable ``f(epoch) -> value``.

    ``spec`` may be a scalar (held constant), a list/tuple/array indexed
    by epoch (the last entry is reused once exhausted), or a callable
    which is returned unchanged."""
    if callable(spec):
        return spec
    if isinstance(spec, (list, tuple, np.ndarray)):
        seq = list(spec)
        return lambda epoch: seq[min(epoch, len(seq) - 1)]
    return lambda epoch: spec


#### Main Network class
class Network(object):

    def __init__(self, sizes):
        """``sizes`` lists the number of neurons in each layer, e.g.
        ``[784, 100, 10]``.  Hidden layers use ReLU activations and the
        output layer is a softmax; weights use He initialization
        (Gaussian scaled by ``sqrt(2 / n_in)``) which suits ReLUs, and
        biases start at zero."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.weights = [np.random.randn(y, x) * np.sqrt(2.0 / x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
        self.biases = [np.zeros((y, 1)) for y in sizes[1:]]
        self.velocities_w = [np.zeros(w.shape) for w in self.weights]
        self.velocities_b = [np.zeros(b.shape) for b in self.biases]

    def feedforward(self, X):
        """Return the softmax output matrix for a batch of inputs ``X``
        (shape ``(784, batch)``).  No dropout is applied -- this is the
        evaluation-time forward pass."""
        a = X
        last = len(self.weights) - 1
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = np.dot(w, a) + b
            a = softmax(z) if i == last else relu(z)
        return a

    def _forward(self, X, keep):
        """Training forward pass through ``X``, applying inverted dropout
        with keep-probability ``keep`` to the hidden activations.

        Returns ``(activations, zs, masks)`` where ``masks[i]`` is the
        dropout mask applied after weight-layer ``i`` (``None`` when no
        dropout was applied, e.g. the output layer)."""
        activations = [X]
        zs = []
        masks = []
        a = X
        last = len(self.weights) - 1
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = np.dot(w, a) + b
            zs.append(z)
            if i == last:
                a = softmax(z)
                masks.append(None)
            else:
                a = relu(z)
                if keep < 1.0:
                    mask = (np.random.rand(*a.shape) < keep) / keep
                    a = a * mask
                    masks.append(mask)
                else:
                    masks.append(None)
            activations.append(a)
        return activations, zs, masks

    def _backprop(self, activations, zs, masks, Y, n, lmbda):
        """Vectorized backprop for a mini-batch.  ``Y`` is the one-hot
        target matrix, ``n`` the total training-set size (for the L2
        term), and ``lmbda`` the regularization strength.  Returns the
        per-parameter gradients averaged over the mini-batch."""
        m = Y.shape[1]
        grad_w = [None] * len(self.weights)
        grad_b = [None] * len(self.biases)
        L = len(self.weights)
        # Softmax + cross-entropy: the output-layer error is just (a - y),
        # averaged over the mini-batch.
        delta = (activations[-1] - Y) / m
        grad_w[-1] = np.dot(delta, activations[-2].T) + (lmbda / n) * self.weights[-1]
        grad_b[-1] = np.sum(delta, axis=1, keepdims=True)
        for l in range(2, L + 1):
            da = np.dot(self.weights[-l + 1].T, delta)
            if masks[-l] is not None:
                da = da * masks[-l]
            delta = da * relu_prime(zs[-l])
            grad_w[-l] = np.dot(delta, activations[-l - 1].T) + (lmbda / n) * self.weights[-l]
            grad_b[-l] = np.sum(delta, axis=1, keepdims=True)
        return grad_w, grad_b

    def update_mini_batch(self, X, Y, eta, lmbda, mu, keep, n):
        """Apply one momentum-SGD step to the mini-batch ``(X, Y)``.
        ``eta`` is the (scheduled) learning rate, ``keep`` the dropout
        keep-probability, ``mu`` the momentum, and ``n`` the training-set
        size used by the L2 term."""
        activations, zs, masks = self._forward(X, keep)
        grad_w, grad_b = self._backprop(activations, zs, masks, Y, n, lmbda)
        self.velocities_w = [mu * v - eta * gw
                             for v, gw in zip(self.velocities_w, grad_w)]
        self.weights = [w + v for w, v in zip(self.weights, self.velocities_w)]
        self.velocities_b = [mu * v - eta * gb
                             for v, gb in zip(self.velocities_b, grad_b)]
        self.biases = [b + v for b, v in zip(self.biases, self.velocities_b)]

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            lmbda=0.0,
            mu=0.0,
            dropout=0.0,
            evaluation_data=None,
            monitor_evaluation_accuracy=False,
            monitor_training_cost=False):
        """Train with mini-batch momentum SGD.

        ``training_data`` is a tuple ``(X, y)`` of matrices (see
        ``mnist_loader.load_data_matrices``).  ``eta`` and ``dropout``
        are schedules: each may be a scalar, a per-epoch list, or a
        callable ``f(epoch) -> value`` (see ``make_schedule``).
        ``dropout`` is the drop probability applied to hidden layers, so
        a keep-probability of ``1 - dropout`` is used.  Returns a tuple
        ``(evaluation_accuracy, training_cost)`` of per-epoch lists (each
        empty unless the corresponding monitor flag is set)."""
        eta_schedule = make_schedule(eta)
        dropout_schedule = make_schedule(dropout)
        X, y = training_data
        Y = one_hot(y, self.sizes[-1])
        n = X.shape[1]

        evaluation_accuracy, training_cost = [], []
        for j in range(epochs):
            perm = np.random.permutation(n)
            X, Y = X[:, perm], Y[:, perm]
            eta_j = eta_schedule(j)
            keep_j = 1.0 - dropout_schedule(j)
            for k in range(0, n, mini_batch_size):
                self.update_mini_batch(
                    X[:, k:k + mini_batch_size], Y[:, k:k + mini_batch_size],
                    eta_j, lmbda, mu, keep_j, n)
            print("Epoch {0} complete (eta={1:.4g}, dropout={2:.2g})".format(
                j, eta_j, 1.0 - keep_j))
            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda)
                training_cost.append(cost)
                print("  training cost: {0:.4f}".format(cost))
            if monitor_evaluation_accuracy and evaluation_data is not None:
                acc = self.accuracy(evaluation_data)
                n_eval = evaluation_data[0].shape[1]
                evaluation_accuracy.append(acc)
                print("  evaluation accuracy: {0} / {1}".format(acc, n_eval))
        return evaluation_accuracy, training_cost

    def accuracy(self, data):
        """Number of correctly classified examples in ``data`` (a tuple
        ``(X, y)`` of matrices)."""
        X, y = data
        predictions = np.argmax(self.feedforward(X), axis=0)
        return int(np.sum(predictions == y))

    def total_cost(self, data, lmbda):
        """Mean cross-entropy cost over ``data`` plus the L2 penalty."""
        X, y = data
        m = X.shape[1]
        a = self.feedforward(X)
        Y = one_hot(y, self.sizes[-1])
        cost = -np.sum(Y * np.log(a + 1e-12)) / m
        cost += 0.5 * (lmbda / m) * sum(np.sum(w ** 2) for w in self.weights)
        return cost

    def save(self, filename):
        """Save the network's architecture and parameters to ``filename``."""
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases]}
        with open(filename, "w") as f:
            json.dump(data, f)


#### Loading a Network
def load(filename):
    """Load a network saved by ``Network.save`` and return it."""
    with open(filename, "r") as f:
        data = json.load(f)
    net = Network(data["sizes"])
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    net.velocities_w = [np.zeros(w.shape) for w in net.weights]
    net.velocities_b = [np.zeros(b.shape) for b in net.biases]
    return net


#### Miscellaneous functions
def one_hot(y, num_classes):
    """Return a ``(num_classes, len(y))`` one-hot matrix for the integer
    label vector ``y``."""
    y = np.asarray(y)
    out = np.zeros((num_classes, y.shape[0]), dtype=np.float64)
    out[y, np.arange(y.shape[0])] = 1.0
    return out

def relu(z):
    """The ReLU activation."""
    return np.maximum(0.0, z)

def relu_prime(z):
    """Derivative of the ReLU activation."""
    return (z > 0).astype(z.dtype)

def softmax(z):
    """Column-wise softmax, computed in a numerically stable way (each
    column is a separate example)."""
    z = z - np.max(z, axis=0, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=0, keepdims=True)
