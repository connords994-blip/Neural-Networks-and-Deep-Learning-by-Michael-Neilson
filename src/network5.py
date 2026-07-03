"""network5.py
~~~~~~~~~~~~~~

A vectorized feedforward network with **batch normalization** on the
hidden layers.  Like ``network4.py`` it processes a whole mini-batch as
a matrix (one example per column) and trains with momentum SGD, L2
regularization, and a learning-rate schedule.  The distinguishing
feature is that each hidden layer normalizes its pre-activations across
the mini-batch before the ReLU:

    z   = W a                      (no bias -- beta below plays its role)
    z_hat = (z - mu) / sqrt(var + eps)
    y   = gamma * z_hat + beta
    a   = relu(y)

``mu`` and ``var`` are the per-feature mean and variance over the
mini-batch during training; an exponential running average of them is
kept for use at evaluation time.  ``gamma`` and ``beta`` are learned.
The output layer is an ordinary softmax with a bias (no batch norm).

Batch normalization tends to stabilize and speed up training and makes
the network far less sensitive to the weight-initialization scale.

Data is consumed in the matrix format from
``mnist_loader.load_data_matrices``: a tuple ``(X, y)`` where ``X`` is
``(784, n)`` and ``y`` is an integer label vector.

"""

#### Libraries
# Standard library
import json

# Third-party libraries
import numpy as np


def make_schedule(spec):
    """Normalize a schedule ``spec`` into a callable ``f(epoch) -> value``.

    ``spec`` may be a scalar (constant), a list/tuple/array indexed by
    epoch (last entry reused once exhausted), or a callable (returned
    unchanged)."""
    if callable(spec):
        return spec
    if isinstance(spec, (list, tuple, np.ndarray)):
        seq = list(spec)
        return lambda epoch: seq[min(epoch, len(seq) - 1)]
    return lambda epoch: spec


#### Main Network class
class Network(object):

    def __init__(self, sizes, bn_momentum=0.9, bn_eps=1e-5):
        """``sizes`` lists the neuron counts per layer, e.g.
        ``[784, 100, 10]``.  Every hidden layer is batch-normalized and
        ReLU-activated; the output layer is a plain softmax.
        ``bn_momentum`` controls the exponential running average of the
        batch-norm statistics used at evaluation time."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.bn_momentum = bn_momentum
        self.bn_eps = bn_eps
        self.num_hidden = len(sizes) - 2  # batch-normalized layers

        # He-scaled weights.  Hidden layers carry no bias (batch norm's
        # beta shifts instead); the softmax output layer keeps its bias.
        self.weights = [np.random.randn(y, x) * np.sqrt(2.0 / x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
        self.out_bias = np.zeros((sizes[-1], 1))

        # Batch-norm parameters and running statistics, one set per
        # hidden layer.
        self.gammas = [np.ones((y, 1)) for y in sizes[1:-1]]
        self.betas = [np.zeros((y, 1)) for y in sizes[1:-1]]
        self.running_mean = [np.zeros((y, 1)) for y in sizes[1:-1]]
        self.running_var = [np.ones((y, 1)) for y in sizes[1:-1]]

        # Momentum buffers for every learnable parameter.
        self.v_w = [np.zeros(w.shape) for w in self.weights]
        self.v_out_bias = np.zeros(self.out_bias.shape)
        self.v_gamma = [np.zeros(g.shape) for g in self.gammas]
        self.v_beta = [np.zeros(b.shape) for b in self.betas]

    def feedforward(self, X):
        """Evaluation-time forward pass: batch-norm uses the stored
        running statistics rather than the batch's own."""
        a = X
        for i in range(self.num_hidden):
            z = np.dot(self.weights[i], a)
            z_hat = (z - self.running_mean[i]) / np.sqrt(self.running_var[i] + self.bn_eps)
            a = relu(self.gammas[i] * z_hat + self.betas[i])
        z = np.dot(self.weights[-1], a) + self.out_bias
        return softmax(z)

    def _forward(self, X):
        """Training forward pass.  Returns the softmax output and a list
        of per-hidden-layer caches needed by backprop, updating the
        running batch-norm statistics along the way."""
        a = X
        caches = []
        for i in range(self.num_hidden):
            z = np.dot(self.weights[i], a)
            mean = np.mean(z, axis=1, keepdims=True)
            var = np.var(z, axis=1, keepdims=True)
            std = np.sqrt(var + self.bn_eps)
            z_hat = (z - mean) / std
            y = self.gammas[i] * z_hat + self.betas[i]
            out = relu(y)
            caches.append((a, z, z_hat, std, y))
            # Update running statistics for evaluation.
            m = self.bn_momentum
            self.running_mean[i] = m * self.running_mean[i] + (1 - m) * mean
            self.running_var[i] = m * self.running_var[i] + (1 - m) * var
            a = out
        z = np.dot(self.weights[-1], a) + self.out_bias
        return softmax(z), a, caches

    def update_mini_batch(self, X, Y, eta, lmbda, mu, n):
        """One momentum-SGD step on mini-batch ``(X, Y)``.  ``Y`` is the
        one-hot target matrix, ``n`` the training-set size for L2."""
        m = Y.shape[1]
        output, last_hidden, caches = self._forward(X)

        # Output layer (softmax + cross-entropy).
        delta = (output - Y) / m
        grad_w = [None] * len(self.weights)
        grad_w[-1] = np.dot(delta, last_hidden.T) + (lmbda / n) * self.weights[-1]
        grad_out_bias = np.sum(delta, axis=1, keepdims=True)
        grad_gamma = [None] * self.num_hidden
        grad_beta = [None] * self.num_hidden

        # Backprop through the hidden (batch-normalized) layers.
        da = np.dot(self.weights[-1].T, delta)
        for i in reversed(range(self.num_hidden)):
            a_in, z, z_hat, std, y = caches[i]
            dy = da * relu_prime(y)
            grad_gamma[i] = np.sum(dy * z_hat, axis=1, keepdims=True)
            grad_beta[i] = np.sum(dy, axis=1, keepdims=True)
            # Backprop through the batch-norm transform (standard result).
            dz_hat = dy * self.gammas[i]
            dz = (1.0 / (m * std)) * (
                m * dz_hat
                - np.sum(dz_hat, axis=1, keepdims=True)
                - z_hat * np.sum(dz_hat * z_hat, axis=1, keepdims=True))
            grad_w[i] = np.dot(dz, a_in.T) + (lmbda / n) * self.weights[i]
            da = np.dot(self.weights[i].T, dz)

        # Momentum update for every parameter.
        self.v_w = [mu * v - eta * gw for v, gw in zip(self.v_w, grad_w)]
        self.weights = [w + v for w, v in zip(self.weights, self.v_w)]
        self.v_out_bias = mu * self.v_out_bias - eta * grad_out_bias
        self.out_bias = self.out_bias + self.v_out_bias
        self.v_gamma = [mu * v - eta * g for v, g in zip(self.v_gamma, grad_gamma)]
        self.gammas = [g + v for g, v in zip(self.gammas, self.v_gamma)]
        self.v_beta = [mu * v - eta * g for v, g in zip(self.v_beta, grad_beta)]
        self.betas = [b + v for b, v in zip(self.betas, self.v_beta)]

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            lmbda=0.0,
            mu=0.0,
            evaluation_data=None,
            monitor_evaluation_accuracy=False,
            monitor_training_cost=False):
        """Train with mini-batch momentum SGD.  ``training_data`` is a
        tuple ``(X, y)`` of matrices.  ``eta`` is a learning-rate
        schedule (scalar, per-epoch list, or callable).  Returns
        ``(evaluation_accuracy, training_cost)`` per-epoch lists."""
        eta_schedule = make_schedule(eta)
        X, y = training_data
        Y = one_hot(y, self.sizes[-1])
        n = X.shape[1]

        evaluation_accuracy, training_cost = [], []
        for j in range(epochs):
            perm = np.random.permutation(n)
            X, Y = X[:, perm], Y[:, perm]
            eta_j = eta_schedule(j)
            for k in range(0, n, mini_batch_size):
                self.update_mini_batch(
                    X[:, k:k + mini_batch_size], Y[:, k:k + mini_batch_size],
                    eta_j, lmbda, mu, n)
            print("Epoch {0} complete (eta={1:.4g})".format(j, eta_j))
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
        """Save architecture, parameters, and running batch-norm
        statistics to ``filename``."""
        data = {"sizes": self.sizes,
                "bn_momentum": self.bn_momentum,
                "bn_eps": self.bn_eps,
                "weights": [w.tolist() for w in self.weights],
                "out_bias": self.out_bias.tolist(),
                "gammas": [g.tolist() for g in self.gammas],
                "betas": [b.tolist() for b in self.betas],
                "running_mean": [r.tolist() for r in self.running_mean],
                "running_var": [r.tolist() for r in self.running_var]}
        with open(filename, "w") as f:
            json.dump(data, f)


#### Loading a Network
def load(filename):
    """Load a network saved by ``Network.save`` and return it."""
    with open(filename, "r") as f:
        data = json.load(f)
    net = Network(data["sizes"],
                  bn_momentum=data["bn_momentum"], bn_eps=data["bn_eps"])
    net.weights = [np.array(w) for w in data["weights"]]
    net.out_bias = np.array(data["out_bias"])
    net.gammas = [np.array(g) for g in data["gammas"]]
    net.betas = [np.array(b) for b in data["betas"]]
    net.running_mean = [np.array(r) for r in data["running_mean"]]
    net.running_var = [np.array(r) for r in data["running_var"]]
    net.v_w = [np.zeros(w.shape) for w in net.weights]
    net.v_out_bias = np.zeros(net.out_bias.shape)
    net.v_gamma = [np.zeros(g.shape) for g in net.gammas]
    net.v_beta = [np.zeros(b.shape) for b in net.betas]
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
    """Column-wise numerically stable softmax."""
    z = z - np.max(z, axis=0, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=0, keepdims=True)
