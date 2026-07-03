"""network6.py
~~~~~~~~~~~~~~

A feedforward MLP trained by a **target-propagation** rule instead of
end-to-end backpropagation.  This is an experiment in alternative credit
assignment: no gradient is propagated across layers.  Each layer is given
a *target activation* and updated locally to move toward it.

The rule, per mini-batch (top-down):

  1. Forward pass; store activations.
  2. Output layer: standard softmax/cross-entropy delta, standard weight
     step (this is the only place a "true" gradient is used, and only for
     the last layer).
  3. For each hidden layer, from the top down, *search* for a target
     activation -- the activation that would drive the (already-updated)
     downstream layers toward the correct output.  The search is
     derivative-free: sample K candidates from ``N(a, sigma^2)`` around the
     current activation, keep the best per example, shrink ``sigma``, and
     repeat (random search with variance annealing).
  4. Update that layer's weights with a **local delta rule**: one
     single-layer gradient step on ``||a - target||^2`` (uses only the
     layer's own derivative and its input -- no cross-layer backprop).

The search objective is *global*: a candidate activation is forwarded all
the way to the output through the (updated) downstream layers and scored
by the cross-entropy against the true label.  This keeps every layer's
target tied to the real objective.

A standard backprop trainer is included as a baseline reference.

Data uses the matrix format from ``mnist_loader.load_data_matrices``:
``(X, y)`` with ``X`` of shape ``(784, n)`` and integer labels ``y``.
"""

#### Libraries
# Standard library
# (none beyond numpy)

# Third-party libraries
import numpy as np

import mnist_loader


#### Main Network class
class Network(object):

    def __init__(self, sizes, seed=0):
        """``sizes`` lists neuron counts per layer, e.g. ``[784, 100, 10]``.
        Hidden layers are ReLU, the output is softmax; weights use He
        initialization and biases start at zero."""
        rng = np.random.default_rng(seed)
        self.sizes = sizes
        self.L = len(sizes) - 1  # number of weight layers
        self.weights = [rng.standard_normal((y, x)) * np.sqrt(2.0 / x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
        self.biases = [np.zeros((y, 1)) for y in sizes[1:]]

    #### Forward passes
    def _forward(self, X):
        """Full forward pass; returns ``(activations, zs)`` where
        ``activations[i]`` is the layer-``i`` activation (``activations[0]``
        is the input, ``activations[L]`` the softmax output) and ``zs[i]``
        the corresponding pre-activation (``zs[0]`` is ``None``)."""
        activations = [X]
        zs = [None]
        a = X
        for i in range(self.L):
            z = self.weights[i] @ a + self.biases[i]
            zs.append(z)
            a = softmax(z) if i == self.L - 1 else relu(z)
            activations.append(a)
        return activations, zs

    def feedforward(self, X):
        """Return the softmax output for a batch ``X``."""
        return self._forward(X)[0][-1]

    def _downstream_forward(self, a, l):
        """Forward activation ``a`` (the layer-``l`` activation) through
        weight layers ``l .. L-1`` to the softmax output."""
        for i in range(l, self.L):
            z = self.weights[i] @ a + self.biases[i]
            a = softmax(z) if i == self.L - 1 else relu(z)
        return a

    #### Target search (derivative-free, random search + variance annealing)
    def _search_target(self, a_center, l, y_onehot, rng, K, rounds, sigma0, gamma):
        """Search for a target activation for hidden layer ``l``.

        Starting from the current activation ``a_center`` (shape
        ``(H, m)``), each round samples ``K`` candidates from a Gaussian of
        width ``sigma`` around the running best, keeps the best candidate
        per example (by the global cross-entropy objective), then shrinks
        ``sigma`` by ``gamma``.  The current best is always included as an
        elite so the target never gets worse.  Candidates are clipped to
        ``>= 0`` (the ReLU output range)."""
        H, m = a_center.shape
        a_best = a_center.copy()
        sigma = sigma0
        for _ in range(rounds):
            noise = rng.standard_normal((K, H, m)) * sigma
            cand = a_best[None] + noise
            cand[0] = a_best                       # elitism
            np.maximum(cand, 0.0, out=cand)        # ReLU range
            flat = cand.transpose(1, 0, 2).reshape(H, K * m)  # (H, K*m)
            out = self._downstream_forward(flat, l).reshape(-1, K, m)  # (C,K,m)
            loss = -np.sum(y_onehot[:, None, :] * np.log(out + 1e-9), axis=0)
            best_k = np.argmin(loss, axis=0)                      # (m,)
            a_best = cand[best_k, :, np.arange(m)].T              # (H, m)
            sigma *= gamma
        return a_best

    #### Weight updates
    def update_mini_batch_targetprop(self, X, Y, eta, rng,
                                     K=64, rounds=5, sigma0=0.5, gamma=0.6):
        """One target-propagation step on mini-batch ``(X, Y)`` (``Y`` is
        the one-hot target matrix)."""
        acts, zs = self._forward(X)
        m = X.shape[1]

        # Output layer: standard cross-entropy gradient step.
        delta = (acts[self.L] - Y) / m
        self.weights[self.L - 1] -= eta * (delta @ acts[self.L - 1].T)
        self.biases[self.L - 1] -= eta * np.sum(delta, axis=1, keepdims=True)

        # Hidden layers, top-down: search a target, then local delta-rule step.
        for l in range(self.L - 1, 0, -1):
            t_l = self._search_target(acts[l], l, Y, rng, K, rounds, sigma0, gamma)
            # Local step on ||a_l - t_l||^2 through this layer's own ReLU only.
            e = (acts[l] - t_l) * relu_prime(zs[l])
            self.weights[l - 1] -= eta * ((e @ acts[l - 1].T) / m)
            self.biases[l - 1] -= eta * np.mean(e, axis=1, keepdims=True)

    def update_mini_batch_backprop(self, X, Y, eta):
        """One standard backprop step -- baseline for comparison."""
        acts, zs = self._forward(X)
        m = X.shape[1]
        delta = (acts[self.L] - Y) / m
        grads_w = [None] * self.L
        grads_b = [None] * self.L
        grads_w[self.L - 1] = delta @ acts[self.L - 1].T
        grads_b[self.L - 1] = np.sum(delta, axis=1, keepdims=True)
        for l in range(self.L - 1, 0, -1):
            delta = (self.weights[l].T @ delta) * relu_prime(zs[l])
            grads_w[l - 1] = delta @ acts[l - 1].T
            grads_b[l - 1] = np.sum(delta, axis=1, keepdims=True)
        self.weights = [w - eta * gw for w, gw in zip(self.weights, grads_w)]
        self.biases = [b - eta * gb for b, gb in zip(self.biases, grads_b)]

    #### Training
    def SGD(self, training_data, epochs, mini_batch_size, eta,
            method="targetprop", evaluation_data=None,
            K=64, rounds=5, sigma0=0.5, gamma=0.6, lr_decay=1.0, seed=0):
        """Train with ``method`` in {``"targetprop"``, ``"backprop"``}.

        The learning rate decays geometrically each epoch by ``lr_decay``
        (``eta_epoch = eta * lr_decay ** epoch``); ``lr_decay=1.0`` keeps it
        fixed.  Returns the list of per-epoch evaluation accuracies."""
        rng = np.random.default_rng(seed)
        X, y = training_data
        Y = one_hot(y, self.sizes[-1])
        n = X.shape[1]
        history = []
        for epoch in range(epochs):
            eta_epoch = eta * (lr_decay ** epoch)
            perm = rng.permutation(n)
            X, Y = X[:, perm], Y[:, perm]
            for k in range(0, n, mini_batch_size):
                xb, yb = X[:, k:k + mini_batch_size], Y[:, k:k + mini_batch_size]
                if method == "targetprop":
                    self.update_mini_batch_targetprop(
                        xb, yb, eta_epoch, rng, K, rounds, sigma0, gamma)
                else:
                    self.update_mini_batch_backprop(xb, yb, eta_epoch)
            if evaluation_data is not None:
                acc = self.accuracy(evaluation_data)
                n_eval = evaluation_data[0].shape[1]
                history.append(acc / n_eval)
                print("  epoch {0:2d}: eval accuracy {1}/{2} = {3:.3f} (eta {4:.3g})".format(
                    epoch, acc, n_eval, acc / n_eval, eta_epoch))
        return history

    def accuracy(self, data):
        """Number of correctly classified examples in ``data`` (``(X, y)``)."""
        X, y = data
        return int(np.sum(np.argmax(self.feedforward(X), axis=0) == y))


#### Miscellaneous functions
def one_hot(y, num_classes):
    y = np.asarray(y)
    out = np.zeros((num_classes, y.shape[0]))
    out[y, np.arange(y.shape[0])] = 1.0
    return out

def relu(z):
    return np.maximum(0.0, z)

def relu_prime(z):
    return (z > 0).astype(z.dtype)

def softmax(z):
    z = z - np.max(z, axis=0, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=0, keepdims=True)


#### Demo: backprop vs target prop on a DEEP net, side by side
def main():
    tr, va, te = mnist_loader.load_data_matrices()
    # Subset for a quick comparison (target-prop search is compute-heavy).
    train = (tr[0][:, :10000], tr[1][:10000])
    val = (va[0][:, :2000], va[1][:2000])
    sizes = [784, 128, 128, 128, 10]  # deep: 3 hidden layers
    epochs = 12

    print("Deep net {0}".format(sizes))
    results = {}
    for method, label in [("backprop", "backprop baseline"),
                          ("targetprop", "target prop (global, lr decay)")]:
        print("\n=== {0} ===".format(label))
        net = Network(sizes, seed=0)
        results[label] = net.SGD(train, epochs, 64, 0.1, method=method,
                                 evaluation_data=val, lr_decay=0.9, seed=0)

    print("\n=== side-by-side eval accuracy per epoch ===")
    print("epoch | backprop | target-prop")
    bp, tp = list(results.values())
    for e in range(epochs):
        print("  {0:2d}  |  {1:.3f}   |   {2:.3f}".format(e, bp[e], tp[e]))


if __name__ == "__main__":
    main()
