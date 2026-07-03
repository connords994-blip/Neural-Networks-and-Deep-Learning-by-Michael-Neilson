# Neural Networks and Deep Learning — extended

This repository is **built on Michael Nielsen's
[Neural Networks and Deep Learning](https://github.com/mnielsen/neural-networks-and-deep-learning)**,
the code accompanying his book at
[neuralnetworksanddeeplearning.com](http://neuralnetworksanddeeplearning.com).
The original networks have been ported to **Python 3 / PyTorch** and the
repository has been extended with extra data augmentation, several
from-scratch networks, a Siamese one-shot classifier, and an experimental
target-propagation learner.

All original code and the book are by Michael Nielsen; this fork's
additions are described below. The project remains under the MIT License
(see the bottom of this file).

## Setup

```bash
pip install -r requirements.txt
```

MNIST ships in `data/mnist.pkl.gz`. The Siamese model additionally
downloads EMNIST (via `torchvision`) into `data/emnist/` and renders
reference glyphs from a system font (Arial/Helvetica). Large generated
artifacts (the expanded MNIST set, EMNIST, figures) are git-ignored.

## Original networks (Nielsen, ported to Python 3)

- `src/network.py` — basic feedforward net, SGD via backpropagation.
- `src/network2.py` — cross-entropy cost, L2 regularization, better weight
  initialization, momentum.
- `src/network3.py` — convolutional networks; **ported from Theano to
  PyTorch**.
- Supporting: `src/mnist_loader.py`, `src/mnist_svm.py`,
  `src/mnist_average_darkness.py`, `src/expand_mnist.py`.

## Additions in this fork

### Data pipeline
- **`src/expand_mnist.py`** — augments MNIST with **rotations and shear
  (skew)** in addition to the original one-pixel shifts (via SciPy affine
  transforms), and **streams** the ~950k-image expanded set to disk in
  shuffled blocks to keep memory bounded.
- **`src/mnist_loader.py`** — auto-detects the streamed/chunked expanded
  file; adds `load_data_matrices()` (whole-dataset matrices) and
  `one_hot()`. `network3.load_data_shared` auto-detects the format too.

### From-scratch networks (NumPy)
- **`src/network4.py`** — a fully **vectorized** MLP: each mini-batch is a
  single matrix instead of a Python loop over examples. ReLU + softmax,
  momentum, L2, plus **dropout and learning-rate schedules**. ~93% on a
  10k MNIST subset; backprop gradient-checked to ~1e-10.
- **`src/network5.py`** — a **batch-normalized** MLP (learned γ/β with
  running statistics) that trains stably at higher learning rates than
  network4 (~89% on a 10k subset in 3 epochs).

### Siamese one-shot classifier (PyTorch) — `src/siamese.py`
Trains a *same-glyph* verifier on MNIST-digit / typeface pairs, then
classifies **unseen EMNIST letters** one-shot: each query is matched
against font-rendered A–Z references (encode once, score all 26 with the
match head, take the argmax). References use per-case font **prototypes**
combined by **max across cases**.

- Pair accuracy **0.999**; **~0.29** EMNIST-letters one-shot accuracy
  (chance = 0.038).
- Embedding dropout is deliberately kept **off** — the two branches encode
  in separate passes, so independent dropout masks corrupt `|e1 − e2|` for
  true matches (it collapsed pair accuracy from 0.99 to 0.64).
- Training on the augmented set scored slightly *lower* than standard
  MNIST, because rotating/shearing the handwritten side moves it away from
  the upright typeface references.

### Target-propagation MLP (NumPy) — `src/network6.py`
An experiment in credit assignment **without cross-layer backpropagation**.
Each hidden layer is given a *target activation*, found by **derivative-free
random search with variance annealing** (a candidate activation is
forwarded to the output and scored by cross-entropy against the true
label); the layer's weights are then updated with a **local delta rule**.
A backprop trainer is included for comparison.

- The rule genuinely learns MNIST (~0.87 on a deep 3-hidden-layer net) but
  plateaus below backprop (~0.95). The limiter is target-search variance
  (which compounds with depth), not vanishing gradients — ReLU + He init
  keeps backprop healthy at this depth.

## License

MIT License

Copyright (c) 2012-2022 Michael Nielsen

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
