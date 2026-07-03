"""siamese.py
~~~~~~~~~~~~

A Siamese convolutional network for *one-shot* character classification.

The idea
--------
Instead of learning "this image is the digit 7", the network learns a
class-agnostic verifier: given two glyph images it outputs the
probability that they depict the **same** character.  Because that skill
is about sameness rather than about specific classes, it transfers to
character classes the network was never trained on.

We exploit that here:

  * **Train** the verifier on MNIST *digits*.  Each training pair is a
    handwritten MNIST digit and a *rendered typeface* digit (Arial); the
    label is 1 if they are the same digit, 0 otherwise.  Pairing
    handwriting against a typeface template mirrors the test setup, so
    the network learns to bridge that domain gap during training.

  * **Test** on EMNIST *letters* (classes disjoint from digits).  To turn
    the yes/no verifier into a 26-way classifier with no letter training,
    we render a typeface reference for each letter A-Z and, for a query
    image, score the match probability against every reference and take
    the argmax.  Reference embeddings are cached, so a query costs one
    encoder pass plus 26 cheap head evaluations.  Averaging the match
    probability over several reference fonts (a reference-side ensemble)
    can sharpen the decision.

Architecture
------------
A shared-weight CNN encoder maps a 1x28x28 image to a 128-d embedding
(two conv+BN+ReLU+pool blocks then a linear layer).  The match head takes
the element-wise |e1 - e2| and maps it through a linear layer to a single
logit, trained with binary cross-entropy (Koch et al., 2015).

Run ``python3 siamese.py`` to train and evaluate.
"""

#### Libraries
# Standard library
import os

# Third-party libraries
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageFont, ImageDraw
from torchvision.datasets import EMNIST

import mnist_loader

#### Configuration
IMG_SIZE = 28
EMBED_DIM = 256
MNIST_PATH = "../data/mnist.pkl.gz"
EXPANDED_PATH = "../data/mnist_expanded.pkl.gz"
EMNIST_ROOT = "../data/emnist"
DEFAULT_FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"
# A few extra fonts for the optional reference-side ensemble.
ENSEMBLE_FONTS = [
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
]
DIGITS = [str(d) for d in range(10)]
LETTERS = [chr(ord("A") + i) for i in range(26)]


def get_device():
    """Prefer CUDA, then Apple MPS, else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


#### Rendering typeface glyphs -------------------------------------------------
def render_glyph(ch, font_path=DEFAULT_FONT, size=IMG_SIZE, font_px=22):
    """Render character ``ch`` white-on-black, centered in a ``size`` x
    ``size`` image, roughly matching MNIST's scale and framing.  Returns
    a float32 array in [0, 1]."""
    font = ImageFont.truetype(font_path, font_px)
    measure = ImageDraw.Draw(Image.new("L", (size * 3, size * 3), 0))
    bbox = measure.textbbox((0, 0), ch, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    img = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(img)
    x = (size - w) / 2 - bbox[0]
    y = (size - h) / 2 - bbox[1]
    draw.text((x, y), ch, fill=255, font=font)
    return np.asarray(img, dtype=np.float32) / 255.0

def render_glyph_tensor(chars, font_path=DEFAULT_FONT):
    """Render a list of characters into a ``(len(chars), 1, 28, 28)`` tensor."""
    imgs = [render_glyph(ch, font_path) for ch in chars]
    arr = np.stack(imgs)[:, None, :, :]
    return torch.from_numpy(arr)


#### Data ----------------------------------------------------------------------
def load_mnist_digits(filename=MNIST_PATH):
    """Return ``(X, y)`` for the MNIST training set as a float32 tensor of
    shape ``(n, 1, 28, 28)`` and an int64 label tensor.  ``filename`` may be
    the standard MNIST file or the augmented ``mnist_expanded.pkl.gz`` (which
    ``mnist_loader.load_data`` auto-detects)."""
    tr, _, _ = mnist_loader.load_data(filename)
    X = torch.from_numpy(np.asarray(tr[0], dtype=np.float32)).reshape(-1, 1, IMG_SIZE, IMG_SIZE)
    y = torch.from_numpy(np.asarray(tr[1]).astype(np.int64))
    return X, y

def load_emnist_letters(train=False):
    """Return ``(X, y)`` for the EMNIST letters split.  Images are
    transposed to match MNIST orientation and scaled to [0, 1]; labels are
    remapped from EMNIST's 1..26 to 0..25 (A..Z)."""
    ds = EMNIST(root=EMNIST_ROOT, split="letters", train=train, download=False)
    # ds.data is (N, 28, 28) uint8, stored transposed vs MNIST.
    data = ds.data.numpy().astype(np.float32) / 255.0
    data = np.transpose(data, (0, 2, 1))  # fix orientation
    X = torch.from_numpy(data)[:, None, :, :]
    y = ds.targets.numpy().astype(np.int64) - 1  # 1..26 -> 0..25
    return X, torch.from_numpy(y)


#### Model ---------------------------------------------------------------------
class Encoder(nn.Module):
    """Shared CNN mapping a 1x28x28 image to a ``EMBED_DIM``-d embedding."""

    def __init__(self, embed_dim=EMBED_DIM, normalize=False, dropout=0.0):
        super().__init__()
        # Two conv blocks of two 3x3 convs each (VGG-style), widening
        # 1 -> 64 -> 128 channels, each block halving the spatial size.
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),                                     # 14x14
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),                                     # 7x7
        )
        # NOTE: dropout defaults to 0 on purpose.  In a Siamese network the
        # two branches are encoded in *separate* forward passes, so dropout on
        # the embedding gives them independent masks -- even identical inputs
        # then yield different embeddings, corrupting |e1 - e2| for true
        # matches.  Empirically, dropout>0 here collapses pair accuracy from
        # ~0.99 to ~0.64.  Leave at 0 (regularize via BN / data instead).
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, embed_dim), nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.normalize = normalize

    def forward(self, x):
        e = self.fc(self.features(x))
        if self.normalize:
            # Project embeddings onto the unit hypersphere so comparisons
            # depend on direction (cosine geometry), not vector magnitude.
            e = nn.functional.normalize(e, p=2, dim=-1)
        return e


class SiameseNet(nn.Module):
    """Twin encoder + a weighted-L1 match head producing a single logit."""

    def __init__(self, embed_dim=EMBED_DIM, normalize=False):
        super().__init__()
        self.encoder = Encoder(embed_dim, normalize=normalize)
        self.head = nn.Linear(embed_dim, 1)

    def encode(self, x):
        return self.encoder(x)

    def match_logit(self, e1, e2):
        """Match logit from two (batches of) embeddings."""
        return self.head(torch.abs(e1 - e2)).squeeze(-1)

    def forward(self, x1, x2):
        return self.match_logit(self.encode(x1), self.encode(x2))


#### Training ------------------------------------------------------------------
def digit_index(mnist_y):
    """Return a list of index tensors, one per digit 0..9, giving the
    positions of that digit in the training set (for sampling handwritten
    exemplars of a chosen digit)."""
    return [(mnist_y == d).nonzero(as_tuple=True)[0] for d in range(10)]

def _random_images_for(mnist_X, idx_by_digit, digits, generator):
    """For each label in ``digits``, pick a random MNIST image of that digit."""
    out = torch.empty((digits.shape[0], 1, IMG_SIZE, IMG_SIZE))
    for d in range(10):
        mask = digits == d
        k = int(mask.sum())
        if k == 0:
            continue
        pool = idx_by_digit[d]
        pick = pool[torch.randint(0, pool.numel(), (k,), generator=generator)]
        out[mask] = mnist_X[pick]
    return out

def sample_pairs(mnist_X, mnist_y, idx_by_digit, digit_glyphs, batch_size,
                 device, generator, hw_pair_frac=0.5):
    """Draw a balanced batch of digit pairs.

    The first branch is always a handwritten MNIST digit.  For the second
    branch, a fraction ``hw_pair_frac`` of the batch uses another
    *handwritten* MNIST digit (handwriting<->handwriting, which teaches
    style-invariant glyph features), while the rest use the *typeface*
    rendering (handwriting<->typeface, which mirrors test time).  Half of
    each are positives (same digit)."""
    n = mnist_X.shape[0]
    idx = torch.randint(0, n, (batch_size,), generator=generator)
    a_img = mnist_X[idx]
    a_digit = mnist_y[idx]
    match = torch.randint(0, 2, (batch_size,), generator=generator).bool()
    # Second-branch digit: same when matching, else a guaranteed-different digit.
    offset = torch.randint(1, 10, (batch_size,), generator=generator)
    b_digit = torch.where(match, a_digit, (a_digit + offset) % 10)

    typeface_b = digit_glyphs[b_digit]
    handwritten_b = _random_images_for(mnist_X, idx_by_digit, b_digit, generator)
    use_hw = (torch.rand(batch_size, generator=generator) < hw_pair_frac)
    b_img = torch.where(use_hw[:, None, None, None], handwritten_b, typeface_b)
    return a_img.to(device), b_img.to(device), match.float().to(device)

def train(net, mnist_X, mnist_y, device, epochs=15, batch_size=128,
          steps_per_epoch=600, lr=1e-3, seed=0, hw_pair_frac=0.15,
          use_scheduler=True):
    """Train the verifier on mixed handwriting/typeface digit pairs with BCE.

    ``hw_pair_frac`` sets the share of handwriting<->handwriting pairs.  A
    small fraction (~0.15) modestly helps letter transfer by encouraging
    style-invariant features; large fractions hurt, because the test task is
    always handwriting-vs-typeface and too many hw<->hw pairs pull training
    away from that distribution.

    When ``use_scheduler`` is set, the learning rate follows a cosine
    annealing schedule from ``lr`` down to ~0 over the ``epochs`` epochs,
    which typically gives a cleaner final convergence than a fixed rate."""
    digit_glyphs = render_glyph_tensor(DIGITS)  # (10, 1, 28, 28)
    idx_by_digit = digit_index(mnist_y)
    generator = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
                 if use_scheduler else None)
    criterion = nn.BCEWithLogitsLoss()
    net.to(device).train()
    for epoch in range(epochs):
        running_loss, correct, total = 0.0, 0, 0
        for _ in range(steps_per_epoch):
            hw, ref, label = sample_pairs(
                mnist_X, mnist_y, idx_by_digit, digit_glyphs, batch_size,
                device, generator, hw_pair_frac)
            optimizer.zero_grad()
            logit = net(hw, ref)
            loss = criterion(logit, label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            correct += ((logit > 0).float() == label).sum().item()
            total += label.numel()
        current_lr = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            scheduler.step()
        print("Epoch {0:2d}: pair loss {1:.4f}, pair acc {2:.3f} (lr {3:.2g})".format(
            epoch, running_loss / steps_per_epoch, correct / total, current_lr))
    return net


#### One-shot classification ---------------------------------------------------
def letter_variants(fonts=(DEFAULT_FONT,), include_lower=True):
    """Build the list of reference *variants* for the 26 letter classes.

    Each variant is a ``(chars, font)`` pair where ``chars[c]`` is the glyph
    string for class ``c`` (index 0->'A').  EMNIST's ``letters`` split merges
    upper- and lower-case into one class, so rendering both cases -- and
    several fonts -- gives multiple canonical exemplars per class, i.e. a
    reference-side ensemble."""
    upper = [chr(ord("A") + i) for i in range(26)]
    lower = [chr(ord("a") + i) for i in range(26)]
    cases = [upper, lower] if include_lower else [upper]
    return [(chars, font) for chars in cases for font in fonts]

def reference_embeddings(net, variants, device):
    """Return cached reference embeddings of shape ``(R, C, EMBED_DIM)`` --
    one embedding per (variant, class) for the ``R`` variants in
    ``variants``.  These never change, so they are computed once and reused
    for every query."""
    net.eval()
    embs = []
    with torch.no_grad():
        for chars, font in variants:
            glyphs = render_glyph_tensor(chars, font).to(device)
            embs.append(net.encode(glyphs))
    return torch.stack(embs)  # (R, C, D)

@torch.no_grad()
def classify(net, queries, ref_embs, device, batch_size=512, agg="max"):
    """Predict a class index for each query by comparing it to the ``R``
    reference variants of each class and taking the argmax over classes.

    ``ref_embs`` has shape ``(R, C, D)``.  ``agg`` sets how the ``R``
    variants of a class are combined:

      * ``"proto"`` -- **per-class exemplar averaging**: average the ``R``
        variant *embeddings* into one prototype per class, then score the
        query once against each prototype.  Denoises exemplar quirks; best
        for near-identical variants (e.g. fonts).
      * ``"max"`` -- score each variant separately, keep the best match
        probability.  Best when variants are genuinely different shapes
        (e.g. upper- vs lower-case), where a single centroid is meaningless.
      * ``"mean"`` -- average the per-variant probabilities."""
    net.eval()
    if agg == "proto":
        # Collapse variants to one prototype embedding per class up front.
        ref_embs = ref_embs.mean(dim=0, keepdim=True)  # (1, C, D)
    preds = []
    for start in range(0, queries.shape[0], batch_size):
        q = queries[start:start + batch_size].to(device)
        qe = net.encode(q)                                             # (B, D)
        diff = torch.abs(qe[:, None, None, :] - ref_embs[None, :, :, :])  # (B,R,C,D)
        prob = torch.sigmoid(net.head(diff).squeeze(-1))              # (B, R, C)
        prob = prob.max(dim=1).values if agg == "max" else prob.mean(dim=1)
        preds.append(prob.argmax(dim=1).cpu())
    return torch.cat(preds)

def case_group_embeddings(net, fonts=(DEFAULT_FONT,), include_lower=True,
                          device="cpu"):
    """Return one *prototype* embedding per case per letter class, shape
    ``(G, C, D)`` with ``G`` in {1, 2}.

    Within a case, the references differ only by font -- near-identical
    shapes -- so we average their embeddings into a single prototype
    (per-class exemplar averaging).  The two cases ('A' vs 'a') are
    genuinely different shapes, so they are kept as separate groups to be
    combined by a *max* in ``classify`` -- i.e. average across fonts within
    a case, then take the best-matching case."""
    upper = [chr(ord("A") + i) for i in range(26)]
    lower = [chr(ord("a") + i) for i in range(26)]
    cases = [upper, lower] if include_lower else [upper]
    net.eval()
    groups = []
    with torch.no_grad():
        for chars in cases:
            font_embs = [net.encode(render_glyph_tensor(chars, f).to(device))
                         for f in fonts]
            groups.append(torch.stack(font_embs).mean(dim=0))  # (C, D) prototype
    return torch.stack(groups)  # (G, C, D)

def evaluate(net, X, y, device, fonts=(DEFAULT_FONT,), include_lower=True):
    """One-shot classification accuracy on ``(X, y)`` using the recommended
    aggregation: font prototypes per case, combined by max across cases."""
    protos = case_group_embeddings(net, fonts, include_lower, device)
    preds = classify(net, X, protos, device, agg="max")
    return (preds == y).float().mean().item()


#### Main ----------------------------------------------------------------------
def main():
    device = get_device()
    print("Using device:", device)
    # Standard MNIST trains best for this task.  The augmented
    # EXPANDED_PATH set was measured ~2 points *lower* (0.263 vs 0.286):
    # rotating/shearing the handwritten side moves it away from the upright
    # typeface references we match against at test time.  Swap to
    # EXPANDED_PATH here if you want to experiment with it.
    train_path = MNIST_PATH
    print("Loading training digits from", train_path, "...")
    mnist_X, mnist_y = load_mnist_digits(train_path)
    print("  training images:", mnist_X.shape[0])
    print("Loading EMNIST letters (evaluation) ...")
    emnist_X, emnist_y = load_emnist_letters(train=False)

    net = SiameseNet()
    print("Training verifier on MNIST-digit / typeface pairs ...")
    train(net, mnist_X, mnist_y, device)

    fonts = [f for f in ENSEMBLE_FONTS if os.path.exists(f)]
    print("\nEMNIST letters one-shot accuracy (chance = {0:.3f}):".format(1 / 26))
    acc = evaluate(net, emnist_X, emnist_y, device,
                   fonts=(DEFAULT_FONT,), include_lower=False)
    print("  uppercase, Arial:                       {0:.3f}".format(acc))
    acc = evaluate(net, emnist_X, emnist_y, device,
                   fonts=fonts, include_lower=False)
    print("  uppercase, {0}-font prototype:            {1:.3f}".format(len(fonts), acc))
    acc = evaluate(net, emnist_X, emnist_y, device,
                   fonts=fonts, include_lower=True)
    print("  upper+lower, {0}-font proto + case-max:   {1:.3f}".format(len(fonts), acc))


if __name__ == "__main__":
    main()
