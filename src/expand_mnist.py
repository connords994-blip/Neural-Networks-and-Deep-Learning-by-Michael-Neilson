"""expand_mnist.py
~~~~~~~~~~~~~~~~~~

Take the 50,000 MNIST training images, and create an expanded training
set by applying label-preserving transformations to each image:

  * displacement up, down, left and right by one pixel,
  * small rotations (clockwise and counter-clockwise),
  * small horizontal and vertical shears (skewing).

Save the resulting file to ../data/mnist_expanded.pkl.gz.

To keep memory use bounded, the expanded training set is *streamed* to
disk in shuffled blocks rather than being accumulated in RAM and written
in one shot.  Only ``BLOCK_PAIRS`` images are held in memory at a time.
The on-disk layout is a sequence of pickled objects:

  1. a header dict describing the stream,
  2. the validation data as ``[images, labels]``,
  3. the test data as ``[images, labels]``,
  4. one or more ``(X_block, Y_block)`` ndarray pairs of training data.

``network3.load_data_shared`` auto-detects this layout, so the older
single-pickle format is still readable too.  The number of variants
generated per image is controlled by the transformation lists below, so
you can trim them if you run out of memory.

"""

from __future__ import print_function

#### Libraries

# Standard library
import pickle
import gzip
import os.path

# Third-party libraries
import numpy as np
from scipy.ndimage import affine_transform

IMG_SIZE = 28

# Rotation angles in degrees.  Positive is counter-clockwise.
ROTATION_ANGLES = [-15, -10, -5, 5, 10, 15]

# Shear factors.  A shear of s maps a pixel at (row, col) so that the
# offset grows linearly across the image, producing a slanted digit.
SHEAR_FACTORS = [-0.3, -0.15, 0.15, 0.3]


def displaced_images(image):
    """Yield the four one-pixel displacements of ``image`` (up, down,
    left, right).  ``image`` is a 28x28 float array."""
    for d, axis, index_position, index in [
            (1,  0, "first", 0),
            (-1, 0, "first", IMG_SIZE - 1),
            (1,  1, "last",  0),
            (-1, 1, "last",  IMG_SIZE - 1)]:
        new_img = np.roll(image, d, axis)
        if index_position == "first":
            new_img[index, :] = np.zeros(IMG_SIZE)
        else:
            new_img[:, index] = np.zeros(IMG_SIZE)
        yield new_img


def _affine_about_center(image, matrix):
    """Apply the 2x2 affine ``matrix`` to ``image``, keeping the image
    centre fixed so the digit does not drift toward a corner."""
    center = (np.array(image.shape) - 1) / 2.0
    # affine_transform maps output coords -> input coords, so we offset
    # by center - matrix @ center to pivot about the middle pixel.
    offset = center - matrix.dot(center)
    return affine_transform(
        image, matrix, offset=offset, order=1, mode="constant", cval=0.0)


def rotated_images(image):
    """Yield rotations of ``image`` for each angle in ROTATION_ANGLES."""
    for angle in ROTATION_ANGLES:
        theta = np.deg2rad(angle)
        cos, sin = np.cos(theta), np.sin(theta)
        matrix = np.array([[cos, -sin], [sin, cos]])
        yield _affine_about_center(image, matrix)


def sheared_images(image):
    """Yield horizontal and vertical shears of ``image`` for each factor
    in SHEAR_FACTORS."""
    for s in SHEAR_FACTORS:
        # Horizontal shear: columns shift proportionally to the row.
        yield _affine_about_center(image, np.array([[1.0, s], [0.0, 1.0]]))
        # Vertical shear: rows shift proportionally to the column.
        yield _affine_about_center(image, np.array([[1.0, 0.0], [s, 1.0]]))


def expand(x):
    """Return a list of flattened (784,) variants of the flattened input
    image ``x``, including the original."""
    image = np.reshape(x, (IMG_SIZE, IMG_SIZE))
    variants = [image]
    variants.extend(displaced_images(image))
    variants.extend(rotated_images(image))
    variants.extend(sheared_images(image))
    return [np.reshape(v, IMG_SIZE * IMG_SIZE).astype(x.dtype) for v in variants]


# Number of expanded (x, y) pairs buffered in memory before being
# shuffled and flushed to disk as one block.  This bounds peak memory:
# only one block is held at a time rather than the whole expanded set.
BLOCK_PAIRS = 50000

# Identifies the streamed on-disk layout so ``load_data_shared`` can tell
# it apart from the legacy single-pickle format.
STREAM_FORMAT = "expanded-chunked-v1"

OUTPUT_PATH = "../data/mnist_expanded.pkl.gz"
SOURCE_PATH = "../data/mnist.pkl.gz"


def _as_pair(data):
    """Normalise a ``(images, labels)`` dataset into plain ndarrays."""
    return [np.asarray(data[0]), np.asarray(data[1])]


def main():
    print("Expanding the MNIST training set")

    if os.path.exists(OUTPUT_PATH):
        print("The expanded training set already exists.  Exiting.")
        return

    f = gzip.open(SOURCE_PATH, 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
    f.close()

    images, labels = training_data[0], training_data[1]
    variants_per_image = len(expand(images[0]))
    n_train = len(images) * variants_per_image
    print("Generating {0} images ({1} variants x {2} originals)".format(
        n_train, variants_per_image, len(images)))

    out = gzip.open(OUTPUT_PATH, "wb")
    # Header + val/test are written first; the training data then follows
    # as a sequence of independently pickled (X_block, Y_block) arrays.
    header = {"format": STREAM_FORMAT,
              "n_train": n_train,
              "n_features": IMG_SIZE * IMG_SIZE,
              "variants_per_image": variants_per_image}
    pickle.dump(header, out, protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(_as_pair(validation_data), out, protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(_as_pair(test_data), out, protocol=pickle.HIGHEST_PROTOCOL)

    buffer_x, buffer_y = [], []
    written = 0

    def flush():
        # Shuffle within the buffer and stream it out as one block.  The
        # MNIST training set is not label-sorted, so block-local shuffling
        # mixes labels well; SGD reshuffles each epoch besides.
        if not buffer_x:
            return 0
        block_x = np.asarray(buffer_x, dtype=np.float32)
        block_y = np.asarray(buffer_y, dtype=np.int64)
        perm = np.random.permutation(len(block_x))
        pickle.dump((block_x[perm], block_y[perm]), out,
                    protocol=pickle.HIGHEST_PROTOCOL)
        buffer_x.clear()
        buffer_y.clear()
        return len(block_x)

    for j, (x, y) in enumerate(zip(images, labels), start=1):
        for variant in expand(x):
            buffer_x.append(variant)
            buffer_y.append(y)
        if j % 1000 == 0:
            print("Expanding image number", j)
        if len(buffer_x) >= BLOCK_PAIRS:
            written += flush()
    written += flush()
    out.close()

    assert written == n_train, (written, n_train)
    print("Saved {0} expanded images to {1}".format(written, OUTPUT_PATH))


if __name__ == "__main__":
    main()
