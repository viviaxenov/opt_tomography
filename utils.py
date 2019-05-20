import os
import numpy as np
from scipy import special
from scipy.sparse import coo_matrix, csr
from skimage import io, img_as_ubyte
from skimage.color import rgb2gray


def generate_stochastic_matrix(
        area: np.ndarray,
        detectors_x: int,
        detectors_y: int
):
    """
    :param area: basis for stoch.matrix
    :param detectors_x: the amount of detectors along fixed x axis
    :param detectors_y: the amount of detectors along fixed y axis
    :return: scipy sparse coord matrix
    of size (area.shape[0] * area.shape[1], detectors_x * detectors_y)
    Generates a row-stochastic matrix based on 2d numpy array specified in 'area'.
    It is considered that area is covered with detectors_x * detectors_y detectors
    placed equidistantly over the area, having a chance to detect a particle within area
    corresponding to 2d normal
    distribution with dispersion comparable with distance between them

    A element of matrix indexed (i, j) stands for the probability for detector i to catch
    a particle received by the j-th cell of the area
    """

    sources = detectors_x * detectors_y
    h_x = 1.0 / (detectors_x - 1)
    h_y = 1.0 / (detectors_y - 1)
    # optional, may be changed
    dispersion_multiplier = 1  # square root of dispersion of sources / distance between sources

    distr_along_x = np.zeros(2 * area.shape[1])
    distr_along_y = np.zeros(2 * area.shape[0])
    for i in range(-dispersion_multiplier, dispersion_multiplier):
        distr_along_x[area.shape[1] + i] = \
            special.erf((i + 1) / (2 ** 0.5)) - special.erf(i / (2 ** 0.5))

    for i in range(-dispersion_multiplier, dispersion_multiplier):
        distr_along_y[area.shape[0] + i] = \
            special.erf((i + 1) / (2 ** 0.5)) - special.erf(i / (2 ** 0.5))

    stochastic_matrix = np.array([])

    for i in range(detectors_y):
        for j in range(detectors_x):

            source_prob_matrix = np.zeros(area.shape)
            offset_x = int(round(area.shape[1] * h_x * j))
            offset_y = int(round(area.shape[0] * h_y * i))
            for l in range(-2, 2):
                for m in range(-2, 2):
                    if (((offset_y + m) < 0) or
                            ((offset_y + m) > area.shape[0] - 1) or
                            ((offset_x + l) < 0) or
                            ((offset_x + l) > area.shape[1] - 1)
                    ):
                        continue
                    source_prob_matrix[offset_y + m][offset_x + l] = 1 / 4 * \
                        distr_along_y[area.shape[0] + m] * \
                        distr_along_x[area.shape[1] + l]

            stochastic_matrix = np.append(
                stochastic_matrix,
                source_prob_matrix.flatten(),
                axis=0
            )

    stochastic_matrix = stochastic_matrix.reshape((sources, area.shape[0] * area.shape[1]))
    rows, cols = np.nonzero(stochastic_matrix)
    data = stochastic_matrix[np.nonzero(stochastic_matrix)]
    return coo_matrix((data, (rows, cols)))


def create_detector_image(img: np.ndarray, a: csr):
    """
    :param img: np.array representation of image
    :param a: sparse detector likelihood matrix
    :return: np.array image, which will be obtained by detector with the given
    likelihood matrix, if the sources radiate as img
    """

    print(a.toarray().shape)
    detector_img = a @ img.flatten()
    return detector_img


def create_noisy_image(filename, a: csr):
    """
    :param filename: path to image
    :param a: scipy.csr stochastic matrix
    :return: (image shape, original image, noisy image) tuple
    Returns a vector-like representation of image distorted with Poisson noise, based on file
    specified in 'filename', and  stochastic matrix, specified in 'a'
    """

    filename = os.path.join(os.getcwd(), filename)
    try:
        img = rgb2gray(img_as_ubyte(io.imread(filename))) * 255
    except FileNotFoundError as err:
        print(err.args)
        return None

    # It is unknown how to convert random samples into scale of white color intensity
    # optional, may be changed
    _lambda = 100.0

    detector_img = create_detector_image(img, a)
    noise = a @ np.random.poisson(_lambda, img.shape[0] * img.shape[1])

    return img.shape, img.flatten(), detector_img + noise

