import os
import numpy as np
from scipy import special
from skimage import io, img_as_ubyte
from skimage.color import rgb2gray


def generate_stochastic_matrix(
        area: np.ndarray,
        detectors_x: int,
        detectors_y: int
    ):
    """
    :param area: basis for stoch.matrix
    :param detectors_y: the amount of detectors along fixed x axis
    :param detectors_x: the amount of detectors along fixed y axis
    :return: np.ndarray of size (area.shape[0] * area.shape[1], detectors_x * detectors_y)
    Generates a row-stochastic matrix (see https://en.wikipedia.org/wiki/Stochastic_matrix)
    based on 2d numpy array specified in area.
    It is considered that area is covered with detectors_x * detectors_y detectors
    placed equidistantly over the array, having a chance to detect a particle within area
    corresponding to 2d normal
    distribution with dispersion comparable with distance between them

    A element of matrix indexed (i, j) stands for the probability for detector i to catch
    a particle received by the j-th cell of the area
    """

    sources = detectors_x * detectors_y
    h_x = 1.0 / (detectors_y - 1)
    h_y = 1.0 / (detectors_x - 1)
    # optional, may be changed
    dispersion_multiplier = 5.0  # square root of dispersion of sources / distance between sources
    sigma_x = dispersion_multiplier * h_x
    sigma_y = dispersion_multiplier * h_y

    x = np.linspace(-1.0, 1.0, 2 * area.shape[1] + 1)
    y = np.linspace(-1.0, 1.0, 2 * area.shape[0] + 1)

    distr_along_x = np.array(
        [special.erf(x[i + 1] / sigma_x / 2 ** 0.5) -
         special.erf(x[i] / sigma_x / 2 ** 0.5)
         for i in range(2 * area.shape[1])])

    distr_along_y = np.array(
        [special.erf(y[i + 1] / sigma_y / 2 ** 0.5) -
         special.erf(y[i] / sigma_y / 2 ** 0.5)
         for i in range(2 * area.shape[0])])

    stochastic_matrix = np.array([])

    for i in range(detectors_y):
        for j in range(detectors_x):

            source_prob_matrix = np.zeros(area.shape)
            for l in range(area.shape[1]):
                for m in range(area.shape[0]):
                    offset_x = int(round(area.shape[1] * (1 - h_x * i)))
                    offset_y = int(round(area.shape[0] * (1 - h_y * j)))
                    source_prob_matrix[m][l] = 1 / 4 *\
                        distr_along_x[offset_x + l] *\
                        distr_along_y[offset_y + m]

            stochastic_matrix = np.concatenate(
                (stochastic_matrix, source_prob_matrix.flatten()),
                axis=None
            )

    return stochastic_matrix.reshape(sources, area.shape[0] * area.shape[1])


def create_noisy_image(filename='tiger.jpg'):
    """
    :param filename: string
    :return: boolean
    Generates a image distorted with Poisson noise, based on image
    specified in filename, and saves it into current working directory with the same
    extension. Returns True if the image creation was successful, False otherwise
    """

    img = None
    filename = os.path.join(os.getcwd(), filename)
    try:
        img = rgb2gray(img_as_ubyte(io.imread(filename))) * 255
    except FileNotFoundError as err:
        print(err.args)
        return False

    # It is unknown how to convert random samples into scale of white color intensity
    # optional, may be changed
    _lambda = 100.0

    noise = np.random.poisson(_lambda, img.shape)

    img = np.array([i if i <= 255 else 255 for i in np.nditer(img + noise)]).reshape(
        img.shape[0],
        img.shape[1],
    )

    result_file = filename.split(".")[0] + "_noisy." + filename.split(".")[1]
    io.imsave(result_file, img.astype(np.uint8))

    return True
