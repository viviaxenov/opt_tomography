import os
import numpy as np
from scipy import special
from skimage import io, img_as_ubyte
from skimage.color import rgb2gray


def generate_stochastic_matrix(
        area: np.ndarray,
        sources_x: int,
        sources_y: int
    ):
    """
    :param area: basis for stoch.matrix
    :param sources_y: the amount of sources along fixed x axis
    :param sources_x: the amount of sources along fixed y axis
    :return: np.ndarray of size (area.shape[0] * area.shape[1], sources_x * sources_y)
    Generates a column-stochastic matrix (see https://en.wikipedia.org/wiki/Stochastic_matrix)
    based on 2d numpy array specified in area.
    It is considered that area is lit with sources_x * sources_y emitters
    placed equidistantly over the area, radiating with 2d normal
    distribution with dispersion comparable with distance between them

    A element of matrix indexed (i, j) stands for the probability for emitter j to radiate
    into the ith cell of the area
    """

    sources = sources_x * sources_y
    h_x = 1.0 / (sources_y - 1)
    h_y = 1.0 / (sources_x - 1)
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

    for i in range(sources_y):
        for j in range(sources_x):

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

    return stochastic_matrix.reshape(
        sources,
        area.shape[0] * area.shape[1],
    ).transpose()


def create_noisy_image(filename='tiger.jpg'):
    """
    :param filename: string
    :return: boolean
    Generates a image distorted with Poisson noise, based on image
    specified in filename, and saves it into current working directory with the same
    extension. Returns True if the image creation was successful, False otherwise
    """

    filename = os.path.join(os.getcwd(), filename)
    try:
        img = rgb2gray(img_as_ubyte(io.imread(filename))) * 255
    except FileNotFoundError as err:
        print(err.args)
        return False

    stochastic_matrix = generate_stochastic_matrix(img, 30, 30)

    # It is unknown how to convert random samples into scale of white color intensity
    # optional, may (or even must) be changed
    intensity_multiplier = 1.0

    noise = (stochastic_matrix @ np.random.poisson(10000, stochastic_matrix.shape[1])).reshape(
        img.shape[0],
        img.shape[1],
    ) * intensity_multiplier

    img = np.array([i if i <= 255 else 255 for i in np.nditer(img + noise)]).reshape(
        img.shape[0],
        img.shape[1],
    )

    result_file = filename.split(".")[0] + "_noisy." + filename.split(".")[1]
    io.imsave(result_file, img.astype(np.uint8))

    return True
