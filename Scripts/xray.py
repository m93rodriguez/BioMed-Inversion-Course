import numpy as np
from scipy.interpolate import RegularGridInterpolator
import scipy.sparse as sp
from PIL import Image
import matplotlib.pyplot as plt

def make_sinogram(mua, dx,  s, theta):
    """
    Return the sinogram of the given absorption coefficient distribution. Uses interpolation to approximate integrals.
    :param mua: 2D square matrix of absorption coefficients
    :param dx: Discretization step. Uniform in both dimensions
    :param s: Vector representing the detector plane. Should contain matrix mua assuming the center of mua is at (0,0).
    :param theta: Vector of angles
    :return:
    """

    def line_integral(theta, s, t, prop_interp):
        S, T = np.meshgrid(s, t, sparse=True, indexing='ij')
        cost, sint = np.cos(theta), np.sin(theta)
        line_x = cost * S + sint * T
        line_y = sint * S - cost * T
        integral = prop_interp(np.stack([line_x, line_y], axis=2)).sum(axis=1) * (t[1] - t[0])
        return integral

    n = np.max(mua.shape)
    l = dx * (n - 1)
    radius = l / np.sqrt(2)
    x =  np.arange(n) * l / (n - 1) - l / 2
    t = np.arange(-radius, radius, dx * np.sqrt(2))

    interp = RegularGridInterpolator((x, x), mua, bounds_error=False, fill_value=0)
    measures = np.zeros((s.size, theta.size))
    for i, th in enumerate(theta):
        measures[:, i] = line_integral(th, s, t, interp)
    return measures

def radon_matrix(x, s, theta, sinogram=None):
    """
    Returns a matrix encoding the Radon transform linear operator.
    :param x: Discretization vector the object of interest positions. It is assumed that the domain is square.
        Should be approximately symmetric around 0.
    :param s: Discretization vector of the detector position. Should be approximately symmetric around 0.
    :param theta: Discretization vector of detector rotations
    :param sinogram: Optional. If given, only gives the rows of the matrix with at least one non-zero measurement.
    :return: Matrix of the Radon transform operation. Given as a sparse Compressed-Sparse-Row (CSR) matrix.
        Recall that after using the resulting vector should be reshaped.
    """

    if sinogram is not None and sinogram.shape != (s.size, theta.size):
        raise ValueError("Dimensions of sinogram do not match with s, theta")

    dx = x[1] - x[0]
    n = x.size
    t = np.arange(1.5*x[0], 1.5*x[-1], dx*1.5)

    S, Theta, T = np.meshgrid(s, theta, t, sparse=True, indexing='ij')
    X = np.round((np.cos(Theta) * S + np.sin(Theta) * T -x[0])/dx).astype(int)
    Y = np.round((np.sin(Theta) * S - np.cos(Theta) * T - x[0]) /dx).astype(int)

    valid = (X >= 0) & (X<n) & (Y >= 0) & (Y<n)

    if sinogram is not None:
        valid = valid & (sinogram[:,:, None]>0)

    indices = np.nonzero(valid)
    valid = valid.sum()

    coords = np.empty((2, valid), dtype=int)
    coords[0, :] = np.ravel_multi_index(indices[:2], [s.size, theta.size])
    coords[1, :] = np.ravel_multi_index((X[indices], Y[indices]), (n,n))
    values = np.full(valid, t[1]-t[0])
    return sp.coo_array((values, coords), (s.size*theta.size, n**2)).tocsr()


if __name__ == '__main__':

    l = 2
    theta = np.linspace(0, 2 * np.pi, 300)
    s = np.linspace(-l / np.sqrt(2), l / np.sqrt(2), 250)



