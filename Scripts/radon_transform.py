import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
import scipy.sparse as sp

def main():
    l = 2  # meters
    n = 200

    x = np.arange(n) * l / (n - 1) - l / 2
    dx = x[1] - x[0]
    ds = dx*1.42
    dt = dx*1.42

    radius = l / np.sqrt(2)
    s, t = np.arange(-radius, radius, ds), np.arange(-radius, radius, dt)
    theta = np.arange(0, 2*np.pi, 0.02)

    X, Y = np.meshgrid(x, x, indexing='ij')

    mua = np.zeros((n, n))
    mua[(X - 0.5) ** 2 + (Y+0.5) ** 2 < 0.5**2] = 1
    I = (X + 0.5) ** 2 + (Y - 0.5) ** 2 < 0.3 ** 2
    mua += np.exp(-(X + 0.5)**2/0.1 + -(Y - 0.5) ** 2/0.1)
    #mua[(X + 0.5) ** 2 + (Y - 0.5) ** 2 < 0.2 ** 2] = 1

    sinogram = make_sinogram(x, mua, s, t, theta)
    sinogram +=  np.random.uniform(-1, 1.0, size=sinogram.shape)*0.1

    mat = linear_system(sinogram, x, s, t , theta)
    est = art(mat, sinogram.ravel(), 0.1)
    est2 = back_projection(X, Y, sinogram, theta, s)/2

    fig, ax = plt.subplots(2, 2)
    ax = ax.ravel()
    ax[0].imshow(est, cmap="gray", vmin=0, vmax=1.5)
    ax[2].imshow(est2, cmap="gray", vmin=0, vmax=1.5)
    ax[1].imshow(mua, cmap="gray", vmin=0, vmax=1.5)
    ax[3].imshow(sinogram, cmap="gray")
    plt.show()

def art(mat, y, alpha, rep=100000):
    #alpha = 0.99
    x = np.zeros(mat.shape[1])
    for k in range(rep):
        #i = np.random.randint(mat.shape[0])
        i = k % mat.shape[0]
        mag = (mat[i, :]**2).sum()
        if mag <= 0: continue
        v = y[i] - mat[i, :] @ x
        x = x + alpha * v / mag * mat[i, :]
        x[x<0] = 0
    return x.reshape([int(np.sqrt(x.size)), -1])


def linear_system(sinogram, x, s, t, theta):
    S, Theta, T = np.meshgrid(s, theta, t, sparse=True, indexing='ij')
    dx = x[1] - x[0]
    n = x.size

    X = np.round((np.cos(Theta) * S + np.sin(Theta) * T -x[0])/dx).astype(int)
    Y = np.round((np.sin(Theta) * S - np.cos(Theta) * T - x[0]) /dx).astype(int)

    valid = (X >= 0) & (X<n) & (Y >= 0) & (Y<n) & (sinogram[:,:, None] > 0)
    indices = np.nonzero(valid)
    valid = valid.sum()

    coords = np.empty((2, valid), dtype=int)
    coords[0, :] = np.ravel_multi_index(indices[:2], sinogram.shape)
    coords[1, :] = np.ravel_multi_index((X[indices], Y[indices]), (n,n))
    values = np.full(valid, t[1]-t[0])
    return sp.coo_array((values, coords), (sinogram.size, n**2)).tocsr()

def line_integral(theta, s, t, prop_interp):
    S, T = np.meshgrid(s, t, sparse=True, indexing='ij')
    cost, sint = np.cos(theta), np.sin(theta)
    line_x = cost*S + sint*T
    line_y = sint*S - cost*T
    integral = prop_interp(np.stack([line_x, line_y], axis=2)).sum(axis=1) *(t[1] - t[0])
    return integral

def make_sinogram(x, mua, s, t, theta):
    interp = RegularGridInterpolator((x, x), mua, bounds_error=False, fill_value=0)
    measures = np.zeros((s.size, theta.size))
    for i, th in enumerate(theta):
        measures[:, i] = line_integral(th, s, t, interp)
    return measures

def back_projection(X,Y, sinogram, theta, s):

    def project_slice(sino_slice, X, Y, s, theta):
        S = X*np.cos(theta) + Y*np.sin(theta)
        return np.interp(S, s, sino_slice)

    est = np.zeros_like(X, dtype=complex)
    w = np.linspace(-np.pi, np.pi, sinogram.shape[0])
    w = np.fft.fftshift(w)
    w = np.abs(w)
    for i, th in enumerate(theta):
        measure = sinogram[:, i]
        signal = np.fft.fft(measure) * w
        signal = np.fft.ifft(signal)
        est += project_slice(signal, X, Y, s, th) * (theta[1] - theta[0])
    return np.abs(est) * 2 * np.pi



if __name__ == '__main__':
    main()


