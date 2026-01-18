import numpy as np
import matplotlib.pyplot as plt


x, y = np.meshgrid(*[np.linspace(0, 2, 200)]*2)

z = np.zeros_like(x)

def piece_wise(z, x, y, xlims, ylims, fx, fy):
    i = (xlims[0] <= x) & (x <= xlims[1]) & (ylims[0] <= y) &  (y <= ylims[1])
    z[i] = fx(x[i]) * fy(y[i])
    return z

piece_wise(z, x, y, [0, 1], [0, 1], lambda x: x, lambda y: y)
piece_wise(z, x, y, [1, 2], [0, 1], lambda x: 2-x, lambda y: y)
piece_wise(z, x, y, [0, 1], [1, 2], lambda x: x, lambda y: 2-y)
piece_wise(z, x, y, [1, 2], [1, 2], lambda x: 2-x, lambda y: 2-y)

A = np.random.rand(2,2)
xp = A[0,0] * x + A[1,0] * y
yp = A[0,1] * x + A[1,1] * y

ax = plt.axes(projection='3d')
ax.plot_surface(xp, yp, z, cmap="jet")
#plt.contour()

plt.show()