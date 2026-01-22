from math import prod

import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from scipy.interpolate import interpn


def da2d(k2, delta, source_pos):
    """
    Returns the Green's function to the modified Helmholtz equation: -nabla^2 phi(r) + k2(r) phi(r) = delta(r-r').
    Dirichlet boundary conditions. Uses FEM.
    :param k2: k-squared parameter of the Helmholtz equation. Given as a 2D matrix. It is assumed that each point
        is the center of the square-pixel and that k2 is uniform inside each square.
    :param delta: Discretization mesh size. It is assumed equal in each dimension.
    :param source_pos: Position of the dirac delta, r' = (sx,sy).
    :return: Solution phi(r) given as a 2D array of same shape as k2. Interpolated from vertex values to pixel centers.
    """

    element_grid = k2.shape
    vertex_grid = [k2.shape[0] + 1, k2.shape[1] + 1]  # Num vertices per dimension

    num_elements = k2.size
    num_vertices = prod(vertex_grid)

    corners = 4  # 4 neighbors
    sides = (vertex_grid[0] - 2) * 2 + (vertex_grid[1] - 2) * 2  # 6 neighbors
    inside = (vertex_grid[0] - 2) * (vertex_grid[1] - 2)  # 9 neighbors
    nnz = inside * 9 + corners + sides

    # Build the matrix
    inside_grid = [g - 2 for g in vertex_grid]
    row_offset = np.arange(num_vertices + 1) * 9
    row_offset[inside:] = inside * 9 + np.arange(corners + sides + 1)

    neighbors = np.arange(3) - 1
    neighbors = np.reshape(vertex_grid[1] * neighbors.reshape(-1, 1) + neighbors.reshape(1, -1), [1, -1])

    vals = np.zeros(nnz)
    cols_idx = np.zeros(nnz, dtype=int)

    vals[9 * inside:] = 1.0

    inside_index = np.meshgrid(*[np.arange(1, s - 1) for s in vertex_grid], indexing='ij')
    inside_index = np.ravel_multi_index(inside_index, vertex_grid).ravel()
    cols_idx[:inside * 9] = np.ravel(inside_index[:, None] + neighbors)

    corners_x = np.array([0, 0, 1, 1], dtype=int) * (vertex_grid[0] - 1)
    corners_y = np.array([0, 1, 0, 1], dtype=int) * (vertex_grid[1] - 1)
    cols_idx[inside * 9:inside * 9 + corners] = np.ravel_multi_index((corners_x, corners_y), vertex_grid)

    j = inside * 9 + corners

    def apply_sides(j, along):
        vary = (along + 1) % 2

        x = [0, vertex_grid[along] - 1]
        y = np.arange(1, vertex_grid[vary] - 1)
        dims = (x, y) if along == 0 else (y, x)

        idx = np.meshgrid(*dims, indexing='ij')
        cols_idx[j: j + (vertex_grid[vary] - 2) * 2] = np.ravel_multi_index(idx, vertex_grid).ravel()
        return j + (vertex_grid[vary] - 2) * 2

    j = apply_sides(j, 0)
    j = apply_sides(j, 1)

    def element_indices(start_offset, end_offset):
        col_offset = np.array(end_offset) - start_offset
        col_offset = np.ravel_multi_index(col_offset + 1, [3, 3])

        ex = np.arange(1 - start_offset[0], element_grid[0] - start_offset[0])
        ey = np.arange(1 - start_offset[1], element_grid[1] - start_offset[1])
        e_idx = np.ravel_multi_index(np.meshgrid(ex, ey, indexing='ij'), element_grid).ravel()

        return e_idx, col_offset

    for edge in range(16):
        points = np.unravel_index(edge, [2, 2, 2, 2])
        ele_idx, col = element_indices(start_offset=points[0:2], end_offset=points[2:4])

        if (col % 2) == 1:
            vals[row_offset[:inside] + col] += delta ** 2 * k2.flat[ele_idx] / 18 - 1 / 6
            continue

        if col == 4:
            vals[row_offset[:inside] + col] += delta ** 2 * k2.flat[ele_idx] / 9 + 2 / 3
            continue

        vals[row_offset[:inside] + col] += delta ** 2 * k2.flat[ele_idx] / 36 - 1 / 3

    matrix = sp.csr_matrix((vals, cols_idx, row_offset))

    sources = np.zeros(num_vertices)

    sx =  round(source_pos[0]/delta)
    sy = round(source_pos[1]/delta)

    s_idx = (vertex_grid[1] - 2) * sx + sy

    sources[s_idx] = 1

    phi_vertex = sp.linalg.spsolve(matrix, sources)
    phi_vertex = phi_vertex.reshape(vertex_grid)

    coords = [delta*np.arange(vert) for vert in vertex_grid]
    xi = np.stack(np.meshgrid(*[(np.arange(s) + 0.5)*delta for s in k2.shape], indexing='ij'), axis=2)
    return interpn(coords, phi_vertex, xi)

def da1d(k2, s, dx):
    n = k2.size
    mat = np.zeros((n,n))

    idx_in = np.arange(1, n-1)
    mat[idx_in, idx_in] = 2.0/dx**2 + k2[idx_in]
    mat[idx_in, idx_in-1] = -1.0 / dx ** 2
    mat[idx_in, idx_in + 1] = -1.0 / dx ** 2
    mat[0, 0] = 1
    mat[-1, -1] = 1

    return np.linalg.solve(mat, s)

if __name__ == '__main__':
    # Domain properties
    delta = 0.1
    k2 = np.full((150, 350), 0.03)

    x, y = np.meshgrid(*[(np.arange(s) - 0.5) for s in k2.shape], sparse=True, indexing='ij')
    k2[((x-50)**2 + (y-200)**2) < 50**2] = 0.5

    phi = da2d(k2, delta, [0.5, 0.5])

    func = lambda x : np.log10(x)
    plt.imshow(func(phi), cmap='jet')
    plt.show()

    print("Done")