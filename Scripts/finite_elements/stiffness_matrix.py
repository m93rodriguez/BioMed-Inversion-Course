from math import ceil, prod

import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

# Domain properties
delta = 0.1
k2 = np.full((150, 350), 0.03)

x, y = np.meshgrid(*[(np.arange(s) - 0.5) for s in k2.shape], sparse=True, indexing='ij')
i = ((x-50)**2 + (y-200)**2) < 50**2
k2[i] = 0.5


element_grid = k2.shape
vertex_grid = [k2.shape[0] + 1, k2.shape[1] + 1]  # Num vertices per dimension

num_elements = k2.size
num_vertices =  prod(vertex_grid)

corners = 4 # 4 neighbors
sides = (vertex_grid[0] - 2) *2 + (vertex_grid[1] - 2) *2 # 6 neighbors
inside = (vertex_grid[0]-2) * (vertex_grid[1]-2) # 9 neighbors
nnz = inside * 9 + corners + sides

# Build the matrix
inside_grid = [g-2 for g in vertex_grid]
row_offset = np.arange(num_vertices + 1) * 9
row_offset[inside:] = inside*9 + np.arange(corners+sides+1)

neighbors = np.arange(3)-1
neighbors = np.reshape(vertex_grid[1] * neighbors.reshape(-1, 1) + neighbors.reshape(1, -1), [1, -1])

vals = np.zeros(nnz)
cols_idx = np.zeros(nnz, dtype=int)

vals[9*inside:] = 1.0

inside_index = np.meshgrid(*[np.arange(1, s-1) for s in vertex_grid], indexing='ij')
inside_index = np.ravel_multi_index(inside_index, vertex_grid).ravel()
cols_idx[:inside*9] = np.ravel(inside_index[:, None] + neighbors)

corners_x = np.array([0, 0, 1, 1], dtype=int) * (vertex_grid[0]-1)
corners_y = np.array([0, 1, 0, 1], dtype=int) * (vertex_grid[1]-1)
cols_idx[inside*9:inside*9+corners] = np.ravel_multi_index((corners_x, corners_y), vertex_grid)

j = inside*9 + corners
def apply_sides(j, along):
    vary = (along + 1) % 2

    x = [0, vertex_grid[along]-1]
    y = np.arange(1, vertex_grid[vary]-1)
    dims = (x,y) if along == 0 else (y,x)

    idx = np.meshgrid(*dims , indexing='ij')
    cols_idx[j: j + (vertex_grid[vary] - 2)*2] = np.ravel_multi_index(idx, vertex_grid).ravel()
    return j + (vertex_grid[vary] - 2)*2

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

    if (col%2) == 1:
        vals[row_offset[:inside] + col] += delta**2 * k2.flat[ele_idx]/18 - 1/6
        continue

    if col == 4:
        vals[row_offset[:inside] + col] += delta**2 *k2.flat[ele_idx] / 9 + 2 / 3
        continue

    vals[row_offset[:inside] + col] += delta**2 *k2.flat[ele_idx] / 36 - 1 / 3

matrix = sp.csr_matrix((vals, cols_idx, row_offset))

sources = np.zeros(num_vertices)

s_idx = (vertex_grid[1] - 2)*int((vertex_grid[0] - 3)*0.15) + int((vertex_grid[1] - 3)*0.5)

sources[s_idx] = 1

x = sp.linalg.spsolve(matrix, sources)
x = x.reshape(vertex_grid)

func = lambda x : np.log10(x+1e-10)

plt.imshow(func(x), cmap='jet', vmin=-5)
plt.contour( func(x), cmap='jet', levels=np.arange(-5, 0, 0.5), linestyles="dotted")
plt.show()

print("Done")