import numpy as np

file_matrix = "matrix"
A = np.loadtxt(f"{file_matrix}.txt").astype(np.float64)
np.savez(f"{file_matrix}.npz", matrix=A)
