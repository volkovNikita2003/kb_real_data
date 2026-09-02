import numpy as np

file_matrix = "matrix"
A = np.load(f"{file_matrix}.npz")["matrix"].astype(np.float64)
np.savetxt(f"{file_matrix}.txt", A)
