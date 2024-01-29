import numpy as np
import torch
import time

# Matrix dimensions
size = 1000
np_matrix = np.random.rand(size, size)
torch_matrix = torch.rand(size, size)

# NumPy matrix multiplication
start_time = time.time()
np_result = np.dot(np_matrix, np_matrix)
np_time = time.time() - start_time

# PyTorch matrix multiplication
start_time = time.time()
torch_result = torch.mm(torch_matrix, torch_matrix)
torch_time = time.time() - start_time

# Display timings and check if the results are close
print(f"NumPy Matrix Multiplication Time: {np_time:.6f} seconds")
print(f"PyTorch Matrix Multiplication Time: {torch_time:.6f} seconds")

# Check if the results are close
if np.allclose(np_result, torch_result.numpy()):
    print("Results are close.")
else:
    print("Results differ.")

