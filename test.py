import cvxopt
import torch
import time

def quadratic_optimization(Q, c):
    # Convert PyTorch tensors to NumPy arrays
    Q = Q.numpy()
    c = c.numpy()

    # Create CVXOPT matrices
    Q = cvxopt.matrix(Q)
    c = cvxopt.matrix(c)
    
    # Solve the quadratic problem
    sol = cvxopt.solvers.qp(Q, c)
    
    # Convert the result back to a PyTorch tensor
    result = torch.tensor(sol['x'])
    return result.squeeze()  # Ensure the result is a 1D tensor

# Define the problem size
size = 23508032

# Create a diagonal matrix Q and vector c
Q = torch.diag(torch.ones(size) * 0.5)
c = torch.ones(size)

# Measure the time taken to solve the problem
start_time = time.time()
result = quadratic_optimization(Q, c)
elapsed_time = time.time() - start_time

# Return the shape of the result and the time taken
print(result.shape, elapsed_time)
