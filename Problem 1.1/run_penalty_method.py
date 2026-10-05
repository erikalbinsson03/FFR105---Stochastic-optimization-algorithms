import math
import numpy as np

# ==============================
# run_gradient_descent function:
# ==============================

def run_gradient_descent(x_start, mu, eta, gradient_tolerance):
  gradient = compute_gradient(x_start, mu)
  x = x_start
  while np.linalg.norm(gradient) > gradient_tolerance:
    x_new = np.subtract(x, eta * np.array(gradient))
    gradient = compute_gradient(x_new, mu)
    x = x_new
  return x

# ==============================
# compute_gradient function:
# ==============================

def compute_gradient(x, mu):
  x1 = x[0]
  x2 = x[1]
  if x1**2 + x2**2 >= 1:
    gradient = [2 * (x1-1) + 4 * mu * x1 * (x1**2+x2**2-1), 4 * (x2-2) + 4 * mu * x2 * (x1**2+x2**2-1)]
  else:
    gradient = [2 * (x1 - 1), 4 * (x2 - 2)]
  return gradient

# ==============================
# Main program:
# ==============================

mu_values = [1, 10, 100, 1000]
eta = 1e-4
x_start = [1, 2]
gradient_tolerance = 1e-6

for mu in mu_values:
  x = run_gradient_descent(x_start, mu, eta, gradient_tolerance)
  output = f"x = ({x[0]:.4f}, {x[1]:.4f}), mu = {mu:.1f}"
  print(output)
