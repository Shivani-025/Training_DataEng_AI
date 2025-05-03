import numpy as np
from findiff import FinDiff
# Define the function
def f(x):
    return x**2 + 3*x + 2
# Compute the derivative at a point, say x=1
x_val = 1.0
dx = 1e-6
d_dx = FinDiff(0, dx, 1)
f_prime = d_dx(f)(x_val)
print("Function: f(x) = x**2 + 3*x + 2")
print(f"Derivative at x = {x_val}: f'(x) =", f_prime)


# Define the functions
def g(x):
    return x**2
def h(x):
    return 3*x + 1
# Sum function
def sum_func(x):
    return g(x) + h(x)
# Compute the derivative of the sum at a point, say x=1
x_val = 1.0
dx = 1e-6
d_dx = FinDiff(0, dx, 1)
sum_derivative = d_dx(sum_func)(x_val)
print("Sum Function: g(x) + h(x) = x**2 + 3*x + 1")
print(f"Sum Derivative at x = {x_val}: (g + h)'(x) =", sum_derivative)

# Define the functions
def g(x):
    return x**2
def h(x):
    return 3*x + 1
# Product function
def product_func(x):
    return g(x) * h(x)
# Compute the derivative of the product at a point, say x=1
x_val = 1.0
dx = 1e-6
d_dx = FinDiff(0, dx, 1)
product_derivative = d_dx(product_func)(x_val)
print("Product Function: g(x) * h(x) = (x**2) * (3*x + 1)")
print(f"Product Derivative at x = {x_val}: (g * h)'(x) =", product_derivative)

# Define the functions
def g(x):
    return x**2
def h(x):
    return 3*x + 1
# Quotient function
def quotient_func(x):
    return g(x) / h(x)
# Compute the derivative of the quotient at a point, say x=1
x_val = 1.0
dx = 1e-6
d_dx = FinDiff(0, dx, 1)
quotient_derivative = d_dx(quotient_func)(x_val)
print("Quotient Function: g(x) / h(x) = (x**2) / (3*x + 1)")
print(f"Quotient Derivative at x = {x_val}: (g / h)'(x) =", quotient_derivative)

# Define the inner and outer functions
def g(x):
    return np.sin(x)
def f(u):
    return np.exp(u)
# Composite function
def composite_func(x):
    return f(g(x))
# Compute the derivative using the chain rule at a point, say x=1
x_val = 1.0
dx = 1e-6
d_dx = FinDiff(0, dx, 1)
# Compute derivatives of inner and outer functions
inner_derivative = d_dx(g)(x_val)
outer_derivative = d_dx(f)(g(x_val))
chain_derivative = outer_derivative * inner_derivative
print("Composite Function: f(g(x)) = exp(sin(x))")
print(f"Chain Rule Derivative at x = {x_val}: dy/dx =", chain_derivative)

# Define the function
def f(x):
    return np.sin(x)
# Compute the first, second, and third derivatives at a point, say x=1
x_val = 1.0
dx = 1e-6
d_dx = FinDiff(0, dx, 1)
d2_dx2 = FinDiff(0, dx, 2)
d3_dx3 = FinDiff(0, dx, 3)
f_prime = d_dx(f)(x_val)
f_double_prime = d2_dx2(f)(x_val)
f_triple_prime = d3_dx3(f)(x_val)
print("Function: f(x) = sin(x)")
print(f"First Derivative at x = {x_val}: f'(x) =", f_prime)
print(f"Second Derivative at x = {x_val}: f''(x) =", f_double_prime)
print(f"Third Derivative at x = {x_val}: f(triple dash)(x) = ", f_triple_prime)


