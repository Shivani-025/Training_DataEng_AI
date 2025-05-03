'''
Question 3: Calculus with SciPy

Define the function f(x) = x^3 + 2x^2 + x + 1.
Compute the first and second derivatives of f(x) at x = 1.
Compute the definite integral of f(x) from x = 0 to x = 2.
'''

import numpy as np
from scipy.linalg import solve
import sympy as sp


# Define the function f(x) = x^3 + 2x^2 + x + 1 symbolically
x = sp.symbols('x')
f_x = sp.sympify('x**3 + 2*x**2 + x + 1')

# Compute the first derivative of f(x)
first_derivative = sp.diff(f_x, x)

# Compute the second derivative of f(x)
second_derivative = sp.diff(first_derivative, x)

# Evaluate the first and second derivatives at x = 1
first_derivative_value = first_derivative.subs(x, 1)
second_derivative_value = second_derivative.subs(x, 1)

print("\nCalculus Computations:")
print("First derivative of f(x) at x=1:", first_derivative_value)
print("Second derivative of f(x) at x=1:", second_derivative_value)

# Compute the definite integral of f(x) from x = 0 to x = 2
integral_result = sp.integrate(f_x, (x, 0, 2))

print("\nDefinite Integral of f(x):")
print("Integral from x=0 to x=2:", integral_result.evalf())
