'''Library of funcitons for Numerical Differentiation'''
import numpy as np
import matplotlib.pyplot as plt
import sympy as sym

def phi(x,h,f):
    return (f(x+h)-f(x-h))/(2*h)

def Richardsons(n,x,f,h):
    '''Richardson Extrapolation for finite difference approximaiton for f'(x)
    eliminates errors by linear combinations
    Most accurate answer will be the entry in top right corner of output matrix'''
    R= np.zeros((n,n))
    for i in range(n):
        R[i,0] = phi(x,2**i*h, f)
    for i in range(1,n):
        for j in range(n-i):
            R[j,i] = (4**i*R[j,i-1]-R[j+1, i-1])/(4**i-1)
    return R
    
def central(x,h,f):
    '''central difference method, error is O(h^2)'''
    return (f(x+h)-f(x-h))/2*h

def foward(x,h,f):
    '''Classic foward difference, error O(h)'''
    return (f(x+h)-f(x))/h

def calculate_coefficients(n,h):
    '''Calculate coeeficients for undetermined coefficients method so that
    differentiation is exact for polynomials of up to degree n
    REquires solving a linear system.
    Coefficients are independent of grid, we calculate over coefficients over [0,n]
    and transform to relevannt interval'''
    x = sym.Symbol('x')
    c = [sym.Symbol('a'+str(i)) for i in range(n+1)]
    A = []
    b = []
    A.append(sum(c[i] for i in range(n+1)))
    b.append(0)
    for i in range(1,n+1):
        A.append(sum(c[k]*k**i for k in range(n+1)))
        b.append(i*x**(i-1))
    A = sym.Matrix(A)
    b = np.array(b) 
    b = sym.Matrix(b)
    coefficients = sym.solve(A-b, c)
    coeffs = [i/h for i in coefficients.values()]
    return coeffs
    
def undetermined_coefficients(n,h,f,y,e):
    '''numerical approxiamtion of f'(y) at y. n the number of grid points, h uniform distance 
    between each grid point, f function, e lies between 0 and n, determines what difference method used,
    eg if n=2, e=1 then we have central difference method.'''
    '''will be exact for polynomials of up to degree n'''
    coeffs = calculate_coefficients(n,h)
    Coeffs = []
    for i in coeffs:
        g = sym.lambdify(sym.Symbol('x'),i)
        Coeffs.append(g(e))
    f_prime = sum(f(y-e*h+i*h)*Coeffs[i] for i in range(n+1))
    return f_prime
    

def plot_derivative(x,f,n,h,e):
    '''Plot approximation of derivative over linspace x'''
    '''Uses method of undetermined coefficients'''
    y = undetermined_coefficients(n,h,f,x,e)
    plt.plot(x,y, label = 'Approximation of derivative')
    plt.show()
        