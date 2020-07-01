'''Non-Linear Equation Solvers'''
import numpy as np
import math

def Bisection(a,b,f,tol):
    '''Solve f(x) = 0 by bisection method, inputs: interval endpoints a, b, funciton f, 
    tol: tolerance/error, set by user so that approximation within certain tolerance of exact solution
    Note there must be a solution to f(x) = 0 in [a,b],
    f continuous'''
    n = math.ceil(np.log((b-a)/tol)/np.log(2)-1)
    i = 0
    
    while i<= n:
        x = (a+b)/2
        if f(x)*f(b)<0:
            a = x
        else:
            b = x
        
        i += 1
    return x


def FixedPoint(f,n,x0):
    '''Condition for method to work is that the function F(x)=x-f(x) 
    MUST be a contraction so that it has a fixed point, then method will 
    converge to solution for any initial guess x0'''
    i = 0
    while i < n:
        x0 = x0-f(x0)
    return x0


def Newtons(f,f_prime,x,n ):
    '''Newtons method, requires funciton f to be continuously diffferentiable, 
    and have non-zero derivative at root, 
    f_prime derivative of f
    Initial guess x, n number of iterations'''
    i = 0
    print('x,    f(x):')
    while i <n:
        x = x - f(x)/(f_prime(x))
        print(x,  f(x))
        i += 1
