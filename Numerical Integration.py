'''Library of Functions for Numerical Integration'''
import numpy as np
import sympy as sym


def Rect(partition_size, a, b, f):
    xpts = np.linspace(a,b,partition_size+1)
    approx = 0.0
    for i in range(partition_size):
        approx = approx + (xpts[i+1]-xpts[i])*f(xpts[i])
    return approx


def trapezoidal(n,a,b,f):
    '''Evaluate trapezoidal rule with n nodes, n-1 subintervals
    for function f on interval [a,b]'''
    h = (b-a)/(n-1)
    grid = []
    for i in range(1,n-1):
        grid.append(a+i*h)
    s = sum(f(x) for x in grid)
    return h*((f(a)+f(b))/2 + s)


def Romberg(N,f,b,a):
    '''Romberg method, uses linear combinations of trapezoidal rule to decrease error,
    requires argument N, which determines degree of accuract, choose higher N for smaller error at cost of
    computation time. f function to itegrate over interval a,b'''
    R= np.zeros((N,N))
    points = [2**i+1 for i in range(N)]
    for i in range(N):
        R[i,0] = trapezoidal(points[i],a,b, f)
    for i in range(1,N):
        for j in range(i,N):
            R[j,i] = (4**i*R[j,i-1]-R[j-1, i-1])/(4**i-1)
    return R
  
    
#Newton Cotes Method

def int_poly(n,a,b):
    '''Compute itegral of x^n exactly on a,b'''
    return (b**(n+1)-a**(n+1))/(n+1)

def nodes(m,a,b):
    '''Quadrature nodes'''
    return np.linspace(a,b,m+1)

def poly(x,n):
    return x**n

def Newton_Cotes_Coeffs(a,b,m):
    '''Compute Newton Cote coefficients'''
    A = np.zeros((m+1,m+1))
    xpts = nodes(m,a,b)
    B = np.zeros(m+1)
    for n in range(0,m+1):
        A[n] = poly(xpts, n)
        B[n] = int_poly(n,a,b)
    C = np.linalg.solve(A,B)
    return C

def Newton_Cotes(a,b,m,f):
    '''Evaluate Newton Cotes quadrature on a single interval a,b'''
    Coeffs = Newton_Cotes_Coeffs(a,b,m)
    xpts = nodes(m,a,b)
    quad = 0
    for i in range(len(xpts)):
        quad += Coeffs[i]*f(xpts[i])
    return quad

def Composite_NC(a,b,N,m,f):
    '''Divide [a,b] into N subintervals and then divide each subinterval into m further 
    subinterval and compute quadrature of f on each'''
    comp_nodes = nodes(N, a,b)
    integral = 0
    for i in range(N):
        integral += Newton_Cotes(comp_nodes[i], comp_nodes[i+1], m, f)
    return integral
    

#Gaussian Quadrature

def Legendre(n):
    '''Compute the nth Legendre Polynomial'''
    x = sym.Symbol('x')
    if n == 0:
        return 1
    if n==1:
        return x
    else:
        c = sym.integrate(x*Legendre(n-1)*Legendre(n-2), (x,-1,1))/sym.integrate(Legendre(n-2)*Legendre(n-2), (x,-1,1))
        return sym.expand(x*Legendre(n-1)-c*Legendre(n-2))
  
    
def Gauss_Nodes(n):
    '''Computes nodes on [-1,1] which are roots of Legendre Polynomial of degree n, 
    then transform nodes to [a,b]'''
    p = Legendre(n)
    c = sym.Poly(p).all_coeffs() # Legendre coefficients
    nodes = np.roots(c)
    nodes.sort()
    return nodes

def Gauss_Quad_Coeffs(m):
    '''Choose coefficients so that quadrature exact for polynomials of degree m+1
       amounts to solving a linear system, similar to Newton_Cotes_Coeffs'''
    A = np.zeros((m+1,m+1))
    xpts = Gauss_Nodes(m+1)
    B = np.zeros(m+1)
    for n in range(m+1):
        A[n] = poly(xpts, n)
        B[n] = int_poly(n,-1,1)
    C = np.linalg.solve(A,B)
    return C

def Gauss_Quad(a,b,m,f):
    '''Computes gauss quad on [a,b], transforms quadrature nodes on [-1,1] to interval [a,b]'''
    '''function f, m+1 nodes, method exact for polynomials of degree 2m+1'''
    quad = 0
    coeffs = Gauss_Quad_Coeffs(m)
    xpts = Gauss_Nodes(m+1)
    for i in range(len(xpts)):
        quad += coeffs[i]*f((b-a)/2*xpts[i]+(a+b)/2)
    return (b-a)/2*quad 


def Comp_GC(a,b,N,m,f):
    '''Composite Gaussian Quadrature with Nm subintervals'''
    comp_nodes = nodes(N, a,b)
    integral = 0
    for i in range(N):
        integral += Gauss_Quad(comp_nodes[i], comp_nodes[i+1], m, f)
    return integral

    
    
    
    