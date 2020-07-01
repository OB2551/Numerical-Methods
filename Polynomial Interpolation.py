'''Library of functions for polynomial interpolation'''
import numpy as np
import sympy as sym
import matplotlib.pyplot as plt

def p(x, a):
    '''Evaluate Polynomial p at x.
       Use distributive law to reduce number of FLOPs
       a = (a0,a1,a2,....,an)
       p(x) = a0+a1x+...+anx^n = a0+x(a1+x(a2+..(...(an))))'''
    poly = 0
    n = len(a)
    for i in range(n):
        poly = poly*x + a[n-1-i]
    return poly


def p_Newt(x,c,X):
    '''Evaluate a Newtonian polynomial p(x) = c0 + c1(x-x0)+..+cn(x-x0)..(x-x_(n-1))
    Using the same method as function p'''   
    poly = 0
    n = len(c)
    for i in range(n):
        if i==0:
            poly += c[n-1]
        else:
            poly = poly*(x-X[n-1-i]) + c[n-1-i]
    return poly


def Interpolant_P(x,X,Y):
    '''Interpolation of collocation points (x0,y0),.....(xn,yn)
       X = [x0,x1,...,xn]
       Y = [y0,y1,....,yn]
       x = point(s) to evaluate interpolating polynomial at
       Using power basis method, hence the P. Solves a linear system to find coefficients
       Requires every collocation point to have a unique x value'''
    A = []
    for i in range(len(X)):
        A.append(X**i)
    A = np.array(A).T
    
    coeffs = np.linalg.solve(A,Y)
    #use p2 method in part a) to evaluate polynomial at x'''
    return p(x, coeffs)


def Symbolic(X,Y, method):
    '''return symbolic expression for interpolating polynomial for given method'''
    x = sym.Symbol('x')
    if method == 'P':
       return str(sym.expand(Interpolant_P(x, X,Y)))
    if method == 'L':
       return str(sym.expand(Interpolant_L(x, X,Y)))
    if method == 'N':
       return str(sym.expand(Interpolant_N(x,X,Y)))
    else:
        print('Not a valid input, expected a P,L or N')
        

def plot_interpolant(x,y, X,Y, p, method):
    '''Plot y=p(x) interpolation polynomial and data points (X,Y), choose method to be
    either Power basis, Lagrangian, Newton,   p the symbolic expression for the interpolating polynomial'''
    plt.plot(x,y, label = 'Interpolant')
    plt.plot(X,Y, 'o', label = 'Data')
    plt.legend()
    plt.title('Interpolation by ' + method +' \n \np(x)='+p + '\n')
    plt.show()
    
    
def Basis_j(x,j, X):
    '''x: point(s) to evaluate jth cardinal basis at, 
    X = [x0,x1,...xn] collocation x values'''
    
    prod = 1
    x_points = list(X.copy())
    x_points.remove(X[j])
    for i in x_points:
        prod *= (x-i)/(X[j]-i)
    return prod


def Interpolant_L(x,X,Y):
    '''Compute Lagrangian Interpolant, hence L in funciton name
    x: point(s) to evaluate interpolating polynomial at
    X = [x0,...xn], Y = [y0, .. yn] collocation data '''
    
    p = 0
    for j in range(len(X)):
        p += Y[j]*Basis_j(x,j,X)
    return p


def Newton_Basis(N,x,X):
    '''Returns the Nth newton basis function'''
    prod = 1
    for j in range(N):
        prod*= (x-X[j])
    return prod


def find_coeffs(x,k, coeffs, X,Y):
    '''Find the Newton Coefficients'''
    #case to catch when k=0, we have p0 = Y[0]
    if k==0:
        coeffs.append(Y[0])
        return Y[0]
    else:
        #find previous polynomial p_k, dependent on the first k coefficients'''
        p_k = find_coeffs(x, k-1,coeffs,X,Y)
        #'''c_k = (y_k-p_{k-1}(x_k))/(n_k(x_k)) '''
        c_k = (Y[k]-find_coeffs(X[k], k-1,coeffs,X,Y))/Newton_Basis(k, X[k],X)
        #'''add coefficien to list'''
        coeffs.append(c_k)  
        
        return p_k+c_k*Newton_Basis(k,x,X)

def Interpolant_N(x,X,Y):
    '''Compute Interpolant with Newton method, x input value(s), X,Y collocation data'''
    Newt_Coeffs = []
    N = len(X)
    find_coeffs(1,N-1,Newt_Coeffs,X,Y)
    Newt_Coeffs = Newt_Coeffs[(-N):]
    return p_Newt(x,Newt_Coeffs, X)


def ChebyshevNodes(a,b,n,f):
    '''computes chebychev collocation nodes to be used for polynomial interpolation 
     for funciton f on interval [a,b]
     n number of nodes to use.
     Will generally interpolate more accurately than uniformly spaced nodes '''
    #nodes
    X = []
    Y = []
    for i in range(n+1):
        theta = ((2*i+1)*np.pi)/(2*(n+1))
        z = np.cos(theta)
        #transform nodes from [-1,1] to [a,b], z -> (a+b)/2 - (a-b)/2 *z
        x = (a+b)/2-(a-b)/2*z
        X.append(x)
        Y.append(f(x))
    return X,Y


    

