'''Library of Linear System solvers'''
import numpy as np

def GE(A, b):
    '''Gaussian elimination for solving A x = b with no pivoting.
    Can get a divison by 0 error if one of the diagonal entries is 0 
    Pivoting fixes this problem'''
    n =  len(A)
    M = np.column_stack((A,b))
    for k in range(n-1):
        for i in range(k+1, n):
            multiplier = M[i,k]/M[k,k]
            for j in range(k,n+1):
                M[i,j] = M[i,j] - multiplier*M[k,j]
    
    # Back substitution
    x = np.zeros((n,1))
    k = n-1
    x[k] = M[k,n]/M[k,k]
    for i in range(n-2,-1,-1):
        z = 0
        for j in range(i+1,n):
            z = z + M[i,j]*x[j]
        x[i] = (M[i,n]-z)/M[i,i]
        
    
    return x


def findPivot(col,k):
    '''Find pivot in column col, truncated above k,
    note the pivot is entry with largest absolute value'''
    column = col
    pivotrow = k
    row = k
    pivot = column[0]
    for entry in column:
            if abs(entry) > abs(pivot):
                pivot = entry
                pivotrow = row
                row += 1
            else:
                row += 1
    return pivotrow

def findPivot2(col,k):
    column = list(col)
    pivot = max(abs(i) for i in column )
    if pivot in column:
     pivotrow = column.index(pivot)+k
    else:
     pivotrow = column.index(-pivot)+k
    return pivotrow
    

def GEPIVOT(A, b):
    '''Gaussian elimination for solving A x = b with  pivoting'''
    n =  len(A)
    M = np.column_stack((A,b))
    M.astype(float)
    for k in range(n-1):
        p = findPivot(M[k:,k], k)
        #swap rows
        M[[p, k]] = M[[k, p]]
    
        for i in range(k+1, n):
            multiplier = M[i,k]/M[k,k]
            for j in range(k,n+1):
                M[i,j] = (M[i,j]) - (multiplier*M[k,j])
    # Back substitution
    x = np.zeros((n,1))
    k = n-1
    x[k] = M[k,n]/M[k,k]
    for i in range(n-2,-1,-1):
        z = 0
        for j in range(i+1,n):
            z = z + M[i,j]*x[j]
        x[i] = (M[i,n]-z)/M[i,i]
        
    
    return x      


def Jacobi(A,b,tol,x0):
    '''Solve Ax=b with Jacobi iterative method. User specify tolerance 
    and intial guess x0, must be a scalar'''
    n = len(b)
    xk = x0*np.ones((n,))
    rk = np.dot(A,xk) - b
    dinv = 1.0/np.diag(A)   
    #iteration counter i
    i = 0
    #initialise relative error
    rel = 1
    #continue method until tolerance met
    while rel>tol:
        xold = xk
        #update solution
        xk = xk - dinv*rk
        #calculate relative error approximation#
        rel = max(abs(i) for i in (xk-xold))/max(abs(i) for i in xold)
        #update residue
        rk = np.dot(A,xk) - b
        i +=1
    return xk 


def GaussSeidel_2(A,b,tol,x0):
    '''solving Ax=b by the Gauss-Seidel iterative method
    User specify scalar initial guess x0and tolerance'''
    n = len(b)
    xk = x0*np.ones((n,))
    dinv = 1.0/np.diag(A)  
    #residue
    rk = -1.0*b.copy()
    s = 0
    err = 1
    while err>tol:
        #update solution, note that in the Gauss Seidel method, x^(k+1)[i] depends on x^k[j] j>i and 
        #also on x^(k+1)[j] for j<i, hence we must update each x[i] iteratively, as below:'''
        xold = xk.copy()
        for i in range(n):
            rk[i] = np.dot(A[i,:],xk) - b[i]
            xk[i] -= dinv[i]*rk[i]
        err = max(abs(i) for i in (xk-xold))/max(abs(i) for i in xold)
        s+=1     
    return xk
               

def dotA(A,x,y):
    '''Calculate x^TAy for vectors x,y, matrix A '''
    return np.dot(x,np.dot(A,y)) 

def ConjGrad(A,b,x0, tol):
    '''Conjugate gradient method. Matrix must be SPD for convergence
    initial SCALAR guess x0, specify tolerance, will stop when arrive at
    exact solution or tolerance met'''
    n = len(b)
    xk =x0*np.ones(n)
    rk = np.dot(A,xk)-b
    dk = -rk
    k = 0
    err = 1
    while k <n  and err > tol:
        x_old = xk
        dkAdk = dotA(A,dk, dk)
        #step size
        alphak = -np.dot(rk,dk)/dkAdk
        #update solution
        xk = xk + alphak*dk
        err = max(abs(i) for i in (xk-x_old))/max(abs(i) for i in x_old)
        #residue
        rk = np.dot(A, xk) -b
        betak = dotA(A, rk, dk)/dkAdk
        #search direction
        dk = -rk + betak*dk
        k+=1
    return xk


def SteepestDescent(A,b,x0, n):
    '''method of steepest descent to solve Ax = b
    Scalar inital guess x0, n number of iterations'''
    m = len(b)
    xk = x0*np.ones(m)
    k = 0
    while k<n:
        dk = b-np.dot(A,xk)
        alphak = np.dot(dk,dk)/dotA(A,dk,dk)
        xk = xk + alphak*dk
        k+=1
    return xk
                
                
                
