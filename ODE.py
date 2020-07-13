import numpy as np
import matplotlib.pyplot as plt

def phi(t,u,h,f, method='Euler'):
    if method=='Euler':
        return f(t,u)
    elif method=='Heun':
       # LB: to be implemented by student 
        v = f(t,u)
        return 1/2*(v+f(t+h, u+h*v))
    elif method=='RK4':
        # LB: to be implemented by student
        k1 = f(t,u)
        k2 = f(t+h/2, u+h*k1/2)
        k3 = f(t+h/2, u+h*k2/2)
        k4 = f(t+h, u+h*k3)
        return 1/6*(k1+2*k2+2*k3+k4)
    elif method == 'Midpoint':
        return f(t+h/2, u+h*f(t,u)/2)
    
    
def ODE_Solver(T,n,f,method,u0):
    h = T/(n-1)
    sys = False
    #if solving system of ODES
    if type(u0) == np.ndarray:
           u = np.zeros((n, len(list(u0))))
           sys = True
    else:
           u = np.zeros(n)
    u[0] = u0
    # numerical solution
    #u_(k+1) = u_k + h * phi(t_k, u_k)
    for k  in range(n-1): 
        tk = k*h
        u[k+1] = u[k] + h*phi(tk,u[k],h, f, method)
    if sys:
     plt.plot(np.linspace(0,T, n), u[:,0])
    else:
     plt.plot(np.linspace(0,T,n), u)
    plt.show()
    

        
