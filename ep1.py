import numpy as np

A = np.array([[2,-1,1,3],[-1,1,4,2],[1,4,2,-1],[3,2,-1,1]]).astype(np.float32)
n = np.size(A,0)

def get_vector(A, index, column=False):
    if column:
        return A[:,index:index+1]
    else:
        return A[index:index+1,:]

def scalar_product(x,y):
    value = 0
    for i, xi in enumerate(x):
        value+=xi*y[i]
    return value

def norm(x):
    value = 0
    for i,xi in enumerate(x):
        value+=xi**2
    return value**(1/2)

def get_Hx(x, w):        
    xw = scalar_product(x,w)
    ww = scalar_product(w,w)

    def get_sub(wi):
        return 2*xw*wi/ww

    sub = np.array([ get_sub(wi) for wi in w])
    hx = np.subtract(x,sub)
    return hx

def get_w(k):
    ai = A[k+1:n+1, k:k+1].copy()
    sign = np.sign(ai[0])
    ai[0] = ai[0] + sign*norm(ai)
    w=[0, *ai]
    return w

def get_HAH():
    k=0
    HA = np.zeros([n,n])
    w= get_w(k)
    for i in range(n):
        x = A[:,i:i+1]
        HA[:,i:i+1]=get_Hx(x,w)
    print(HA)
    HAH = np.zeros([n,n])
    for i in range(n):
        x = HA[i:i+1,:]
        x = x.T
        HAH[:,i:i+1]=get_Hx(x,w)
    return HAH

T= np.zeros([n,n])