import numpy as np

A = np.array([[2,-1,1,3],[-1,1,4,2],[1,4,2,-1],[3,2,-1,1]]).astype(np.float32)
n = np.size(A,0)
HT = np.identity(n)

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

def get_wk(A):
    ak = A[1:, 0:1].copy()
    sign = np.sign(ak[0])
    ak[0] = ak[0] + sign*norm(ak)
    wk=[0, *ak]
    return wk

def get_Hx(x, w):
    xw = scalar_product(x,w)
    ww = scalar_product(w,w)

    def get_sub(wi):
        return 2*xw*wi/ww

    sub = np.array([ get_sub(wi) for wi in w])
    hx = np.subtract(x,sub)
    return hx


def get_transformation(M):

    def right_transformation(M,result,i):
        x = M[i:i+1,:]
        x = x.T
        result[:,i:i+1]=get_Hx(x,w)

    m = np.size(M, 0)
    w= get_wk(M)

    HM = np.zeros([m,m])

    for i in range(m):
        x = M[:,i:i+1]
        HM[:,i:i+1]=get_Hx(x,w)

    HMH = np.zeros([m,m])
    for i in range(m):
        right_transformation(HM,HMH,i)
        right_transformation(HT[n-m:n,n-m:n], HT[n-m:n,n-m:n], i)
        
        # x = HM[i:i+1,:]
        # x = x.T
        # HMH[:,i:i+1]=get_Hx(x,w)

    return HMH

def get_T():
    T = np.zeros([n,n])
    for i in range(n-2):
        M = A[i:n+1,i:n+1]
        M=get_transformation(M)
        T[i:n+1,i:n+1] = M
    return T

T = get_T()

print(T)
print(HT)