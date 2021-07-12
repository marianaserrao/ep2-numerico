import numpy as np

A = np.array([[2,-1,1,3],[-1,1,4,2],[1,4,2,-1],[3,2,-1,1]]).astype(np.float32)
n = np.size(A,0)

#funcao para obter uma linha ou coluna de matriz
def get_vector(A, index, column=False):
    if column:
        return A[:,index:index+1]
    else:
        return A[index:index+1,:]

#funcao para obter o produto escalar de dois vetores
def scalar_product(x,y):
    value = 0
    for i, xi in enumerate(x):
        value+=xi*y[i]
    return value

#funcao para obter a norma de um vetor
def norm(x):
    value = 0
    for i,xi in enumerate(x):
        value+=xi**2
    return value**(1/2)

#funcao para obter o vetor w para transformacao Householder
def get_wk(M):
    ak = M[1:, 0:1].copy()
    sign = np.sign(ak[0])
    ak[0] = ak[0] + sign*norm(ak)
    wk=[0, *ak]
    return wk

#funcao para obter o produto de um vetor pela matriz Householder
def get_Hx(x, w):
    xw = scalar_product(x,w)
    ww = scalar_product(w,w)

    def get_sub(wi):
        return 2*xw*wi/ww

    sub = np.array([ get_sub(wi) for wi in w])
    hx = np.subtract(x,sub)
    return hx

#funcao para obter um trasformacao Householder M->HMH
def get_transformation(M, HT):

    def left_transformation(M,result,i):
        x = M[:,i:i+1]
        result[:,i:i+1]=get_Hx(x,w)

    #funcao para obter cada coluna da matriz resultante da transformacao Householder pela direita
    def right_transformation(M,result,i):
        x = M[i:i+1,:]
        x = x.T
        result[:,i:i+1]=get_Hx(x,w)

    #variaveis para a trasformacao
    m = np.size(M, 0)
    w= get_wk(M)

    #variavel onde sera armazenado o resultado do produto da matriz Householder pela esquerda
    HM = np.zeros([m,m])

    #iteracao para preenchimento de cada coluna da matriz HM
    for i in range(m):
        left_transformation(M, HM, i)

    #variavel onde sera armazenado o resultado do produto da matriz Householder pela direita
    HMH = np.zeros([m,m])
    
    #iteracao para preencimento de cada coluna das matrizes HMH e HT
    for i in range(m):
        right_transformation(HM,HMH,i)
        right_transformation(HT[n-m:n,n-m:n], HT[n-m:n,n-m:n], i)

    return HMH

#funcao para obtecao da matriz tridiagonalizada simetrica pelo metodo de Householder bem como a matriz H transposta
def get_T(A):
    T = A.copy()
    HT = np.identity(n)

    #iterando n-2 transformacoes Householder
    for i in range(n-2):
        M = T[i:n+1,i:n+1]
        M=get_transformation(M, HT)
        T[i:n+1,i:n+1] = M
    return T, HT

T, HT = get_T(A)

print(T)
print(HT)