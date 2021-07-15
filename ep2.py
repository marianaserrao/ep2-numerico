import numpy as np
from utils import *

case = int(input('Qual item você gostaria de testar? (0 para a e 1 para b)\n'))

A = np.array([[2,4,1,1],[4,2,1,1],[1,1,1,2],[1,1,2,1]]).astype(np.float32)
n = np.size(A,0)
if case:
    n=20
    A=np.zeros([n,n])
    for (i,j),x in np.ndenumerate(A):
        if j<=i:
            A[i,j]=n-i
        else:
            A[i,j]=n-j

#funcao para obter uma linha ou coluna de matriz
def get_vector(A, index, column=False):
    if column:
        return A[:,index:index+1]
    else:
        return A[index:index+1,:]

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

#funcao para obter o produto escalar de dois vetores
def scalar_product(x,y):
    value = 0
    for i, xi in enumerate(x):
        value+=xi*y[i]
    return value

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
        new_line = get_Hx(x,w).T
        result[i:i+1, :]=new_line

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
    HT_result = np.zeros([m,m])
    
    #iteracao para preencimento de cada coluna das matrizes HMH e HT
    for i in range(m):
        right_transformation(HT[n-m:n,n-m:n], HT_result, i)
        if i == 0:
            HMH[0]=HM.T[0].copy()
        else:
            right_transformation(HM,HMH,i)

    return (HMH, HT_result)

#funcao para obtecao da matriz tridiagonalizada simetrica pelo metodo de Householder bem como a matriz H transposta
def get_tridiagonalization(A):
    T = A.copy()
    HT = np.identity(n)

    #iterando n-2 transformacoes Householder
    for i in range(n-2):
        M = T[i:n+1,i:n+1]
        M, HT_i=get_transformation(M, HT)
        T[i:n+1,i:n+1] = M
        HT[i:n+1,i:n+1] = HT_i
    return (T, HT)

T, HT = get_tridiagonalization(A)

# print(T)
# print(HT)

#realizando decompoziacao qr
eigenvalues, eigenvectors, iterations = qr_shifted(T, HT, hasShift = True)
Q = eigenvectors.T

#funcao que checa decomposicao qr
def check_decomposition(M, eigenvectors, eigenvalues):
    for i,vector in enumerate(eigenvectors):
        if not np.array_equal(M@vector,eigenvalues[i]*vector):
            return 'não'
    return 'sim'

#checando decomposicao qr
decomposition_check = check_decomposition(T,eigenvectors,eigenvalues)

#funcao para checar a ortogonalidade de uma matriz
def check_ortho(M):
    n = len(M)
    for i,v in enumerate(M):
        for j in range(i+1,n):
            if scalar_product(v,M[j])>=err:
                return 'não'
    return 'sim'    

#checando ortogonalidade da matriz de auto-vetores
is_ortho = check_ortho(Q)

#mostrando resultados
print('matriz inicial:\n', A)
print('\nmatriz tridiagonalizada:\n', T)
print('\nauto-valores:')
show(eigenvalues)
print('\nauto-vetores:\n', Q)
print('\nproduto de cada auto-vetor pela matriz tridiagonal é equivalente ao produto de cada respectivo auto-valor por auto-vetor? ', decomposition_check)
print('\nmatriz auto-vetor é ortogonal? ', is_ortho)