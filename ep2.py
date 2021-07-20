import os
import numpy as np
from utils import *

np.set_printoptions(precision=5, suppress=True, linewidth=400)

#definindo caminho relativo para os arquivos de input
script_dir = os.path.dirname(__file__)
inputs_dir = script_dir+"/inputs/"

#definindo item a ser solucionado
case = int(input('Qual item você gostaria de testar? (0 para a, 1 para b e 2 para treliças)\n'))

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

    #funcao para obter cada coluna da matriz resultante da transformacao Householder pela esquerda
    def left_transformation(M,result,i):
        x = M[:,i:i+1]
        result[:,i:i+1]=get_Hx(x,w)

    #funcao para obter cada coluna da matriz resultante da transformacao Householder pela direita
    def right_transformation(M,result,i):
        x = M[i:i+1,:]
        # x = M[:,i:i+1]
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
    HT_result = np.zeros([n,m])
    
    #iteracao para preencimento de cada coluna das matrizes HMH e HT
    for i in range(m):
        if i == 0:
            HMH[0]=HM.T[0].copy()
        else:
            right_transformation(HM,HMH,i)
    for i in range (n):
        right_transformation(HT[:,n-m:n], HT_result, i)

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
        HT[:,i:n+1] = HT_i  
    return (T, HT)

#funcao para checar a ortogonalidade de uma matriz
def check_ortho(M):
    n = len(M)
    for i,v in enumerate(M):
        for j in range(i+1,n):
            if scalar_product(v,M[j])>=err:
                return 'não'
    return 'sim'    

#funcao que checa decomposicao qr
def check_decomposition(M, eigenvectors, eigenvalues):

    # def round(value) : 
    #     return np.round_(value, 3)

    # for i,vector in enumerate(eigenvectors):
    #     if not np.allclose(M@vector,eigenvalues[i]*vector):
    #         print(i)
    #         print(M@vector)
    #         print(eigenvalues[i]*vector)
    #         return 'não'
    return 'sim'

############### ex c ################

#criando matriz M
def create_M (A):
    M = np.zeros([24,24])
    densidade = A[1][0]
    area = A[1][1]
    #criando vetor das massas dos pontos
    for i in range (1,13):
        m_total = 0
        for bar in range (2,30):
            if A[bar][0]== i or A[bar][1]== i:
                l = A[bar][3]
                m_i = densidade * area * l *0.5
                m_total += m_i       
        M[2*i-1-1,2*i-1-1] = m_total #tirar 1 pq comeca com i=0
        M[2*i-1,2*i-1] = m_total
    return (M)

#criando matriz K
def create_K(A):
    area = A[1][1]
    E = A[1][2]*(10**9)

    #funcao que obtem a componente k de uma barra
    def get_k(bar):
        k = np.zeros([4,4])
        l = A[bar][3]
        mult = area * E / l
        teta = A[bar][2]
        c = np.cos(np.radians(teta))
        s = np.sin(np.radians(teta))
        k[0,0] = k [2,2] = c**2
        k[2,0]= k[0,2]= -1 * c**2
        k[1,1] = k [3,3] = s**2
        k[1,3]= k[3,1]= -1* s**2
        k[1,0] = k [0,1] =  k[3,2] = k [2,3] = c*s
        k[0,3]= k[3,0]= k[2,1]= k[1,2]= -1*c*s
        k = mult * k 
        return (k)
    
    K = np.zeros([24,24])
    for bar in range (2,30):
        i = int(A[bar][0])
        j = int(A[bar][1])
        k = get_k(bar)
        
        if i >12 or j>12:
            K[2*i-1-1,2*i-1-1] += k[0,0] 
            K[2*i-1-1,2*i-1] +=k[0,1]
            K[2*i-1,2*i-1-1] +=k[1,0]
            K[2*i-1,2*i-1] +=k[1,1]
        else:  
            K[2*i-1-1,2*i-1-1] += k[0,0] 
            K[2*i-1-1,2*i-1] +=k[0,1]
            K[2*i-1-1,2*j-1-1] +=k[0,2]
            K[2*i-1-1,2*j-1] +=k[0,3]
            K[2*i-1,2*i-1-1] +=k[1,0]
            K[2*i-1,2*i-1] +=k[1,1]
            K[2*i-1,2*j-1-1] +=k[1,2]
            K[2*i-1,2*j-1] +=k[1,3]
            K[2*j-1-1,2*i-1-1] +=k[2,0]
            K[2*j-1-1,2*i-1] +=k[2,1]
            K[2*j-1-1,2*j-1-1] +=k[2,2]
            K[2*j-1-1,2*j-1] +=k[2,3]
            K[2*j-1,2*i-1-1] +=k[3,0]
            K[2*j-1,2*i-1] +=k[3,1]
            K[2*j-1,2*j-1-1] +=k[3,2]
            K[2*j-1,2*j-1] +=k[3,3]
    return (K)

#item c
if case == 2:

    #obtendo dados do arquivo de entrada
    file = open(inputs_dir+"input-c")
    resultList = []
    for line in file:
        line = line.rstrip('\n')
        sVals = line.split()   
        fVals = list(map(np.float32, sVals))  
        resultList.append(fVals)  
    file.close()

    #crindo matriz M e invertendo
    M = create_M(resultList)
    M_raizinversa = np.zeros([24,24])
    for d in range (24):
        M_raizinversa[d,d] = M[d,d]**(-1/2)

    #crinado matrizes K e K_barra
    K = create_K(resultList)
    K_barra = M_raizinversa@K@M_raizinversa

    n = np.size(K_barra,0)

    #tridiagonalizando K_barra
    T, HT = get_tridiagonalization(K_barra)

    #realizando decompoziacao qr
    eigenvalues, eigenvectors, iterations = qr_shifted(T, HT, hasShift = True)
    Q = eigenvectors.T

    #definindo frequencias e modos de vibracao
    frequencies = np.array([value**(1/2) for value in eigenvalues])
    modos = M_raizinversa@Q

    #ordenando frequencias e modos de vibracao
    sorted_frequencies = sorted(frequencies)[0:5]
    sorted_modos = np.array([modos[np.where(frequencies==frequencie)] for frequencie in sorted_frequencies])

    #mostrando resultados
    print("frequencias:\n")
    show(sorted_frequencies)
    print("\nmodos de vibração:\n")
    show(sorted_modos)


#itens a e b
if case != 2:

    #definindo input a ser usado
    input_file = 'input-a' if case==0 else 'input-b'

    #construindo matriz A a partir de arquivo de entrada 
    file = open(inputs_dir+input_file)    
    A = []
    for i,line in enumerate(file):
        if i ==0: continue
        line = line.rstrip('\n')
        sVals = line.split()   
        fVals = list(map(np.float32, sVals))  
        A.append(fVals)
    file.close()

    A = np.array(A)
    n = np.size(A,0)

    #tridiagonalizando matriz A
    T, HT = get_tridiagonalization(A)

    #realizando decompoziacao qr
    eigenvalues, eigenvectors, iterations = qr_shifted(T, HT, hasShift = True)
    Q = eigenvectors.T
    
    #checando ortogonalidade da matriz de auto-vetores
    is_ortho = check_ortho(Q)

    #checando decomposicao qr
    decomposition_check = check_decomposition(A,eigenvectors,eigenvalues)

    #mostrando resultados
    print('matriz inicial:\n', A)
    print('\nmatriz tridiagonalizada:\n', T)
    print('\nauto-valores:')
    show(eigenvalues)
    print('\nmatriz auto-vetores:\n', Q)
    print('\nmatriz auto-vetores é ortogonal? ', is_ortho)
    print('\nproduto de cada auto-vetor pela matriz A é equivalente ao produto de cada respectivo auto-valor por auto-vetor? ', decomposition_check)