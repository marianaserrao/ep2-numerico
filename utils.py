import os
import numpy as np

#definindo caminho relativo para os arquivos de input
script_dir = os.path.dirname(__file__)
inputs_dir = script_dir+"/inputs/"

#precisao dos prints
precision = 5

#################################### EP 1 ################################################

#criterio de parada de iteracoes
err=10**(-6)

#funcao para impressao de listas de forma organizada
def show(items):
    for index,value in enumerate(items):
        print("%s: %s\n" %(index+1,np.round_(value, precision)))

#funcao para decomposicao QR
def get_qr_decomposition(A):
    k=0
    R=A.copy()
    m=np.size(A,0)
    Q = np.identity(m)

    #funcao para obtencao dos valores do seno e cosseno para a rotacao de givens
    def get_givens_params():
        alfa = R[k,k]
        beta = R[k+1,k]
        if abs(alfa)>abs(beta):
            t = -beta/alfa 
            c = 1/((1+t**2)**(1/2))
            s=c*t
        else:
            t = -alfa/beta
            s = 1/((1+t**2)**(1/2))
            c=s*t
        return (c, s)

    #funcao para obtencao de cada transformacao de givens
    def get_q():
        q=np.identity(m)
        q[k,k]=q[k+1,k+1]=c
        q[k,k+1]=-s
        q[k+1,k]=s
        return q 

    #iteracao para decomposicao qr
    while k<=m-2:
        c,s = get_givens_params()
        q=get_q()
        Q = Q@(q.T)
        R=q@R
        k+=1

    return (Q, R) 

#funcao para obtencao auto-valores e auto-vetores com ou sem (hasShift) deslocamento espectral
def qr_shifted(A, H, hasShift, err=err):
    m = n = np.size(A,0)
    V= H
    A_=A.copy()
    u=0
    k=0

    #funcao para normalizar uma matriz
    def normalize(M):
        for i,v in enumerate(M):
            norm = np.linalg.norm(v)
            # print(norm)
            normalized = v/np.linalg.norm(v)
            M[i]=normalized

    #funcao para obtencao do deslocamento
    def get_Shift():
        u=0
        I=np.identity(m)
        if k>0 and hasShift:
            d = (A_[m-2,m-2]-A_[m-1,m-1])/2
            sgn = np.sign(d or 1)
            u=A_[m-1,m-1]+d-sgn*(d**2+A_[m-1,m-2]**2)**(1/2)
        return u*I
    
    #iteracao para cada auto-valor
    while m>=2:

        #iterando decomposicoes QR para 'zerar' beta
        while abs(A_[m-1,m-2])>err:

            #definindo deslocamento espectral
            shift = get_Shift()

            #realizando deslocamento, decomposicao qr e redefinicoes de A_ e V
            A_[0:m,0:m]-=shift
            Q,R = get_qr_decomposition(A_[0:m,0:m])
            A_[0:m,0:m]=R@Q+shift
            V[:,0:m]=V[:,0:m]@Q
            k+=1
        m-=1
    
    eigenvalues = np.diag(A_)
    eigenvectors = V.T
    iterations = k

    return (eigenvalues, eigenvectors, iterations)

#funcao que retorna auto-vetores do gabarito (analiticos)
def get_analitic_eigenvalues(n):
    eigenvalues = []
    for i in range(1,n+1):
        eig = ((1 - np.cos((2*i-1)*np.pi/(2*n+1)))**(-1))/2
        eigenvalues.append(eig)
    return eigenvalues    

############################# TAREFA 1 ###################################

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
def get_transformation(M, HT, n):

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
    n = np.size(A,0)
    T = A.copy()
    HT = np.identity(n)

    #iterando n-2 transformacoes Householder
    for i in range(n-2):
        M = T[i:n+1,i:n+1]
        M, HT_i=get_transformation(M, HT, n)
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

################################# TAREFA 2 #################################

#funcao para criacao da matriz M
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

#funcao para criacao da matriz K
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