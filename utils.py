import numpy as np

#criterio de parada de iteracoes
err=10**(-6)

#funcao para impressao de listas de forma organizada
def show(items):
    for index,value in enumerate(items):
        print("%s: %s\n" %(index+1,np.round_(value, 5)))

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