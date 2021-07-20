import numpy as np
from utils import *

np.set_printoptions(precision=precision, suppress=True, linewidth=400)

def item_c():
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
    for i, f in enumerate(sorted_frequencies):
        print("\nmodo de vibração para frequência %f:\n" % f)
        print(sorted_modos[i])