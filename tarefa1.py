import numpy as np
from utils import *

np.set_printoptions(precision=precision, suppress=True, linewidth=400)

def item_a_b(item):
    #definindo input a ser usado
    input_file = 'input-a' if item==0 else 'input-b'

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

    #tridiagonalizando matriz A
    T, HT = get_tridiagonalization(A)

    #realizando decompoziacao qr
    eigenvalues, eigenvectors, iterations = qr_shifted(T, HT, hasShift = True)
    Q = eigenvectors.T
    
    #checando ortogonalidade da matriz de auto-vetores
    is_ortho = check_ortho(Q)

    #checando decomposicao qr
    decomposition_check, max_err = check_decomposition(A,eigenvectors,eigenvalues)

    #mostrando resultados
    print('matriz inicial:\n', A)
    print('\nmatriz tridiagonalizada:\n', T)
    print('\nauto-valores:')
    show(eigenvalues)
    print('\nauto-valores fornecidos (ou sua fórmula analítica) no enunciado:')
    if item:
        show(get_analitic_eigenvalues(20))
    else:
        show([7, 2, -1, 2])
    print('\nmatriz auto-vetores:\n', Q)
    print('\nmatriz auto-vetores é ortogonal? ', is_ortho)
    print('\nproduto de cada auto-vetor pela matriz A é equivalente ao produto de cada respectivo auto-valor por auto-vetor? ', decomposition_check)
    if decomposition_check:
        print('\no máximo erro foi da validação acima foi %f \n' % max_err)