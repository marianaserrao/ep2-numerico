import numpy as np
from utils import *
from tarefa1 import *
from tarefa2 import *

np.set_printoptions(precision=precision, suppress=True, linewidth=400)

#definindo item a ser solucionado
case = int(input('Qual item você gostaria de testar? (0 para a, 1 para b e 2 para treliças)\n'))

#definindo se deseja-se usar  outro arquivo de entrada
input = input('Adicione o caminho absoluto do input que deseja utilizar ou enter vazio para utilizar o padrão:\n')

if case != 2:
    item_a_b(case, input)
else:
    item_c(input)