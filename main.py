import numpy as np
from utils import *
from tarefa1 import *
from tarefa2 import *

np.set_printoptions(precision=precision, suppress=True, linewidth=400)

#definindo item a ser solucionado
case = int(input('Qual item você gostaria de testar? (0 para a, 1 para b e 2 para treliças)\n'))

if case != 2:
    item_a_b(case)
else:
    item_c()