import numpy as np
import random

M = 3
N = 6
MAXFT = 5
#np.random.seed(6)
#random.seed(5)

FT = np.random.randint(0, MAXFT, M)

# contar quantas Features
F = 0
for i in range(M):
    F += int(FT[i]>0)

# jogar a moeda pra decidir se mudar ou nao quando F == M
joga_moeda = bool(random.randint(0,1))

print(F)
print(FT)
print(joga_moeda)

if F == M and joga_moeda:
    mask = np.random.randint(2, size=M)
    print(mask)
    while mask.sum() == M:
        mask = np.random.randint(2, size=M)
    
    print(mask)
    FT2 = FT * mask
    F2 = 0
    for i in range(M):
        F2 += int(FT2[i]>0)
    print(F2)
    print(FT2)



