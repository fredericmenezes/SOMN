import random
import numpy as np
from scipy.stats import poisson

class Demand:

    cont=0      # by_frederic ---> mudei de 1 para 0
    load=0
    reject=0         # rejeitar
    reject_w_waste=0 # rejeitar com lixo
    #atraso=None

    # Somn(Y=10,M=10,N=10,MAXDO=10,MAXAM=3,MAXPR=2,MAXPE=10,MAXFT=5,MAXMT=3,MAXTI=2,MAXEU = 5, atraso=atraso)
    def __init__(self,
                 M:int,
                 N:int,
                 MAXDO:int,
                 MAXAM:int,
                 MAXPR:float,
                 MAXPE:int,
                 MAXFT:int,
                 MAXMT:int,
                 MAXTI:int,
                 MAXEU:int,
                 t: int,
                 atraso: int = None):

        Demand.M=M
        Demand.N=N
        Demand.MAXDO=MAXDO
        Demand.MAXAM=MAXAM
        Demand.MAXPR=MAXPR
        Demand.MAXPE=MAXPE
        Demand.MAXFT=MAXFT
        Demand.MAXMT=MAXMT
        Demand.MAXTI=MAXTI
        Demand.MAXEU=MAXEU
        Demand.EU = np.random.random(M)*MAXEU
        self.ST = int(-1)                  ### free(-1) received(0), ready(1), rejected(2), produced(3), stored(4) and delivered(5)
        self.action = 0
        
        Demand.cont +=1
        self.CU = Demand.cont
        Demand.atraso=atraso


    def __call__(self, t:int):

        
    #   self.PR = random.randrange(3,Demand.MAXPR)  below -----------------
        self.AM = random.randrange(1,Demand.MAXAM)
        self.PE = random.randint(1,Demand.MAXPE)
        self.ST = int(0)                  ###received0, ready1, rejected2, produced3, stored4 and delivered5
        
        # Escolhe aleatoriamente as features (tempos para cada maquina)
        # Exemplo: self.FT = array([2, 4, 0, 1, 3]) #valores de: 0 a (MAXFT -1)
        self.FT = np.random.randint(0,Demand.MAXFT,self.M)

        # enquanto der tudo zero, escolhe randomicamente novamente
        # self.FT = array([0, 0, 0, 0, 0]) #np.any(self.FT) == False
        while not np.any(self.FT):
            self.FT = np.random.randint(0,Demand.MAXFT,self.M)

        # mask_FT eh um vetor de zeros e uns indicando quais features estao ativas (maquinas usadas)
        # Exemplo: self.mask_FT = array([1, 1, 0, 1, 1])
        self.mask_FT = self.FT.copy()
        self.mask_FT[self.mask_FT > 0] = 1

        # contar quantas Features estao sendo usadas (total de maquinas usadas)
        self.F = self.mask_FT.sum()

        # jogar a moeda pra decidir se mudar ou nao quando F == M (usando todas as maquinas)
        # o objetivo disso é equilibrar a base de dados
        # porque a probabilidade de (F==M) é muito alta.
        joga_moeda = bool(random.randint(0,1))
        if self.F == self.M and joga_moeda:

            # mask_clear multiplica self.FT para apagar alguns valores
            mask_clear = np.random.randint(2, size=self.M)
            # enquanto der tudo zero ou tudo um, escolhe randomicamente novamente
            while mask_clear.sum() == self.M or mask_clear.sum() == 0:
                mask_clear = np.random.randint(2, size=self.M)
            
            self.FT = self.FT * mask_clear
            
            self.mask_FT = self.FT.copy()
            self.mask_FT[self.mask_FT > 0] = 1

            # contar quantas Features estao sendo usadas (total de maquinas usadas)
            self.F = self.mask_FT.sum()

        # self.LT = int(self.F/2) + 2                      ###  --- 1.0*self.fun_tau() * self.F
        self.LT = self.fun_tau()
        if self.atraso >=0:   ### Não é mais None (-1 para habilitar poisson)
            self.real_LT = self.LT + self.atraso
        else:
            self.real_LT = poisson.rvs(mu=self.LT) # by_frederic
        self.DI = t
        self.DO = t + self.LT + random.randint(0,Demand.MAXDO)

        self.CO = 0.0
        for j in range(Demand.M):
            self.CO += self.FT[j] * Demand.EU[j]
        #self.CO = self.AM * self.CO  -- custo sem o amount
        self.PR = Demand.MAXPR*self.CO  ### LUCRO EH 2X CUSTO  self.PR = Demand.MAXPE  (by fred)

        self.SP = self.fun_gamma() ####* 'cpu'.Y   #SPACE CONSUMPTION FACTOR
        self.VA = self.fun_upsilon() ### [0low 1up]
        self.SU = 1- self.fun_sigma() ### [0low 1up]
        self.TP = self.DO - t


    def fun_gamma(self) -> float:
        x = (self.AM*self.F)/((Demand.MAXAM -1) * self.M)
        return x

    # def fun_tau(self) -> float:
    #     x = (self.AM*self.F)/((Demand.MAXAM-1) * self.M)
    #     return x
    def fun_tau(self) -> float:
        x = self.AM * self.FT
        return x.sum()

    def fun_upsilon(self) -> float:
        x = self.F/self.M
        return x

    def fun_sigma(self) -> float:
        x = self.F/self.M
        return x

#  def fun_beta(self, IN, OU) -> float:
#    x=0
#    for i in range(self.M):
#      if IN[i]==OU[i]:
#        x+=1
#    x = x/self.M
#    return x

#  def calculate_statics(self):