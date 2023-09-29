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
        Demand.cont +=1
        Demand.atraso=atraso


    def __call__(self, t:int):

        self.CU = Demand.cont
    #   self.PR = random.randrange(3,Demand.MAXPR)  below -----------------
        self.AM = random.randrange(1,Demand.MAXAM)
        self.PE = random.randint(1,Demand.MAXPE)
        self.ST = int(0)                  ###received0, ready1, rejected2, produced3, stored4 and delivered5
        self.FT = np.random.randint(0,Demand.MAXFT,self.M)
        if not np.any(self.FT):
            self.FT[0] = 1      ## by_frederic

        ### Tempo ###
        self.F = 0
        for i in range(self.M):
            self.F += int(self.FT[i]>0)

        self.LT = int(self.F/2) + 2                      ###  --- 1.0*self.fun_tau() * self.F
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

        self.SP = self.fun_gamma() ####* Yard.Y   #SPACE CONSUMPTION FACTOR
        self.VA = self.fun_upsilon() ### [0low 1up]
        self.SU = 1- self.fun_sigma() ### [0low 1up]
        self.TP = self.DO - t


    def fun_gamma(self) -> float:
        x = (self.AM*self.F)/(Demand.MAXAM * self.M)
        return x

    def fun_tau(self) -> float:
        x = (self.AM*self.F)/(Demand.MAXAM * self.M)
        return x

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