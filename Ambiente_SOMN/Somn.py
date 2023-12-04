from JobShop.JobShop import JobShop
from Ambiente_SOMN.Demand import Demand
from Ambiente_SOMN.Yard import Yard
from Stablebaselines3.OnPolicyAlgirithm import OnPolicyAlgorithm
import matplotlib.pyplot as plt

from numpy.random.mtrand import seed
# a biblioteca gym mudou
import gymnasium as gym
from gymnasium import spaces  # Discrete, Box, Tuple,  Dict
from gymnasium import Env
from heapdict import heapdict

# biblioteca stable_baselines3
from stable_baselines3.common.vec_env import DummyVecEnv
#from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env
# from stable_baselines3 import PPO, A2C, DQN, DDPG, TD3

# outras bibliotecas
import os
import numpy as np
import random
import numpy as np
from absl import flags
from scipy.stats import poisson
import torch
import wandb


class Somn(Env):

    """Custom Environment that follows gym interface."""
    def __init__(
        self,
        M: int,
        N: int,
        Y: int,
        MAXDO: int,
        MAXAM: int,
        MAXPR: int,
        MAXPE: int,
        MAXFT: int,
        MAXMT: int,
        MAXTI: int,
        MAXEU: int,
        #seed: int,
        atraso: int,
        objetivo: int
    ):
        super(Somn).__init__()

        Somn.obj_list = ['pr', 'va', 'su']
        Somn.priorq = [heapdict() for objetivo in Somn.obj_list]
        Somn.objetivo = objetivo
        # Somn.instance = JobShop()
        # Somn.priorqsu = heapdict()
        # Somn.priorqva = heapdict()
        Somn.time = 1
        
        # self.reward = 0.0
        # self.lucro = 0.0
        # self.variabilidade = 0.0
        # self.sustentabilidade = 0.0



        self.M = M
        self.N = N
        self.Y = Y
        self.MAXDO = MAXDO
        self.MAXAM = MAXAM
        self.MAXPR = MAXPR
        self.MAXPE = MAXPE
        self.MAXFT = MAXFT
        self.MAXMT = MAXMT
        self.MAXTI = MAXTI
        self.MAXEU = MAXEU
        # self.MT = np.random.randint(0,MAXFT,M)



       
        self.EU = np.random.random(M) * MAXEU
        self.BA = np.random.randint(10, 10*MAXFT, M)
        self.IN = np.random.randint(0, MAXFT, M)
        self.OU = np.random.randint(0, MAXFT, M)
        #self.seed = seed
        self.atraso = atraso # (by_frederic)
        # self.DE_state = np.zeros((N,5))

        # print('Inicializado', M, N , Y)

        self.DE = [
            Demand(
                M, N, MAXDO, MAXAM, MAXPR, MAXPE, MAXFT, MAXMT, MAXTI, MAXEU, Somn.time, self.atraso
            )
            for _ in range(N)
        ]
        self.YA = Yard(self.Y)

        ######################
        #      lb e ub       #
        ######################
        """
        (lb=lowerbound ub=upperbound) para o espaco de Observacao e Acao
        """

        # ST varia de -2 a 5
        self.lb_ST = -2
        self.ub_ST = 5

        # time varia de 1 a 100 (era de 1 ate 10*MAXDO + M)
        self.lb_time = 1
        # self.ub_time = 10 * self.MAXDO + self.M
        self.ub_time = 100

        # LT varia de 2 a (M/2 + 2)
        self.lb_LT = 1
        # self.ub_LT = int(self.M / 2) + 2    #### ACMO LT AFETADO POR LT(M) + CARGA(N)
        #self.ub_LT = self.M + self.N
        self.ub_LT = self.M * (self.MAXFT - 1) * (self.MAXAM - 1)

        # DO varia de 3 a (ub_time + ub_LT + MAXDO)
        self.lb_DO = 3
        self.ub_DO = self.ub_time + self.ub_LT + self.MAXDO
        # DI varia de 1 a (ub_time + ub_LT + MAXDO)
        self.lb_DI = 1
        self.ub_DI = self.ub_time

        # TP varia de 2 a (ub_time + ub_LT + 2) onde 2 e um ruido (troquei 2 pela distribuicao de poisson)
        self.lb_TP = 2
        p = [poisson.rvs(mu=self.ub_LT) for _ in range(10000)]
        self.ub_TP = self.ub_time + self.ub_LT + max(p)

        # CO varia de 0 a (M * (MAXFT-1) * MAXEU)
        self.lb_CO = 0
        self.ub_CO = self.M * (self.MAXFT-1) * self.MAXEU

        # PR varia de 0 a (M * (MAXFT-1) * MAXEU) * MAXPR
        self.lb_PR = 0
        self.ub_PR = self.M * (self.MAXFT-1) * self.MAXEU * self.MAXPR

        

        # AM varia de 1 a MAXAM - 1
        self.lb_AM = 1
        self.ub_AM = self.MAXAM - 1

        # SP varia de 0 a 1
        self.lb_SP = 0
        self.ub_SP = 1

        # PE varia de 0 a MAXPE
        self.lb_PE = 0
        self.ub_PE = self.MAXPE

        # VA varia de 0 a 1
        self.lb_VA = 0
        self.ub_VA = 1

        # SU varia de 0 a 1
        self.lb_SU = 0
        self.ub_SU = 1




        # MT varia de 0 a MAXFT
        self.lb_MT = np.array([0 for _ in range(self.M)]).astype(np.int64)
        self.ub_MT = np.array([MAXFT-1 for _ in range(self.M)]).astype(np.int64)
        # EU varia de 0 a MAXEU
        self.lb_EU = np.array([0 for _ in range(self.M)]).astype(np.float64)
        self.ub_EU = np.array([self.MAXEU for _ in range(self.M)]).astype(np.float64)
        # BA varia de 0 a MAXFT
        self.lb_BA = np.array([0 for _ in range(self.M)]).astype(np.int64)
        self.ub_BA = np.array([100*self.MAXFT-1 for _ in range(self.M)]).astype(np.int64)
        # IN varia de 0 a MAXFT
        self.lb_IN = np.array([0 for _ in range(self.M)]).astype(np.int64)
        self.ub_IN = np.array([self.MAXFT-1 for _ in range(self.M)]).astype(np.int64)
        # OU varia de 0 a MAXFT
        self.lb_OU = np.array([0 for _ in range(self.M)]).astype(np.int64)
        self.ub_OU = np.array([self.MAXFT-1 for _ in range(self.M)]).astype(np.int64)

        # OU varia de 0 a MAXFT
        self.lb_FT = np.array([0 for _ in range(self.M)]).astype(np.int64)
        self.ub_FT = np.array([self.MAXFT-1 for _ in range(self.M)]).astype(np.int64)

        # yard varia de 1 a self.Y
        self.lb_yard = 0
        self.ub_yard = self.Y

        # yard varia de 1 a self.Y
        self.lb_load = 0
        self.ub_load = 100

        # lb e ub--- segunda versao (sem a coluna com os valores de Somn.time)
        # self.lb = np.array([[self.lb_ST, self.lb_LT, self.lb_DO, self.lb_TP] for _ in range(self.N)])
        # self.ub = np.array([[self.ub_ST, self.ub_LT, self.ub_DO, self.ub_TP] for _ in range(self.N)])

        # # lb e ub--- primeira versao (com a coluna com os valores de Somn.time)
        # self.lb = np.array([[self.lb_ST, self.lb_time, self.lb_LT, self.lb_DO, self.lb_TP] for _ in range(self.N)])
        # self.ub = np.array([[self.ub_ST, self.ub_time, self.ub_LT, self.ub_DO, self.ub_TP] for _ in range(self.N)])



        ######################
        #      Espacos       #
        ######################
        """
        Precisa mudar o espaco de acao
        de acordo com o algoritmo utilizado
        """

        # accept to produce or reject
        # self.action_space = spaces.Box(0, 4, shape=(1,)) # usar o TD3
        #self.action_space = spaces.Discrete(self.MAXDO)  # usar com o PPO, DQN, A2C
        self.action_space = spaces.Discrete(self.ub_time)
        

        # Espaco de observacao (como ficam as demandas depois da acao)
        # self.observation_space = spaces.Box(self.lb, self.ub, dtype=int)
        # self.observation_space = spaces.Dict({'tempo':spaces.Box(self.lb_time, self.ub_time, shape=(1,), dtype=int),
        #                                 'estado': spaces.Box(self.lb, self.ub, dtype=int)})      # versao para MultiInputPolicy
        # self.observation_space = spaces.Dict({'time':spaces.Box(self.lb_time, self.ub_time, shape=(1,), dtype=int),
        #                                'MT': spaces.Box(self.lb_MT, self.ub_MT, dtype=int),
        #                                'EU': spaces.Box(self.lb_EU, self.ub_EU, dtype=float),
        #                                'BA': spaces.Box(self.lb_BA, self.ub_BA, dtype=int),
        #                                'IN': spaces.Box(self.lb_IN, self.ub_IN, dtype=int),
        #                                'OU': spaces.Box(self.lb_OU, self.ub_OU, dtype=int),
        #                                'state': spaces.Box(self.lb, self.ub, dtype=int)})          # versao para MultiInputPolicy

        self.observation_space = spaces.Dict(
            {
                "time": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float64),
                "MT": spaces.Box(low=0.0, high=1.0, shape=(self.M,), dtype=np.float64),
                "EU": spaces.Box(low=0.0, high=1.0, shape=(self.M,), dtype=np.float64),
                "BA": spaces.Box(low=0.0, high=1.0, shape=(self.M,), dtype=np.float64),
                "IN": spaces.Box(low=0.0, high=1.0, shape=(self.M,), dtype=np.float64),
                "OU": spaces.Box(low=0.0, high=1.0, shape=(self.M,), dtype=np.float64),
                "DE_state": spaces.Box(
                    low=0.0, high=1.0, shape=(self.N, 12), dtype=np.float64
                ),
                "FT_state": spaces.Box(low=0.0, high=1.0, shape=(self.N,self.M), dtype=np.float64),
                "yard": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float64),
                "load": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float64),
            }
        )  # versao para MultiInputPolicy Normalizada


    ######################
    #      funcoes       #
    ######################

    # recebe um atributo por exemplo 'LT' ou 'real_LT' e devolve os valores
    # de Lead Time das Demandas ou Real Lead Time, conforme o atributo passado.
    def get_Demands_Attr(env, atributo):
        d = [getattr(demandas, atributo) for demandas in env.DE]
        return d
    def get_atraso(self):
        a = self.atraso
        return a
    def get_reward(self):
        rw = self.reward
        return rw
    def get_lucro(self):
        lu = self.lucro
        return lu
    def get_variabilidade(self):
        va = self.variabilidade
        return va
    def get_sustentabilidade(self):
        su = self.sustentabilidade
        return su

    # Normaliza o valor dentro do range passado como parametro
    def normaliza(self, x, min, max):
        # verificar se eh um escalar ou um np.array
        # se for um escalar evitar a divisao por zero.
        if type(x).__module__ != np.__name__:
            if max == min: return 1

        x_norm = np.array((x - min) / (max - min)).astype(np.float64)
        return x_norm

    def readDemand(self):
        for i in range(Demand.N):
            if (
                self.DE[i].ST == -1
            ):  # or self.DE[i].ST == 0: ZERO não pode ser status de livre
                self.DE[i](Somn.time)

    def match_demand_with_inventory(self) -> bool:
        matched = False
        for i in range(Demand.N):
          if self.DE[i].ST == 0: ## SÓ PODE DAR MATCH DEMANDAS CHEGADAS
            if self.Y > 0: # ALTERAÇÃO PARA TESTE DE YARD == 0
                if self.YA.inYard(self.DE[i].FT):

                    self.YA.remove_yard(self.DE[i].FT)

                    self.DE[i].ST = 5  ## delivered p/ contar lucro
                    matched = True

                
                # for y in range(Yard.cont):
                #     match = 0
                #     # print('Y...', y, 'YA=', YA[y].yard,Yard.cont, 'l=', limiar)
                #     for j in range(Demand.M):
                #         # print('Y(y,j):', y,j, 'Y x D:', self.YA[y].yard[j],self.DE[i].FT[j], 'cont:', Yard.cont, 'l x m:', limiar, match)
                #         if self.DE[i].FT[j] > 0:
                #             if self.DE[i].FT[j] == self.YA[y].yard[j]: #mudança de <= para == 
                #                 match = match + 1
                #         # se for ZERO então não pode ter a caracteristica
                #         else:
                #             if self.YA[y].yard[j] == 0:
                #                 match = match + 1

                #     if match >= limiar:
                #         # print("\n Match: Casou", Yard.cont)
                #         self.YA[y].yard = self.YA[Yard.cont - 1].yard  ## apaga o registro de match com o último da lista
                #         Yard.cont -= 1    ### FRED NAO DEIXAR BAIXAR DE ZERO
                #         self.DE[i].ST = 5  ## delivered p/ contar lucro
                #         matched = True

                

        return matched

    def product_schedulingold(self, t: int, action):
        for i in range(self.N):
            if self.DE[i].ST == 1:
                
                if self.DE[i].DO > (t + self.DE[i].LT + action):
                    self.DE[i].ST = 3  ## produced status --- remember to run time for each case
                    self.OU -= self.DE[i].FT  ## CONSOME OS RECURSOS
                    ################################################
                    #                                              #
                    #                  atraso                      #
                    #                                              #
                    ################################################

                    #Somn.producing = min(self.M, Somn.producing + 1)
                    #Somn.producing = Somn.producing + 1
                    #Demand.load = Demand.load + 1
                    #self.DE[i].TP = t + self.DE[i].LT + poisson.rvs(mu=2) -- LOAD NA CONTA
                    if self.atraso >= 0:
                        self.DE[i].real_LT = self.DE[i].LT + self.atraso
                    else:
                        self.DE[i].real_LT = poisson.rvs(mu=(self.DE[i].LT+Demand.load)) # by_frederic

                    self.DE[i].TP = t + self.DE[i].real_LT

                    # if self.atraso > 5:
                    #     self.DE[i].TP = t + self.DE[i].LT + poisson.rvs(mu=2)
                    # else:
                    #     self.DE[i].TP = t + self.DE[i].LT + self.atraso  # ruido  --- trocar por distribuição poison --- ou por algo que dependa de AM random.randint(1,Demand.MAXTI)
                    #     # print('\n **** PRODUCED because', self.DE[i].DO, '>', t + self.DE[i].LT + action)
                else:
                    #Somn.producing = max(0, Somn.producing - 1)
                    #Somn.producing = Somn.producing - 1
                    self.DE[i].ST = 2  ## rejected status
                    self.OU -= self.DE[i].FT  ### libera do buffer de produção
                    self.BA += self.DE[i].FT  ## devolve para o saldo para os próximos
                    # print('\n **** REJECTED by DO', self.DE[i].DO, ' <= DI+LT+act', t , self.DE[i].LT , action)

    def product_scheduling(self, t: int, action):
        
        for _ in range(len(Somn.priorq[self.objetivo])):  ### ACMO UTILIZAR 3 FILAS E ESCOLHER UMA DELAS AQUI
            # objetivo  0: price, 
            #           1: variabilidade, 
            #           2: sustentabilidade
            obj = Somn.priorq[self.objetivo].popitem()
            i = obj[0]
            if i > 0:
                if self.DE[i].ST == 1:  ## DE[I].ST VAI SER SEMPRE 1 PORQUE VEM DA FILAP
### COPY JOB TO JOBSHOP SCHEDULING
                #   for j in range(self.M):
                #     if self.DE[i].FT[j]!= 0:
                #     #   Somn.instance.InsertJobs(i, j, self.DE[i].FT[j])
                #       flag = 1
###
                    self.DE[i].action = action
                    if self.DE[i].DO > (t + self.DE[i].LT + action):
                        self.DE[i].ST = 3  ## produced status --- remember to run time for each case
                        self.OU -= self.DE[i].FT  ## CONSOME OS RECURSOS
                        Demand.load = Demand.load + 1
                        self.DE[i].real_LT = poisson.rvs(mu=(self.DE[i].LT+Demand.load)) # by_frederic
                        self.DE[i].TP = t + self.DE[i].real_LT
                    else:
                        Demand.reject = Demand.reject + 1
                        self.DE[i].ST = 2  ## rejected status
                        self.OU -= self.DE[i].FT  ### libera do buffer de produção
                        self.BA += self.DE[i].FT  ## devolve para o saldo para os próximos
## se formou buffer, resolve para comparar depois
        # if flag ==1:
        #   Somn.instance.BuildModel()
        #   Somn.instance.Solve()
        #   Somn.instance.Output()  ## precisa salvar a lista de resultados


    def product_destination(self, t: int):
        for i in range(Demand.N):
            if self.DE[i].ST == 3:
                if self.DE[i].TP < t:  ### TP eh resultado de LT(#f) + RAND
                    #Somn.producing = Somn.producing - 1
                    Demand.load = Demand.load - 1
                    if t < self.DE[i].DO:
                        self.DE[i].ST = 5  ## produced status --- remember to run time for each case
                        # print("\n Destination: Enviou", Yard.cont)
                    else:
                        self.DE[i].ST = 4  ## stored status
                        
                        # VALIDAÇÃO DE TESTE PARA YARD = 0 #####################################################
                        if self.Y == 0:
                            self.DE[i].ST = -2  ## NAO CABE ... REJEITADO COM GERAÇÃO DE LIXO (CASO MAIS GRAVE)
                            Demand.reject_w_waste = Demand.reject_w_waste + 1
                        ##########################################################################

                        elif self.YA.cont < self.YA.Y:
                            self.YA.yard.append(self.DE[i].FT)

                            mask_YA = self.DE[i].FT.copy()
                            mask_YA[mask_YA > 0] = 1

                            self.YA.mask_YA.append(mask_YA)
                            self.YA.cont = len(self.YA.yard)
                            

                            # self.YA[Yard.cont].yard = self.DE[i].FT
                            # self.YA[Yard.cont].mask_YA = self.DE[i].mask_FT
                            # Yard.cont += 1
                            # print("\n Destination: Armazenou no YARD", Yard.cont)

                        else:
                            self.DE[i].ST = -2  ## NAO CABE ... REJEITADO COM GERAÇÃO DE LIXO (CASO MAIS GRAVE)
                            Demand.reject_w_waste = Demand.reject_w_waste + 1
                            # print(f'\n\n\n\nReject total: {Demand.reject_w_waste} \n\n\n\n')
    

    def stock_covers_demand(self):
        covered = True
        for i in range(self.N):
            if self.DE[i].ST == 0:
                DF = self.BA - self.DE[i].FT
                OR = np.array(
                    [abs(i) if i < 0 else 0 for i in DF]
                )  # O QUE PRECISA SER COMPRADO
                # print('\n ORDER from ', DF, ':', OR)
                if not np.any(OR):
                    self.DE[i].ST = 1

## ACMO SETAR A PRIORIDADE
# BY PROFIT -- escolher um ???? e comentar o outro
#                    Somn.priorqpr[i] = int(1/(self.DE[i].AM * self.DE[i].PR))
# BY SUSTAIN
#                    Somn.priorqsu[i] = 1 - int(self.DE[i].SU)  ## [0low 1up]
# BY VARIATI
#                    Somn.priorqva[i] = 1 - int(self.DE[i].VA)  ## [0low 1up]
                    # if self.object == 1:
                    #   Somn.priorqpr[i] = 1/(self.DE[i].AM * self.DE[i].PR)
                    # elif self.object == 2:
                    #   Somn.priorqsu[i] = 1 - self.DE[i].SU  ## [0low 1up]
                    # else:
                    #   Somn.priorqva[i] = 1 - self.DE[i].VA  ## [0low 1up]
# STOCK ISSUES

 #                   Somn.priorqpr[i] = (1 - self.DE[i].SU)
                    # fila de prioridade 0 = price
                    Somn.priorq[0][i] = 1/(self.DE[i].AM * self.DE[i].PR)
                    # fila de prioridade 1 = variabilidade
                    Somn.priorq[1][i] = 1 - self.DE[i].VA
                    # fila de prioridade 2 = sustentabilidade
                    Somn.priorq[2][i] = 1 - self.DE[i].SU

                    # print ((1 - self.DE[i].SU))
                    self.BA -= np.array(DF)  ### ATUALIZA O SALDO
                    self.OU += np.array(DF)  ### ATUALIZA A SAÍDA
                    # print('\n balance:', self.BA,  'because not buying',self.OU)
                else:
                    covered = False
                    self.IN += np.array(OR)  ## ATUALIZA O TOTAL DE COMPRAVEIS
                    # print('\n balance: ', self.BA, 'because buying',OR, 'accumulating', self.IN)
        return covered

    # def order_raw_material(self, t: int):
    # self.IN = [random.randint(0,i) if i > 0 else 0 for i in self.IN]
    # return self.IN

    def eval_final_states(self) -> float:
        rw_lu = 0.0
        rw_va = 0.0
        rw_su = 0.0

        variabilidade = 0.0
        sustentabilidade = 0.0

        F = 0

        acoes = []
        atrasos_reais = []

        patio = (self.YA.cont/self.YA.Y)*100

        totReward = 0.0
        totPenalty = 0.0

        for i in range(self.N):

            if self.DE[i].ST == 2:
                totPenalty += 0
                acoes.append(self.DE[i].action)
                atrasos_reais.append(abs(self.DE[i].real_LT - self.DE[i].LT))
                # penalidade por estar no ambiente
                # if totReward > 0:
                #     totPenalty += abs(self.DE[i].action - abs(self.DE[i].real_LT - self.DE[i].LT))
                self.DE[i].ST = -1  # LIBERA O ESPAÇO APÓS CONTABILIZADO
                # print('REJECTED vvvvvvvvvvvvvvvvvvvvvvvvvvvv')

            if self.DE[i].ST == -2:
                totPenalty += self.DE[i].AM * self.DE[i].CO         # PENALIDADE PELO DESCARTE
                acoes.append(self.DE[i].action)
                atrasos_reais.append(abs(self.DE[i].real_LT - self.DE[i].LT))
                # penalidade por estar no ambiente
                # if totReward > 0:
                #     totPenalty += abs(self.DE[i].action - abs(self.DE[i].real_LT - self.DE[i].LT))
                self.DE[i].ST = -1  # LIBERA O ESPAÇO APÓS CONTABILIZADO
                # print('PREJUIZO $$$$$$$$$$$$$$$$$$$$$$$$$')

            if self.DE[i].ST == 4:
                totPenalty += self.YA.cont/self.YA.space * self.DE[i].AM * self.DE[i].CO
                acoes.append(self.DE[i].action)
                atrasos_reais.append(abs(self.DE[i].real_LT - self.DE[i].LT))
                # penalidade por estar no ambiente
                # if totReward > 0:
                #     totPenalty += abs(self.DE[i].action - abs(self.DE[i].real_LT - self.DE[i].LT))
                self.DE[i].ST = -1  # LIBERA O ESPAÇO APÓS CONTABILIZADO
                # totPenalty += totReward / (
                #     Yard.space - Yard.cont + 1
                # )  ### penalidade inversamente proporcional ao espaço remanescente
                # print('STORED <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')

            if self.DE[i].ST == 5:
## ACMO AJUSTAR O REWARD DE ACORDO COM A PRIORIDADE
#                totReward += self.DE[i].AM * self.DE[i].PR
#                totReward += self.DE[i].AM * self.DE[i].SU
#                totReward += self.DE[i].AM * self.DE[i].VA

                
                rw_lu += self.DE[i].AM * self.DE[i].PR
                rw_va += self.DE[i].AM * self.DE[i].PR * self.DE[i].VA
                rw_su += self.DE[i].AM * self.DE[i].PR * self.DE[i].SU

                variabilidade += self.DE[i].VA
                sustentabilidade += self.DE[i].SU
                # mask_FT eh um vetor de zeros e uns indicando quais features estao ativas (maquinas usadas)
                # Exemplo: self.mask_FT = array([1, 1, 0, 1, 1])
                mask_FT = self.DE[i].FT.copy()
                mask_FT[mask_FT > 0] = 1
                # contar quantas Features estao sendo usadas (total de maquinas usadas)
                F = mask_FT.sum()

                acoes.append(self.DE[i].action)
                atrasos_reais.append(abs(self.DE[i].real_LT - self.DE[i].LT))

                if self.objetivo == 0: # lucro
                    totReward = rw_lu
                if self.objetivo == 1: # variabilidade
                    totReward = rw_va
                if self.objetivo == 2: # sustentabilidade
                    totReward = rw_su

                # penalidade por estar no ambiente
                # if totReward > 0:
                #     totPenalty += abs(self.DE[i].action - abs(self.DE[i].real_LT - self.DE[i].LT))
                # print('REWARD ******************************')
                # totReward += self.DE[i].AM * self.DE[i].PR
                self.DE[i].ST = -1  # LIBERA O ESPAÇO APÓS CONTABILIZADO
        
        totReward -= totPenalty #RECOMPENSA COM A PENALIDADE INSERIDA NELA

        return totReward, totPenalty, rw_lu, rw_va, rw_su, variabilidade, sustentabilidade, F, acoes, atrasos_reais


    ######################
    #       step         #
    ######################

    def step(self, action):
        """
        Atualiza tudo aqui e devolve o próximo estado: n_state, reward, done, info

            - n_state: próximo estado;
            - reward: recompensa da ação;
            - done: flag de conclusão;
            - info: informaões extras (opcional)

        Primeira versão vai fazer uma iteração para cada episódio ...
        O Tempo t precisa ser controlado
        """

        # receive RAW MATERIAL AND ORDERS (DEMANDS)
        self.MT = np.array([random.randint(0, i) if i > 0 else 0 for i in self.IN])
        self.readDemand()

        # IF PREVIOUS ORDERS INVENTORY AVAILABLE, PLEASE DISPATCH
        self.match_demand_with_inventory()
        #    self.product_destination(Somn.time)

        # ANYWAY, UPDATE BALANCE AND INCOME RAW MATERIAL REGARDING MT RECEIVED
        self.IN -= self.MT
        self.BA += self.MT

        # IF RAW MATERIAL INVENTORY DOES NOT COVER PLEASE REQUEST RAW MATERIAL
        if not self.stock_covers_demand():
            self.IN = np.array(
                [random.randint(0, i) if i > 0 else 0 for i in self.IN]
            ).astype(np.int64)

        # ANYWAY START PRODUCING AND DISPATCHING
        self.product_scheduling(Somn.time, action)
        self.product_destination(Somn.time)
        Somn.time += 1

        # ORDINARY PROCEDURES IN STEP METHOD INCLUDING REWARD BY INSPECTING FINAL STATES
        # 1 STATE
        DE_arrayState = []
        for i in range(self.N):
            aux_row = [
                self.normaliza(x=self.DE[i].DI, min=self.lb_DI, max=self.ub_DI),
                self.normaliza(x=self.DE[i].DO, min=self.lb_DO, max=self.ub_DO),
                self.normaliza(x=self.DE[i].TP, min=self.lb_TP, max=self.ub_TP),
                self.normaliza(x=self.DE[i].PR, min=self.lb_PR, max=self.ub_PR),
                self.normaliza(x=self.DE[i].CO, min=self.lb_CO, max=self.ub_CO),
                self.normaliza(x=self.DE[i].AM, min=self.lb_AM, max=self.ub_AM),
                self.normaliza(x=self.DE[i].SP, min=self.lb_SP, max=self.ub_SP),
                self.normaliza(x=self.DE[i].PE, min=self.lb_PE, max=self.ub_PE),
                self.normaliza(x=self.DE[i].LT, min=self.lb_LT, max=self.ub_LT),
                self.normaliza(x=self.DE[i].VA, min=self.lb_VA, max=self.ub_VA),
                self.normaliza(x=self.DE[i].SU, min=self.lb_SU, max=self.ub_SU),
                self.normaliza(x=self.DE[i].ST, min=self.lb_ST, max=self.ub_ST),
                
            ]
            DE_arrayState.append(aux_row)
        self.DE_state = np.array(DE_arrayState)

        FT_arrayState = []
        for i in range(self.N):
            aux_FT = self.normaliza(x=self.DE[i].FT, min=self.lb_FT, max=self.ub_FT)
            FT_arrayState.append(aux_FT)
        self.FT_state = np.array(FT_arrayState)

        # 2 REWARD
        (
            reward,             # recompensa calculada com a penalidade aplicada
            penalty,            # penalidade que foi aplicada
            rw_lu,              # recompensa para lucro
            rw_va,              # recompensa para a variabilidade
            rw_su,              # recompensa para a sustentabilidade
            variabilidade,      # variabilidade de 0 a 1
            sustentabilidade,   # sustentabilidade de 0 a 1
            F,                  # numero de features utilizadas (numero de maquinas)
            acoes,              # acoes no estado ready que geraram os estados finais contabilizados
            atrasos_reais       # atrasos reais para compararar com as acoes
        ) = self.eval_final_states()  # aqui vai a função que calcula a recompensa

        #GRÁFICO PENALIDADE
        wandb.log({
            'Penalidade' : penalty,
        })
        # Gera grafico do Yard (by_frederic)

        #INFORMAÇÃO APENAS DE COMO ACABA O EPISÓDIO, BUSCAR LOCAL PARA RECEBER MELHOR INFORMAÇÃO
        if self.Y > 0:
            wandb.log({
                'Yard': (self.YA.cont/self.YA.Y)*100,           
            })

        else: #APENAS MOSTRANDO O YARD COMPLETAMENTE CHEIO CASO ELE SEJA 0
            wandb.log({
                'Yard' : 100,
            })

        # 3 FINAL CONDITION
        done = False
        truncated = False
        # if penalty>0:
        # reward =0
        # print('\n D -- O -- N -- E --', self.DE_state)
        # done = True

        if Somn.time >= self.ub_time:  # 10*Demand.MAXDO + Demand.M   (TEMPOMAX)
            # print('\n D -- O -- N -- E --', self.DE_state)
            done = True

        # Atualiza o upper bounds
        if np.amax(self.ub_MT) <= np.amax(self.MT):
            self.ub_MT = np.full(self.M, np.amax(self.MT)) 

        if np.amax(self.ub_BA) <= np.amax(self.BA):
            self.ub_BA = np.full(self.M, np.amax(self.BA))

        if np.amax(self.ub_IN) <= np.amax(self.IN):
            self.ub_IN = np.full(self.M, np.amax(self.IN))
        
        if np.amax(self.ub_OU) <= np.amax(self.OU):
            self.ub_OU = np.full(self.M, np.amax(self.OU))
       

        info = {"rw": reward,
                "rw_lu": rw_lu,
                "rw_va": rw_va,
                "rw_su": rw_su,
                "VA": variabilidade,
                "SU": sustentabilidade,
                "F": F,
                "acoes": acoes,
                "atrasos_reais": atrasos_reais}  # Informações adicionais
        # observation = self.DE_state  #by_frederic: retorna quando e um tipo Box
        observation = {
            "time": np.array([self.normaliza(self.time, self.lb_time, self.ub_time)]),
            "MT": self.normaliza(self.MT, self.lb_MT, self.ub_MT),
            "EU": self.normaliza(self.EU, self.lb_EU, self.ub_EU),
            "BA": self.normaliza(self.BA, self.lb_BA, self.ub_BA),
            "IN": self.normaliza(self.IN, self.lb_IN, self.ub_IN),
            "OU": self.normaliza(self.OU, self.lb_OU, self.ub_OU),
            "DE_state": self.DE_state,
            "FT_state": self.FT_state,
            "yard": np.array([self.normaliza(self.YA.cont, self.lb_yard, self.ub_yard)]),
            "load": np.array([self.normaliza(Demand.load, self.lb_load, self.ub_load)]),

        }  # by_frederic: retorna quando e um tipo Dict

        return (
            observation,
            reward,
            done,
            truncated,
            info,
        )  # , exprofit   # by_frederic:

    ######################
    #       reset        #
    ######################

    def reset(self, *, seed=None, options=None):
        #super().reset(seed=None)
        
        Somn.priorq = [heapdict() for objetivo in Somn.obj_list]
        # Somn.priorqsu = heapdict()
        # Somn.priorqva = heapdict()
        self.MT = np.random.randint(0, self.MAXFT, self.M)
        self.EU = np.random.random(self.M) * self.MAXEU
        self.BA = np.random.randint(10, 10*self.MAXFT, self.M)
        self.IN = np.random.randint(0, self.MAXFT, self.M)
        self.OU = np.random.randint(0, self.MAXFT, self.M)
        
        #LOGS PONTUAIS
        wandb.log({
            'reject_w_waste Somn' : Demand.reject_w_waste
        })


        Somn.time = 1
        Demand.load = 0
        Demand.reject = 0
        Demand.reject_w_waste=0

        self.YA = Yard(self.Y)

        DE_arrayState = []
        for i in range(self.N):
            self.DE[i](Somn.time)
            aux_row = [
                self.normaliza(x=self.DE[i].DI, min=self.lb_DI, max=self.ub_DI),
                self.normaliza(x=self.DE[i].DO, min=self.lb_DO, max=self.ub_DO),
                self.normaliza(x=self.DE[i].TP, min=self.lb_TP, max=self.ub_TP),
                self.normaliza(x=self.DE[i].PR, min=self.lb_PR, max=self.ub_PR),
                self.normaliza(x=self.DE[i].CO, min=self.lb_CO, max=self.ub_CO),
                self.normaliza(x=self.DE[i].AM, min=self.lb_AM, max=self.ub_AM),
                self.normaliza(x=self.DE[i].SP, min=self.lb_SP, max=self.ub_SP),
                self.normaliza(x=self.DE[i].PE, min=self.lb_PE, max=self.ub_PE),
                self.normaliza(x=self.DE[i].LT, min=self.lb_LT, max=self.ub_LT),
                self.normaliza(x=self.DE[i].VA, min=self.lb_VA, max=self.ub_VA),
                self.normaliza(x=self.DE[i].SU, min=self.lb_SU, max=self.ub_SU),
                self.normaliza(x=self.DE[i].ST, min=self.lb_ST, max=self.ub_ST),
                
            ]
            DE_arrayState.append(aux_row)
        self.DE_state = np.array(DE_arrayState)

        FT_arrayState = []
        for i in range(self.N):
            aux_FT = self.normaliza(x=self.DE[i].FT, min=self.lb_FT, max=self.ub_FT)
            FT_arrayState.append(aux_FT)
        self.FT_state = np.array(FT_arrayState)

        info = dict()
        # observation = (self.DE_state, info)  # by_frederic: retorna quando o tipo é Box
        observation = {
            "time": np.array([self.normaliza(self.time, self.lb_time, self.ub_time)]),
            "MT": self.normaliza(self.MT, self.lb_MT, self.ub_MT),
            "EU": self.normaliza(self.EU, self.lb_EU, self.ub_EU),
            "BA": self.normaliza(self.BA, self.lb_BA, self.ub_BA),
            "IN": self.normaliza(self.IN, self.lb_IN, self.ub_IN),
            "OU": self.normaliza(self.OU, self.lb_OU, self.ub_OU),
            "DE_state": self.DE_state,
            "FT_state": self.FT_state,
            "yard": np.array([self.normaliza(self.YA.cont, self.lb_yard, self.ub_yard)]),
            "load": np.array([self.normaliza(Demand.load, self.lb_load, self.ub_load)]),
        }  # by_frederic: retorna quando e um tipo Dict

        return (observation, info)  # by_frederic: para se adequar ao Gymnasium

    ######################
    #       render       #
    ######################

    def render(self):
        # print("Current state (RENDER): \n", self.DE_state)
        pass

    ######################
    #       close        #
    ######################

    def close(self):
        pass
