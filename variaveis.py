Y = 3 
M = 3
N = 6
MAXDO = 10
MAXAM = 2
MAXPR = 2
MAXPE = 10
MAXFT = 5
MAXMT = 3
MAXTI = 2
MAXEU = 5
atraso = -1


self.EU = array([4.50005489, 2.09216611, 3.64392987])

self.MT = array([1, 1, 2])

self.BA = array([3, 0, 3])
self.IN = array([1, 1, 4])
self.OU = array([2, 0, 4])


# Materia-prima
self.lb_MT = array([0, 0, 0], dtype=int64)
self.ub_MT = array([5, 5, 5], dtype=int64)

# preço (valor monetário)
self.lb_EU = array([0., 0., 0.])
self.ub_EU = array([5., 5., 5.])

self.lb_BA = array([0, 0, 0], dtype=int64)
self.ub_BA = array([5, 5, 5], dtype=int64)

self.lb_IN = array([0, 0, 0], dtype=int64)
self.ub_IN = array([5, 5, 5], dtype=int64)

self.lb_OU = array([0, 0, 0], dtype=int64)
self.ub_OU = array([5, 5, 5], dtype=int64)



# variáveis de tempo
self.lb_time = 1
self.ub_time = 103

self.lb_LT = 2
self.ub_LT = 9

self.lb_TP = 2
self.ub_TP = 136

self.lb_DI = 1
self.ub_DI = 103

self.lb_DO = 3
self.ub_DO = 122

self.lb_ST = -2


# Variaveis Globais
somn.objetivo = 1
somn.time = 1

Demand.load = 0
Demand.reject = 0
Demand.reject_w_waste=0

Yard.Y = 3
Yard.cont = 0
Yard.space = 3
Yard.yard = [0, 0, 0]

# Variávies da Demanda
self.CU = 1     # customer
self.DI = 1
self.DO = 9
self.TP = 8
self.EU = array([2.94841773, 3.40236884, 1.78502145])
self.PR = 14.280171589043714
self.CO = 7.140085794521857
self.AM = 1     # amount
self.SP = 0.16666666666666666
self.PE = 4     # Penalty
self.LT = 2
self.VA = 0.3333333333333333
self.SU = 0.6666666666666667
self.ST = 0     # Status
self.FT = array([0, 0, 4])  # Features


