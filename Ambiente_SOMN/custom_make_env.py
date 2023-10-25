from stable_baselines3.common.monitor import Monitor
from Ambiente_SOMN.Somn import Somn

def custom_make_env(atraso: int = None):
    env = Somn(Y=3,M=10,N=10,MAXDO=10,MAXAM=3,MAXPR=2,MAXPE=10,MAXFT=5,MAXMT=3,MAXTI=2,
               MAXEU = 5, atraso=atraso)
    env = Monitor(env)  # record stats such as returns
    return env