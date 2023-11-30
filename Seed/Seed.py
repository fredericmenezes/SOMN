import random
import numpy as np
import torch as th
import os

#SEED = 1

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.backends.cudnn.deterministic = True
    #env.seed(seed)

##One call at beginning is enough
#seed_everything(SEED)