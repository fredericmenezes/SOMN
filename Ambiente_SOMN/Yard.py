import numpy as np

class Yard:
    def __init__(self, Y, numFeat, typFeat):
        Yard.Y=Y
        Yard.cont=0
        Yard.space = Y
        self.yard = np.zeros(numFeat)
        self.mask_YA = np.zeros(numFeat)