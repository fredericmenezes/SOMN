import numpy as np

class Yard:
    def __init__(self, Y):
        self.Y = Y
        self.space = Y
        
        self.yard = []
        self.mask_YA = []

        self.cont = 0

    def inYard(self, array):
        for idx, element in enumerate(self.mask_YA):
            if np.array_equal(element, array):
                return idx
        return -1
    
    def remove_yard(self, idx):

        self.yard.pop(idx)
        self.mask_YA.pop(idx)
        self.cont = len(self.yard)
        