import numpy as np

class Yard:
    def __init__(self, Y):
        self.Y = Y
        self.space = Y
        
        self.yard = []
        self.mask_YA = []

        self.cont = 0

    def inYard(self, array):
        for element in self.mask_YA:
            if np.array_equal(element, array):
                return True
        return False
    
    def remove_yard(self, array):
        mask = array.copy()
        mask[mask > 0] = 1

        self.yard.remove(array)
        self.mask_YA.remove(mask)
        self.cont = len(self.yard)