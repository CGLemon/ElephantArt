import numpy as np

class Symmetry:
    def __init__(self, xsize, ysize):
        self.xsize = xsize
        self.ysize = ysize
        self.symmetry_index = np.zeros((4, self.xsize * self.ysize), dtype=np.int)
        for y in range(self.ysize):
            for x in range(self.xsize):
                idx = self.get_index(x, y)
                self.symmetry_index[0][idx] = self.get_symmetry_index(x, y, 0)
                self.symmetry_index[1][idx] = self.get_symmetry_index(x, y, 1)
                self.symmetry_index[2][idx] = self.get_symmetry_index(x, y, 2)
                self.symmetry_index[3][idx] = self.get_symmetry_index(x, y, 3)

    def dump(self):
        for s in range(4):
            print("Symmetry: {}".format(s+1))
            for y in range(self.ysize):
                for x in range(self.xsize):
                    idx = self.get_index(x, y)
                    print("{:>2d}".format(self.symmetry_index[s][idx]), end=" ")
                print()
            print()
        print()


    def get_transfer(self, idx, symm):
        return self.symmetry_index[symm]

    def get_index(self, x, y):
        return x + self.xsize * y

    def get_symmetry_index(self, x, y, symm):
        x, y = self.get_symmetry(x, y, symm)
        return x + self.xsize * y

    def get_symmetry(self, x, y, symm):
        idx_x = x
        idx_y = y
        if (symm // 2 == 1):
            idx_x = self.xsize - idx_x - 1;
        if (symm % 2 == 1):
            idx_y = self.ysize - idx_y - 1;
        return idx_x, idx_y
