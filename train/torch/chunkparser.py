import glob
import struct
import config

FIXED_DATA_VERSION = 1

'''
------- claiming -------
 L1        : Version
 
 ------- Inputs data -------
 L2  - L15 : Pieces Index
 L16       : Current Player
 L17       : Game plies 
 L18       : Repeat conut
 L19 - L20 : last move
 
 ------- Prediction data -------
 L21       : Probabilities
 L22       : Which pieces go to move
 L23       : Result

'''

class Data:
    def __init__(self):
        self.version = FIXED_DATA_VERSION

        self.pieces = []

        self.tomove = None
        self.plies = 0
        self.repeat = 0
        self.last_from = 0
        self.last_to = 0

        self.policyindex = []
        self.probabilities = []
        self.move = None
        self.result = None

    def dump(self):
        for p in self.pieces:
            print(" ".join(str(out) for out in p))
        print("Tomove : {}".format(self.tomove))
        print("Plies : {}".format(self.plies))
        print("Repeat : {}".format(self.repeat))
        print("Last Move : from {} | to {}".format(self.last_from, self.last_to))
        for idx, p in zip(self.policyindex, self.probabilities):
            print("({} : {}) ".format(idx, p), end="")
        print()
        print("Moved Piece : {}".format(self.move))
        print("Result : {}".format(self.result))

    def fill(self, linecnt, readline):
        if linecnt == 0:
            v = int(readline)
            assert v == FIXED_DATA_VERSION, "The data is not correct version."
        elif linecnt == 15:
            self.tomove = int(readline)
        elif linecnt == 16:
            self.plies = int(readline)
        elif linecnt == 17:
            self.repeat = int(readline)
        elif linecnt == 18:
            self.last_from = int(readline)
        elif linecnt == 19:
            self.last_to = int(readline)
        elif linecnt == 20:
            p = readline.split()
            for i in range(len(p)):
                if i % 2 == 0:
                    self.policyindex.append(int(p[i]))
                elif i % 2 == 1:
                    self.probabilities.append(float(p[i]))
        elif linecnt == 21:
            self.move = readline.rstrip("\n")
        elif linecnt == 22:
            self.result = int(readline)
        else:
            index = linecnt - 1
            p = readline.split()
            temp = []
            for idx in p:
                temp.append(int(idx))
            self.pieces.append(temp)


    @staticmethod
    def get_datalines(version):
        if version == 1:
            return 23
        return 0


class ChunkParser:
    def __init__(self, dirname):
        self.dirname = dirname
        self.buffer = []

    def linesparser(self, datalines, filestream):
        data = Data()
        for cnt in range(datalines):
            readline = filestream.readline()
            if len(readline) == 0:
                assert cnt == 0, "The data is incomplete."
                return False
            data.fill(cnt, readline)
        if config.debugVerbose:
            print("linesparser sccueess")
        self.buffer.append(data)
        return True

    def run(self):
        datalines = Data.get_datalines(FIXED_DATA_VERSION);

        for name in glob.glob(self.dirname + "/*"):
            if config.debugVerbose:
                print(name)

            with open(name, 'r') as f:
                while True:
                    if self.linesparser(datalines, f) == False:
                        break
            
    def dump(self):
        for b in self.buffer:
            b.dump()
            print()
cp = ChunkParser("test")
cp.run()
cp.dump()
