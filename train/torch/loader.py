import glob
import struct
import random

import numpy as np

# Now the version is zero. This is experiment version. We don't
# promise that the data format will same in the future. 
FIXED_DATA_VERSION = 0

'''
------- claiming -------
 L1       : Version
 
 ------- Inputs data -------
 L2  - L8 : Current player pieces Index
 L9  - L15: Other player pieces Index
 L16      : Current Player
 L17      : Game plies
 L18      : Fifty-Rule ply left
 L19      : Repetitions
 
 ------- Prediction data -------
 L20      : Probabilities
 L21      : Which piece go to move
 L22      : Moves left
 L23      : Result

'''

class PiecesIndex:
    def __init__(self):
        # According to ElephantArt Engine, the pieces sequence follow
        # below sequence. We assume that evey sequence about pieces will
        # follow it.
        self.PAWN_START_INDEX = 0
        self.CANNON_START_INDEX = 1
        self.ROOK_START_INDEX = 2
        self.HORSE_START_INDEX = 3
        self.ELEPHANT_START_INDEX = 4
        self.ADVISOR_START_INDEX = 5
        self.KING_START_INDEX = 6

        # We assume that it is not a variant rule. A player has
        # 5 pawns, 2 cannons, 2 rooks, 2 horses, 2  elephants, 2 
        # advisors, 1 king. On the other hand, each player have 16
        # pieces. 
        self.PIECES_NUMBER = [5, 2, 2, 2, 2, 2, 1]
        self.ACCUMULATE = [0, 5, 7, 9, 11, 13, 15]
        self.TOTAL_NUMBER = 16

class Data(PiecesIndex):
    def __init__(self):
        super().__init__()
        self.version = FIXED_DATA_VERSION

        self.current_pieces = [-1] * self.TOTAL_NUMBER
        self.other_pieces = [-1] * self.TOTAL_NUMBER

        self.tomove = None
        self.plies = 0
        self.rule50_remaining = 0
        self.repetitions = 0

        self.policyindex = []
        self.probabilities = []
        self.move = None
        self.moves_left = 0
        self.result = None

    def dump(self):
        print("Current pieces:")
        print(" ".join(str(out) for out in self.current_pieces))
        print("Other pieces:")
        print(" ".join(str(out) for out in self.other_pieces))
        print("Tomove: {}".format(self.tomove))
        print("Plies: {}".format(self.plies))
        print("Fifty-Rule ply left: {}".format(self.rule50_remaining))
        print("Repetitions: {}".format(self.repetitions))
        print("Probabilities:")
        for idx, p in zip(self.policyindex, self.probabilities):
            print("({}: {}), ".format(idx, p), end="")
        print()
        print("Moved Piece: {}".format(self.move))
        print("Moves left: {}".format(self.moves_left))
        print("Result: {}".format(self.result))

    # Only support version one.
    def fill_v1(self, linecnt, readline):
        if linecnt == 0:
            v = int(readline)
            assert v == 1 or v == 0, "The data is not correct version."
        elif linecnt >= 1 and linecnt <= 7:
            p = readline.split()
            start = self.ACCUMULATE[linecnt-1]
            for i in range(len(p)):
                self.current_pieces[start+i] = int(p[i])
            self.current_pieces = np.array(self.current_pieces, dtype=np.int8)

        elif linecnt >= 8 and linecnt <= 14:
            p = readline.split()
            start = self.ACCUMULATE[linecnt-8]
            for i in range(len(p)):
                self.other_pieces[start+i] = int(p[i])
            self.other_pieces = np.array(self.other_pieces, dtype=np.int8)

        elif linecnt == 15:
            self.tomove = int(readline)
        elif linecnt == 16:
            self.plies = int(readline)
        elif linecnt == 17:
            self.rule50_remaining = int(readline)

        elif linecnt == 18:
            self.repetitions = int(readline)
        elif linecnt == 19:
            p = readline.split()
            for i in range(len(p)):
                if i % 2 == 0:
                    self.policyindex.append(int(p[i]))
                elif i % 2 == 1:
                    self.probabilities.append(float(p[i]))
            self.policyindex = np.array(self.policyindex, dtype=np.int16)
            self.probabilities = np.array(self.probabilities, dtype=np.float32)

        elif linecnt == 20:
            self.move = int(readline)
        elif linecnt == 21:
            self.moves_left = int(readline)   
        elif linecnt == 22:
            self.result = int(readline)

    @staticmethod
    def get_datalines(version):
        if version == 0:
            return 23
        return 0

class Loader:
    def __init__(self, cfg, dirname):
        self.cfg = cfg
        self.dirname = dirname
        self.buffer = []
        self.run()

    def linesparser(self, datalines, filestream):
        data = Data()
        for cnt in range(datalines):
            readline = filestream.readline()
            if len(readline) == 0:
                assert cnt == 0, "The data is incomplete."
                return False
            data.fill_v1(cnt, readline)
        if self.cfg.debugVerbose:
            print("linesparser sccueess")

        self.buffer.append(data)

        return True

    def run(self):
        if self.dirname != None:
            datalines = Data.get_datalines(FIXED_DATA_VERSION);
            for name in glob.glob(self.dirname + "/*"):
                if self.cfg.debugVerbose:
                    print(name)

                with open(name, 'r') as f:
                    while True:
                        if self.linesparser(datalines, f) == False:
                            break
            random.shuffle(self.buffer)

        if self.cfg.debugVerbose:
            self.dump()

    def dump(self):
        for data in self.buffer:
            data.dump()
            print()
        print("----------------------------------------------------------")

    def __getitem__(self, idx):
        return self.buffer[idx]

    def __len__(self):
        return len(self.buffer) 
