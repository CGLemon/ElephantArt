import glob
import struct

FIXED_DATA_VERSION = 1

'''
------- claiming -------
 L1       : Version
 
 ------- Inputs data -------
 L2  - L8 : Current player pieces Index
 L9  - L15: Other player pieces Index
 L16 - L17: last move
 L18      : Current Player
 L19      : Game plies 
 L20      : Repeat conut
 
 ------- Prediction data -------
 L21      : Probabilities
 L22      : Which piece go to move
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
        self.repeat = 0
        self.last_from = 0
        self.last_to = 0

        self.policyindex = []
        self.probabilities = []
        self.move = None
        self.result = None

    def dump(self):
        print("Current pieces:")
        print(" ".join(str(out) for out in self.current_pieces))
        print("Other pieces:")
        print(" ".join(str(out) for out in self.other_pieces))
        print("Tomove: {}".format(self.tomove))
        print("Plies: {}".format(self.plies))
        print("Repeat: {}".format(self.repeat))
        print("Last Move: from {}, to {}".format(self.last_from, self.last_to))
        print("Probabilities:")
        for idx, p in zip(self.policyindex, self.probabilities):
            print("({}: {}), ".format(idx, p), end="")
        print()
        print("Moved Piece: {}".format(self.move))
        print("Result: {}".format(self.result))

    # Only support version one.
    def fill_v1(self, linecnt, readline):
        if linecnt == 0:
            v = int(readline)
            assert v == 1, "The data is not correct version."
        elif linecnt >= 1 and linecnt <= 7:
            p = readline.split()
            start = self.ACCUMULATE[linecnt-1]
            for i in range(len(p)):
                self.current_pieces[start+i] = int(p[i])
        elif linecnt >= 8 and linecnt <= 14:
            p = readline.split()
            start = self.ACCUMULATE[linecnt-8]
            for i in range(len(p)):
                self.other_pieces[start+i] = int(p[i])
        elif linecnt == 15:
            self.last_from = int(readline)
        elif linecnt == 16:
            self.last_to = int(readline)
        elif linecnt == 17:
            self.tomove = int(readline)
        elif linecnt == 18:
            self.plies = int(readline)
        elif linecnt == 19:
            self.repeat = int(readline)
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

    @staticmethod
    def get_datalines(version):
        if version == 1:
            return 23
        return 0

class ChunkParser:
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

    def pack_v1(self, data):
        int_symbol = "i"

        # current player pieces
        inputs_fmt = str(data.TOTAL_NUMBER) + int_symbol

        # other player pieces
        inputs_fmt += str(data.TOTAL_NUMBER) + int_symbol

        # inputs misc(last from move, last to move, current player, plies, repeat)
        inputs_fmt += str(5) + int_symbol
        print(inputs_fmt)
        inputs_misc = [data.last_from, data.last_to, data.tomove, data.plies, data.repeat]
        inputs = struct.pack(inputs_fmt, data.current_pieces, data.other_pieces, inputs_misc)

        # probabilities
        size = len(data.probabilities)
        pred_fmt = str(size) + int_symbol + str(size) + "f"

        # piece to go
        pred_fmt += "1c"

        # result
        pred_fmt += str(1) + int_symbol

        pred = struct.pack(pred_fmt, data.probabilities, data.policyindex, data.move, data.result)

        fmt = [inputs_fmt, pred_fmt]
        mem = [struct.calcsize(inputs_fmt), struct.calcsize(pred_fmt)]

        return inputs, pred, fmt, mem

    def unpack_v1(self, inputs, pred, fmt):
        data = Data()

        data.current_pieces, data.other_pieces, inputs_misc = struct.unpack(fmt[0], inputs)
        data.last_from = inputs_misc[0]
        data.last_to = inputs_misc[1]
        data.tomove = inputs_misc[2]
        data.plies = inputs_misc[3]
        data.repeat = inputs_misc[4]

        data.probabilities, data.policyindex, data.move, data.result = struct.unpack(fmt[1], pred)

        return daat

    def run(self):
        datalines = Data.get_datalines(FIXED_DATA_VERSION);
        for name in glob.glob(self.dirname + "/*"):
            if self.cfg.debugVerbose:
                print(name)

            with open(name, 'r') as f:
                while True:
                    if self.linesparser(datalines, f) == False:
                        break
            
    def dump(self):
        for b in self.buffer:
            b.dump()
            print()
        print("----------------------------------------------------------")
        for b in self.ttbuffer:
            b.dump()
            print()

    def __getitem__(self, idx):
        return self.buffer[idx]

    def __len__(self):
        return len(self.buffer) 
