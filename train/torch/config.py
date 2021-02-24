import json

# Verbose option
debugVerbose = False
miscVerbose = True


# Config.json example
# {
#     "NeuralNetwork" : {
#         "NNType" : "Residual",
#         "InputChannels" : 18,
#         "InputFeatures" : 4
#         "ResidualChannels" : 64,
#         "PolicyExtract" : 256,
#         "ValueExtract" : 8,
#         "Stack" : [
#             "ResidualBlock",
#             "ResidualBlock",
#             "ResidualBlock-SE"
#         ]
#     }
# }

CONFIG_KEYWOED = [
    "NeuralNetwork",    # claiming
    "NNType",           # the type of net
    "Version",          # net version(implies the net structure)
    "InputChannels",    # Input planes channels
    "InputFeatures",    # Input features size
    "PolicyExtract",    # policy shared head channels
    "ValueExtract",     # value shared head channels
    "Stack",            # the net structure(also implies the block number)
    "ResidualChannels", # each resnet block channels
    "ResidualBlock",    # resnet block without variant
    "ResidualBlock-SE", # resnet block with SE structure
]

class NetworkConfig:
    def __init__(self):
        # Adjustable values
        self.stack = []
        self.residual_channels = None
        self.policy_extract = None
        self.value_extract = None

        # Option values(not yet)

        # Fixed values but flexible
        self.nntype = None
        self.input_channels = None
        self.input_features = None

        # Fixed values
        self.xsize = 10 # fixed
        self.ysize = 9 # fixed
        self.policy_map = 50 # fixed
            # 50 = 18 (file moves) + 16 (rank moves) +
            #      8(horse moves) + 4(advisor moves) + 4(elephant moves)

        self.valuelayers = 256 # fixed
        self.winrate_size = 4 # one stm-winrate head + three wdl-winrate head

def json_loader(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data

def nnparser(json_data):
    # We assume that every value is valid.
    nnconfig = NetworkConfig()
    resnet = json_data["NeuralNetwork"]

    nnconfig.nntype = resnet["NNType"]
    nnconfig.input_channels = resnet["InputChannels"]
    nnconfig.residual_channels = resnet["ResidualChannels"]
    nnconfig.policy_extract = resnet["PolicyExtract"]
    nnconfig.value_extract = resnet["ValueExtract"]
    nnconfig.input_features = resnet["InputFeatures"]

    stack = resnet["Stack"]
    for s in stack:
        nnconfig.stack.append(s)
    return nnconfig

def gather_networkconfig(filename):
    d = json_loader(filename)
    n = nnparser(d)
    return n
