import json

# Config.json example
# {
#     "NeuralNetwork" : {
#         "NNType" : "Residual",
#         "InputChannels" : 18,
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
    "NeuralNetwork",
    "NNType",
    "InputChannels",
    "InputFeatures",
    "PolicyExtract",
    "ValueExtract",
    "Stack",
    "ResidualChannels",
    "ResidualBlock",
    "ResidualBlock-SE",
]

class NetworkConfig:
    def __init__(self):
        self.stack = []
        self.nntype = None
        self.input_channels = None
        self.input_features = None
        self.residual_channels = None
        self.policy_extract = None
        self.value_extract = None
        self.xsize = 10
        self.ysize = 9
        self.policy_map = 2 * (9 + 8) + 8 + 4 + 4 # 50
        self.winrate_size = 4
        self.valuelayers = 256

def json_loader(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data

def NN_parser(json_data):
    nn_config = NetworkConfig()
    resnet = json_data["NeuralNetwork"]

    nn_config.nntype = resnet["NNType"]
    nn_config.input_channels = resnet["InputChannels"]
    nn_config.residual_channels = resnet["ResidualChannels"]
    nn_config.policy_extract = resnet["PolicyExtract"]
    nn_config.value_extract = resnet["ValueExtract"]
    nn_config.input_features = resnet["InputFeatures"]

    stack = resnet["Stack"]
    for s in stack:
        nn_config.stack.append(s)
    return nn_config

def gather_networkconfig(filename):
    d = json_loader(filename)
    n = NN_parser(d)
    return n
