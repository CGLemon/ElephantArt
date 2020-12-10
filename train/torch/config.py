import json

CONFIG_KEYWOED = [
    "NeuralNetwork",
    "NNType",
    "InputChannels",
    "PolicyExtract",
    "ValueExtract",
    "Stack",
    "ResidualChannels",
    "ResidualBlock",
    "ResidualBlock-withSE",
]

class NetworkConfig:
    def __init__(self):
        self.stack = []
        self.nntype = None
        self.input_channels = None
        self.residual_channels = None
        self.policy_extract = None
        self.value_extract = None
        self.xsize = 10
        self.ysize = 9
        self.policy_map = 2 * (9 + 8) + 8 + 4

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

    stack = resnet["Stack"]
    for s in stack:
        nn_config.stack.append(s)
    return nn_config

def gather_networkconfig(filename):
    d = json_loader(filename)
    n = NN_parser(d)
    return n