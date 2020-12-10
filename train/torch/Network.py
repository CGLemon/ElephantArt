import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import config
from config import NetworkConfig


def dump_dependent_version():
    print("Name : {name} ->  Version : {ver}".format(name =  "Numpy", ver = np.__version__))
    print("Name : {name} ->  Version : {ver}".format(name =  "Torch", ver = torch.__version__))

class FullyConnect(nn.Module):
    def __init__(self, in_size,
                       out_size,
                       relu=True,
                       collector=None):
        super().__init__()
        self.relu = relu
        self.linear = nn.Linear(in_size, out_size)

        if collector != None:
            collector.append(self.linear.weight)
            collector.append(self.linear.bias)

    def forward(self, x):
        x = self.linear(x)
        return F.relu(x, inplace=True) if self.relu else x 


class Convolve(nn.Module):
    def __init__(self, in_channels,
                       out_channels,
                       kernel_size,
                       relu=True,
                       collector=None):
        super().__init__()
        self.relu = relu
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            padding=1 if kernel_size == 3 else 0,
            bias=False,
        )
        if collector != None:
            collector.append(self.conv.weight)
            collector.append(torch.zeros(out_channels))

        nn.init.kaiming_normal_(self.conv.weight,
                                mode="fan_out",
                                nonlinearity="relu")
    def forward(self, x):
        x = self.conv(x)
        return F.relu(x, inplace=True) if self.relu else x


class ConvBlock(nn.Module):
    def __init__(self, in_channels,
                       out_channels,
                       kernel_size,
                       relu=True,
                       collector=None):
        super().__init__()
 
        assert kernel_size in (1, 3)
        self.relu = relu
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            padding=1 if kernel_size == 3 else 0,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(
            out_channels,
            eps=1e-5,
            affine=False,
        )

        if collector != None:
            collector.append(self.conv.weight)
            collector.append(torch.zeros(out_channels))
            collector.append(self.bn.running_mean)
            collector.append(self.bn.running_var)

        nn.init.kaiming_normal_(self.conv.weight,
                                mode="fan_out",
                                nonlinearity="relu")
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True) if self.relu else x

class ResBlock(nn.Module):
    def __init__(self, channels, collector=None):
        super().__init__()

        self.conv1 = ConvBlock(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            collector=collector
        )
        self.conv2 = ConvBlock(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            relu=False,
            collector=collector
        )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)

        out += identity
        return F.relu(out, inplace=True)


class Network(nn.Module):
    def __init__(self , config : NetworkConfig):
        super().__init__()
        self.tensor_collector = []
        self.nntype = config.nntype
        self.input_channels = config.input_channels
        self.residual_channels = config.residual_channels
        self.xsize = config.xsize
        self.ysize = config.ysize
        self.plane_size = self.xsize * self.ysize
        self.value_extract = config.value_extract
        self.policy_extract = config.policy_extract
        self.policy_map = config.policy_map
        self.stack = config.stack

        # build network
        self.input_conv = ConvBlock(
            in_channels=self.input_channels,
            out_channels=self.residual_channels,
            kernel_size=3,
            collector=self.tensor_collector
        )

        # residual tower
        nn_stack = []
        for s in self.stack:
            if (s == "ResidualBlock"):
                nn_stack.append(ResBlock(self.residual_channels, self.tensor_collector))
        self.residual_tower = nn.Sequential(*nn_stack)

        # policy head
        self.policy_conv = ConvBlock(
            in_channels=self.residual_channels,
            out_channels=self.policy_extract, 
            kernel_size=3,
            collector=self.tensor_collector
        )
        self.map_conv = Convolve(
            in_channels=self.policy_extract,
            out_channels=self.policy_map,
            kernel_size=3,
            relu=False,
            collector=self.tensor_collector
        )

        # value head
        self.value_conv = ConvBlock(
            in_channels=self.residual_channels,
            out_channels=self.value_extract,
            kernel_size=1,
            collector=self.tensor_collector
        )
        self.value_fc_1 = FullyConnect(
            in_size=self.value_extract * self.plane_size,
            out_size=256,
            collector=self.tensor_collector
        )
        self.value_fc_2 = FullyConnect(
            in_size=256,
            out_size=3,
            relu=False,
            collector=self.tensor_collector
        )

        self.dump_info()
        self.trainable()

    def forward(self, planes):
        x = self.input_conv(planes)

        # residual tower
        x = self.residual_tower(x)

        # policy head
        pol = self.policy_conv(x)
        pol = self.map_conv(pol)
        pol = torch.flatten(pol, start_dim=1)

        # value head
        val = self.value_conv(x)
        val = torch.flatten(val, start_dim=1)
        val = self.value_fc_1(val)
        val = self.value_fc_2(val)

        return pol, val

    def trainable(self, t=True):
        if t==True:
            self.train()
        else:
            self.eval()

    def dump_info(self):
        print("Plane size [x,y] : [{xsize}, {ysize}] ".format(xsize=self.xsize, ysize=self.ysize))
        print("Input channels : {channels}".format(channels=self.input_channels))
        print("Residual channels : {channels}".format(channels=self.residual_channels))
        print("Residual tower : size -> {s} [".format(s=len(self.stack)))
        for s in self.stack:
            print("  {}".format(s))
        print("]")
        print("Policy Extract : {policyextract}".format(policyextract=self.policy_extract))
        print("Policy Map : {policymap}".format(policymap=self.policy_map))
        print("Value Extract : {valueextract}".format(valueextract=self.value_extract))

    def transfer2text(self, filename):
        with open(filename, 'w') as f:
            f.write("fork main\n")
            f.write("fork info\n")
            f.write("NNType {}\n".format(self.nntype))
            f.write("InputChannels {}\n".format(self.input_channels))
            f.write("ResidualChannels {}\n".format(self.residual_channels))
            f.write("ResidualBlocks {}\n".format(len(self.stack)))
            f.write("PolicyExtract {}\n".format(self.policy_extract))
            f.write("PolicyMap {}\n".format(self.policy_map))
            f.write("ValueExtract {}\n".format(self.value_extract))
            f.write("end info\n")

            f.write("fork model\n")
            f.write(Network.conv2text(self.input_channels, self.residual_channels, 3))
            f.write(Network.bn2text(self.residual_channels))
            for s in self.stack:
                if s == "ResidualBlock":
                    f.write(Network.conv2text(self.residual_channels, self.residual_channels, 3))
                    f.write(Network.bn2text(self.residual_channels))
                    f.write(Network.conv2text(self.residual_channels, self.residual_channels, 3))
                    f.write(Network.bn2text(self.residual_channels))
                    
            f.write(Network.conv2text(self.residual_channels, self.policy_extract, 3))
            f.write(Network.bn2text(self.policy_extract))
            f.write(Network.conv2text(self.policy_extract, self.policy_map, 3))
            
            f.write(Network.conv2text(self.residual_channels, self.value_extract, 1))
            f.write(Network.bn2text(self.value_extract))
            
            f.write(Network.fullyconnect2text(self.value_extract * self.plane_size, 256))
            f.write(Network.fullyconnect2text(256, 3))
            
            f.write("end model\n")

            f.write("fork parameters\n")
            for tensor in self.tensor_collector:
                f.write(Network.tensor2text(tensor))
            f.write("end parameters\n")
            f.write("end main")

    @staticmethod
    def fullyconnect2text(in_size, out_size):
        return "FullyConnect {iS} {oS}\n".format(iS=in_size, oS=out_size)

    @staticmethod
    def conv2text(in_channels, out_channels, kernel_size):
        return "Convolve {iC} {oC} {KS}\n".format(
                   iC=in_channels,
                   oC=out_channels,
                   KS=kernel_size)

    @staticmethod
    def bn2text(channels):
        return "BatchNorm {C}\n".format(C=channels)
    
    @staticmethod
    def tensor2text(t: torch.Tensor):
        return " ".join([str(w) for w in t.detach().numpy().ravel()]) + "\n"
