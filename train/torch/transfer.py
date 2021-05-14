import argparse

from train import Network
from config import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-j", "--json", help="The json file name", type=str)
    parser.add_argument("-n", "--network", help="", type=str)
    args = parser.parse_args()

    cfg = gather_config(args.json)
    Net = Network(cfg)
    Net.load_pt(args.network + ".pt")
    Net.transfer2text(args.network + ".txt")
