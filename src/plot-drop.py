from src.plot import plot_drop
import sys
import argparse
from src.utils import Bunch
import yaml

parser = argparse.ArgumentParser()
parser.add_argument('--params', default='plot-params/weights-drop.yaml', type=str,
                    help='params for plotting')
parser.add_argument('--file', type=str, help='results pkl file')


def main(argv):
  args = parser.parse_args(argv[1:])
  params = Bunch(yaml.load(open(args.params)))

  pkl_file = args.file or params.file # allow override on cmd line
  params.file = pkl_file

  plot_drop(params)

if __name__ == "__main__":
  main(sys.argv)

