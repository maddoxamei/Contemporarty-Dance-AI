import argparse

parser = argparse.ArgumentParser()

parser.add_argument('steps', type=int, default=1,
                   help='Steps or Skips (Cuts down on the patterns)')
parser.add_argument('-train', action="store_true",
                   help='True: Train on dataset, False: Sample with trained model')
parser.add_argument('-train', action="store_true",
                   help='True: Train on dataset, False: Sample with trained model')

args = parser.parse_args()