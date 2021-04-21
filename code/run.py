from seed import *
from train import *
import inference

import argparse
import json
from attrdict import AttrDict

seed_everything()

parser = argparse.ArgumentParser()
parser.add_argument('--config_file', '-c', type=str, default="config1.json")
cli_args = parser.parse_args()
print(cli_args)

with open(os.path.join('./config', cli_args.config_file)) as f:
    args = AttrDict(json.load(f))

# train(args)
inference.main(args)
