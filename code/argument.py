import argparse
import json
import os
from attrdict import AttrDict

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', '-c', type=str, default="config1.json")
    cli_args = parser.parse_args()
    print(cli_args)
    with open(os.path.join('./config', cli_args.config_file)) as f:
        args = AttrDict(json.load(f))
    return args