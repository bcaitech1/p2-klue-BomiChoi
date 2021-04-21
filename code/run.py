from seed import *
import train
import inference
from argument import get_args

seed_everything()
args = get_args()
train.main(args)
inference.main(args)
