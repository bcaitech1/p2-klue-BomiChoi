from seed import *
import train
import train_KNH
import inference
from argument import get_args

seed_everything()
args = get_args()
# train.main(args)
# train_KNH.main(args)
inference.main(args)
