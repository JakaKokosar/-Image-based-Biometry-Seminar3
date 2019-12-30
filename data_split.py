import split_folders

from configs import ORIGINAL_DATA_PATH, DATA_PATH

# Split with a ratio.
split_folders.ratio(ORIGINAL_DATA_PATH, output=DATA_PATH, seed=1337, ratio=(.7, .1, .2))
