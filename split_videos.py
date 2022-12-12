import splitfolders
import random

splitfolders.ratio("data", # The location of dataset
                   output="split_data", # The output location
                   seed=random.seed(), # The number of seed
                   ratio=(.7, 0, .3), # The ratio of splited dataset
                   group_prefix=None, # If your dataset contains more than one file like ".jpg", ".pdf", etc
                   move=False # If you choose to move, turn this into True
                   )