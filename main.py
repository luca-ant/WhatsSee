from multiprocessing import Process

import whats_see
import os
import sys

working_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
dataset_name="flickr"
whatssee = whats_see.WhatsSee(dataset_name, working_dir)


p = Process(target=whatssee.train, args=(2, 3,))
p.start()
