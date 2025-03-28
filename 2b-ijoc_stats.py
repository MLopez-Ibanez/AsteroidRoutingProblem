#!/usr/bin/env python3
import os
import os.path
from glob import glob
import re
import sys
import numpy as np
import pandas as pd

from pathlib import Path
home = str(Path.home())

res_dir = home + "/scratch/ijoc/results/"

if not os.path.exists(res_dir):
    sys.stderr.write(f'ERROR: directory {res_dir} was not found!')
    sys.exit(1)

unique_files = []
# traverse root directory, and list directories as dirs and files as files
for root, dirs, files in os.walk(res_dir):
    # print(f'root: {root}')
    # print(f'dirs: {dirs}')
    # print(f'files: {files}')
    for file in files:
        result = re.match(r"(.+)\.csv\.xz", file)
        if result:
            unique_files.append(root + "/" + file)
unique_files = set(unique_files)

df = pd.concat([ pd.read_csv(x) for x in unique_files])
df.instance = df.instance.str.replace('_8', '_08')
fitness = df.groupby(['Solver', "instance", 'seed'])['Fitness'].min()
print(fitness.groupby(['Solver', "instance"]).agg(["min", "mean", "std"]))
time = df.groupby(['Solver', "instance", 'seed'])['run_time'].min()
print(time.groupby(['Solver', "instance"]).agg(["min", "mean", "std"]))

