import argparse
from glob import glob
from os.path import normpath
from dedalus.tools import post

parser = argparse.ArgumentParser()

parser.add_argument(
    "directory", nargs=1, help="The directory where the files are located", default="."
)
args = parser.parse_args()

direc = normpath(args.directory[0]) + "/"
title = direc.split("/")[-2]
set_paths = glob(direc + "*_*.h5")

post.merge_sets(direc + title + ".h5", set_paths, cleanup=True)
