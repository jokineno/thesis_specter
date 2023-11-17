import json
import argparse
import glob
import logging
import pickle
from pathlib import Path
import os
import jsonpickle
logger = logging.getLogger(logging.basicConfig(level=logging.INFO))


def unpickle(input_path):
    with open(input_path, "rb") as f_in:
        unpickler = pickle.Unpickler(f_in)

        while True:
            try:
                instance = unpickler.load()
                # TODO instance to json
            except EOFError:
                break
        logger.info("Finished unpickling {}".format(input_path))

def mkdir(path):
    if os.path.exists(path):
        logger.info("Path {} exists".format(path))
    else:
        logger.info("Creating output path  {}".format(path))
        Path(path).mkdir(parents=True)


def main(folder, output_path):

    path = folder.strip("/") + "/" + "*.p"
    pickled_files = glob.glob(path)

    mkdir(output_path)

    if len(pickled_files) == 0:
        logger.info("No pickled files. Error?")
    for pickled_file in pickled_files:
        logger.info("Unpickling file {}".format(pickled_file))
        unpickled_file = unpickle(pickled_file)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    folder = "thesis_data/preprocessed_demo/"
    output_path = "./thesis_data/preprocessed_demo/unpickled"
    # ap.add_argument("--dir", required=True)
    # ap.add_argument("--output_path", required=True)
    # args = ap.parse_args()
    #
    # folder = args.dir
    # output_path = args.output_path
    main(folder, output_path)