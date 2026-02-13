import os
import string

# ----------------- General Constants ----------------- #

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
IMAGES_DIR = os.path.join(ROOT_DIR, "images/")
CONFIG_DIR = os.path.join(ROOT_DIR, "configs/")
DATA_DIR = os.path.join(ROOT_DIR, "data/")
DATA_LOADER_DIR = os.path.join(DATA_DIR + "serialized_data_loaders/")

ALPHABET = string.ascii_letters[26:]
