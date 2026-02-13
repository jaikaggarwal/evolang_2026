import itertools
import string
import json
import numpy as np
import pandas as pd
import os
import sys
import pickle
import matplotlib.pyplot as plt 
import seaborn as sns
from collections import Counter, defaultdict
from itertools import product, chain
from scipy.stats import spearmanr, pearsonr
from urllib import request
from copy import deepcopy
from tqdm import tqdm

from constants import *
# from skimage import color 
# import cv2

def set_intersection(l1, l2):
    """
    Returns the intersection of two lists.
    """
    return list(set(l1).intersection(set(l2)))


def set_union(l1, l2):
    """
    Returns the union of two lists.
    """
    return list(set(l1).union(set(l2)))

def set_difference(l1, l2):
    """
    Returns the difference of two lists.
    """
    return list(set(l1).difference(set(l2)))

def intersect_overlap(l1, l2):
    """
    Returns the intersection of two lists,
    while also describing the size of each list
    and the size of their intersection.
    """
    print(len(l1))
    print(len(l2))
    intersected = set_intersection(l1, l2)
    print(len(intersected))
    return intersected

def jaccard_similarity(l1, l2):
    l1 = set(l1)
    l2 = set(l2)
    return np.round(len(l1.intersection(l2)) / len(l1.union(l2)), 2)

def flatten_logic(arr):
    """
    Flattens a nested array. 

    """
    for i in arr:
        if isinstance(i, (list,tuple)):
            for j in flatten(i):
                yield j
        else:
            yield i


def flatten(arr):
    """
    Wrapper for the generator returned by flatten logic.
    """
    return list(flatten_logic(arr))


def make_directory(dir_name):
    if not os.path.exists(dir_name):
        print(f"Creating directory {dir_name}")
        os.makedirs(dir_name)

def empty_directory(dir):
    files = os.listdir(dir)
    for file in files:
        os.remove(dir + file)


def strip_file_type(path):
    return path[:path.rindex(".")]


def pad_int_with_zeros(i, max_power_of_ten):

    max_num_zeros = int(np.log10(max_power_of_ten))

    if i == 0:
        return "0"*max_num_zeros
    
    else:
        return "0"*(max_num_zeros - 1 - int(np.log10(i))) + str(i)



def load_csv_from_url(url, local_name):

    if not os.path.exists(local_name):
        print("Downloading...")
        request.urlretrieve(url, local_name)
    
    return pd.read_csv(local_name)
    

def load_json(file):
    with open(file, "r") as f:
        return json.load(f)


def serialize_object(obj, file_name):
    with open(file_name, "wb") as f:
        pickle.dump(obj, f)



def load_serialized_object(file_name):
    with open(file_name, "rb") as f:
        return pickle.load(f)
    

def sorted_set(arr):
    seen = []
    for i in arr:
        if i in seen:
            continue
        else:
            seen.append(i)
    return seen


def z_score_arr(arr):
    return (arr - arr.mean())/ arr.std()


def create_batches(arr, num_batches):
    new_arr = []

    batch_size = len(arr) // num_batches

    for i in range(num_batches-1):
        new_arr.append(arr[i*batch_size:(i+1)*batch_size])
    
    new_arr.append(arr[(num_batches-1)*batch_size:])
    return new_arr

def convert_to_base2(arr):
    return arr * np.log2(np.e)