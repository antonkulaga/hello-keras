# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers import Dropout
from keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import pickle
import cv2
import os
cwd = os.getcwd()
data = "/data/training/hello-keras"

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=False, default=data,
                help="path to input dataset of images")
ap.add_argument("-p", "--plot", required=False, default=os.path.join(data, "plot"),
                help="path to output accuracy/loss plot")
ap.add_argument("-m", "--model", required=False, default=os.path.join(data, "dense.h5"),
                help="path to output trained model")
ap.add_argument("-l", "--label-bin", required=False, default=os.path.join(data, "labels.pkl"),
                help="path to output label binarizer")
args = vars(ap.parse_args())


# grab the image paths and randomly shuffle them
imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(imagePaths)