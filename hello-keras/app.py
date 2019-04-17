# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import pickle
import cv2
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
cwd = os.getcwd()

ap.add_argument("-d", "--dataset",
                default="/data/training/hello-keras",
                required=False, help="path to input dataset of images")

ap.add_argument("-m", "--model",
                default=os.path.join(cwd,".model"),
                required=False,
                help="path to output trained model")

ap.add_argument("-l", "--label-bin", required=False,
                default=os.path.join(cwd,"output", "labels"),
                help="path to output label binarizer")

ap.add_argument("-p", "--plot", required=False,
                default=os.path.join(cwd,"output", "plots"),
                help="path to output accuracy/loss plot")

args = vars(ap.parse_args())