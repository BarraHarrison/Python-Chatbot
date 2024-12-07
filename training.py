# This is for training the chatbot
import random
import json
import pickle
import numpy as np
import tensorflow as tf

# NLTK stands for natural language toolkit
import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD


