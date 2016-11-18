#! -*coding=utf-8 *-
import os
import cv2
import numpy as np
import pandas as pd
try:
	from sklearn.model_selection import train_test_split
except:
	from sklearn.cross_validation import train_test_split
from sklearn.metrics import log_loss, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import joblib

from keras.models import Sequential
from keras.layers import Dropout, Flatten, Convolution2D, MaxPooling2D, ZeroPadding2D, Dense, Activation
from keras.optimizers import RMSprop, Adam
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from keras import backend as K

TRAIN_DIR = 'train/'
TEST_DIR = 'test_stg1/'
FISH_CLASSES = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
ROWS = 90
COLS = 160
CHANNELS = 3

def get_images(fish):
	"""Load files from train folder"""
	fish_dir = TRAIN_DIR + '{}'.format(fish)
	images = [fish + '/' + im for im in os.listdir(fish_dir)]
	return images

def read_image(src):
	"""Read and resize individual images"""
	im = cv2.imread(src, cv2.IMREAD_COLOR)
	im = cv2.resize(im, (COLS, ROWS), interpolation=cv2.INTER_CUBIC)
	return im

def load_train_data():
	'''Loading and Preprocessing Data'''
	dim_ordering = K.image_dim_ordering()
	file_name = "XY-Train-%s.dat" %dim_ordering
	if os.path.exists(file_name):
		X_all, y_all = joblib.load(file_name)
		return X_all, y_all

	files = []
	y_all = []

	for fish in FISH_CLASSES:
		fish_files = get_images(fish)
		files.extend(fish_files)

		y_fish = np.tile(fish, len(fish_files))
		y_all.extend(y_fish)
		print("{0} photos of {1}".format(len(fish_files), fish))

	y_all = np.array(y_all)
	X_all = np.ndarray((len(files), ROWS, COLS, CHANNELS), dtype=np.uint8)

	for i, im in enumerate(files):
		X_all[i] = read_image(TRAIN_DIR+im)
		if i%500 == 0:
			print('Processed {} of {}'.format(i, len(files)))

	# normalizing
	X_all = X_all.astype('float32') / 255.0

	dim_ordering = K.image_dim_ordering()
	if dim_ordering == 'th':
		X_all = X_all.reshape(X_all.shape[0], CHANNELS, ROWS, COLS)
	else:
		X_all = X_all.reshape(X_all.shape[0], ROWS, COLS, CHANNELS)

	print(X_all.shape)

	# One-Hot-Encode the labels.
	y_all = LabelEncoder().fit_transform(y_all)
	y_all = np_utils.to_categorical(y_all)

	joblib.dump((X_all, y_all), file_name, compress=3)
	return X_all, y_all

def load_test_data():
	'''Loading Test Data'''
	dim_ordering = K.image_dim_ordering()
	file_name = "XY-Test-%s.dat" %dim_ordering
	if os.path.exists(file_name):
		X_test, test_files = joblib.load(file_name)
		return X_test, test_files

	test_files = [im for im in os.listdir(TEST_DIR)]
	X_test = np.ndarray((len(test_files), ROWS, COLS, CHANNELS), dtype=np.uint8)

	for i, im in enumerate(test_files):
		X_test[i] = read_image(TEST_DIR + im)

	if dim_ordering == 'th':
		X_test = X_test.reshape(X_test.shape[0], CHANNELS, ROWS, COLS)
	else:
		X_test = X_test.reshape(X_test.shape[0], ROWS, COLS, CHANNELS)

	joblib.dump((X_test, test_files), file_name, compress=3)
	return X_test, test_files

def center_normalize(x):
	return (x - K.mean(x)) / K.std(x)

def create_model():
	'''
	Create The Model:
		Pretty typical CNN in Keras with a plenty of dropout regularization
		between the fully connected layers.
	'''

	optimizer = RMSprop(lr=1e-4)
	objective = 'categorical_crossentropy'

	dim_ordering = K.image_dim_ordering()
	if dim_ordering == 'th':
		input_shape = (CHANNELS, ROWS, COLS)
	else:
		input_shape = (ROWS, COLS, CHANNELS)

	#============================================================================================
	model = Sequential()

	'''
	# tensorflow as backend
	model.add(Activation(activation=center_normalize, input_shape=(ROWS, COLS, CHANNELS)))

	model.add(Convolution2D(32, 5, 5, border_mode='same', activation='relu', dim_ordering='tf'))
	model.add(Convolution2D(32, 5, 5, border_mode='same', activation='relu', dim_ordering='tf'))
	model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='tf'))

	model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu', dim_ordering='tf'))
	model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu', dim_ordering='tf'))
	model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='tf'))

	model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu', dim_ordering='tf'))
	model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu', dim_ordering='tf'))
	model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='tf'))

	model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu', dim_ordering='tf'))
	model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu', dim_ordering='tf'))
	model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='tf'))
	'''

	'''
	# theano as backend
	# input: 100x100 images with 3 channels -> (3, 100, 100) tensors.
	# then applies 32 convolution filters of size 5x5 each.
	model.add(Activation(activation=center_normalize, input_shape=(CHANNELS, ROWS, COLS)))

	model.add(Convolution2D(32, 5, 5, border_mode='same', activation='relu', dim_ordering='th'))
	model.add(Convolution2D(32, 5, 5, border_mode='same', activation='relu', dim_ordering='th'))
	model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='th'))

	model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu', dim_ordering='th'))
	model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu', dim_ordering='th'))
	model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='th'))

	model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu', dim_ordering='th'))
	model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu', dim_ordering='th'))
	model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='th'))

	model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu', dim_ordering='th'))
	model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu', dim_ordering='th'))
	model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering='th'))
	'''

	model.add(Activation(activation=center_normalize, input_shape=input_shape))

	model.add(Convolution2D(32, 5, 5, border_mode='same', activation='relu', dim_ordering=dim_ordering))
	model.add(Convolution2D(32, 5, 5, border_mode='same', activation='relu', dim_ordering=dim_ordering))
	model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering=dim_ordering))

	model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu', dim_ordering=dim_ordering))
	model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu', dim_ordering=dim_ordering))
	model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering=dim_ordering))

	model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu', dim_ordering=dim_ordering))
	model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu', dim_ordering=dim_ordering))
	model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering=dim_ordering))

	model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu', dim_ordering=dim_ordering))
	model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu', dim_ordering=dim_ordering))
	model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering=dim_ordering))

	#============================================================================================

	model.add(Flatten())
	model.add(Dense(256, activation='relu'))
	model.add(Dropout(0.5))

	model.add(Dense(64, activation='relu'))
	model.add(Dropout(0.5))

	model.add(Dense(len(FISH_CLASSES)))
	# model.add(Activation('sigmoid'))
	model.add(Activation('softmax'))

	model.compile(loss=objective, optimizer=optimizer)
	return model

def train(model, X_train, y_train, nb_epoch=50):
	'''Training The Model'''
	early_stopping = EarlyStopping(monitor='val_loss', patience=4, verbose=1, mode='auto')

	model.fit(X_train, y_train,
			  batch_size=64, nb_epoch=nb_epoch,
			  validation_split=0.2,
			  verbose=1, shuffle=True,
			  callbacks=[early_stopping]
	)

	model_file = "model-weights-epoch%d.hdf5" %(nb_epoch)
	model.save(model_file, overwrite=True)
	print

def predict(model, X_test):
	'''Predict Lables'''
	preds = model.predict(X_test, verbose=0)
	return preds

def validate(model, X_valid, y_valid):
	'''Validate The Model'''
	# Compute log-loss
	preds_prob = predict(model, X_valid)
	print (" log loss: {}".format(log_loss(y_valid, preds_prob)))

	# Compute precision, recall, F - measure and support
	preds_catagory = np_utils.to_categorical(preds_prob.argmax(axis=1), nb_classes=len(FISH_CLASSES))
	prfs = precision_recall_fscore_support(y_valid, preds_catagory)
	# fields = ["precision", "recall", "fscore", "support"]
	fields = ["precision", "recall", "fscore"]
	for k, v in zip(fields, prfs):
		v_str = "\t".join(["{:.10f}".format(i) for i in v.tolist()])
		info = "%9s: %s" %(k, v_str)
		print(info)

def gen_submission(model, X_test, test_files, nb_epoch=50):
	'''Testing On The TEST Data Set'''
	test_preds = model.predict(X_test, verbose=0)

	submission = pd.DataFrame(test_preds, columns=FISH_CLASSES)
	submission.insert(0, 'image', test_files)
	# print submission.head()
	for i in range(1, 1000):
		submission_csv = "result/submission-epoch%d-%d.csv" %(nb_epoch, i)
		if not os.path.exists(submission_csv):
			submission.to_csv(submission_csv, index=False)
			break

if __name__ == "__main__":

	clear = "cls" if os.name == "nt" else "clear"
	os.system(clear)

	X_all, y_all = load_train_data()
	X_train, X_valid, y_train, y_valid = train_test_split(X_all, y_all, test_size=0.2, random_state=23)
	X_test, test_files = load_test_data()

	# nb_epoch = 50
	for nb_epoch in range(20, 60, 10):
		print "=" * 120
		model = create_model()
		model_file = "model-weights-epoch%d.hdf5" %(nb_epoch)
		if os.path.exists(model_file):
			model.load_weights(model_file)
		else:
			train(model, X_train, y_train, nb_epoch=nb_epoch)

		validate(model, X_valid, y_valid)

		gen_submission(model, X_test, test_files, nb_epoch=nb_epoch)


