#!/usr/bin/env python
# coding: utf-8

import itertools
import json
import os
from datetime import datetime
from math import factorial

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# preprocessing
from utils.converting_raw_data import label_to_index


def normalize_image(opd, cut, normalize=False, smooth=True):
	if len(opd['Scattering']) > 3:
		image = np.asarray(opd['Scattering'])
	else:
		image = np.asarray(opd['Scattering']['Image'])
	image = np.reshape(image, [-1, 24])
	im = np.zeros([2000, 20])
	cut = int(cut)
	im_x = np.sum(image, axis=1) / 256
	N = len(im_x)
	if N < 450:
		cm_x = 0
		for i in range(N):
			cm_x += im_x[i] * i
		cm_x /= im_x.sum()
		cm_x = int(cm_x)
		im[1000 - cm_x:1000 + (N - cm_x), :] = image[:, 2:22]
		im = im[1000 - cut:1000 + cut, :]
		if smooth == True:
			for i in range(20):
				im[:, i] = savitzky_golay(im[:, i] ** 0.5, 5, 3)
		# im[:,0:2] = 0
		# im[:,22:24] = 0
		im = np.transpose(im)
		if normalize == True:
			return np.asarray(im / im.sum())
		else:
			return np.asarray(im)


def normalize_lifitime(opd, normalize=True):
	liti = np.asarray(opd['Lifetime']).reshape(-1, 64)
	lt_im = np.zeros((4, 24))
	liti_low = np.sum(liti, axis=0)
	maxx = np.max(liti_low)
	ind = np.argmax(liti_low)
	if (ind > 10) and (ind < 44):
		lt_im[:, :] = liti[:, ind - 4:20 + ind]
		weights = []
		for i in range(4):
			weights.append(np.sum(liti[i, ind - 4:12 + ind]) - np.sum(liti[i, 0:16]))
		B = np.asarray(weights)
		A = lt_im
		if (normalize == True):
			if (maxx > 0) and (B.max() > 0):
				return A / maxx, B / B.max()
		else:
			return A, B


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
	try:
		window_size = np.abs(np.int(window_size))
		order = np.abs(np.int(order))
	except ValueError:
		raise ValueError("window_size and order have to be of type int")
	if window_size % 2 != 1 or window_size < 1:
		raise TypeError("window_size size must be a positive odd number")
	if window_size < order + 2:
		raise TypeError("window_size is too small for the polynomials order")
	order_range = range(order + 1)
	half_window = (window_size - 1) // 2
	b = np.mat([[k ** i for i in order_range] for k in range(-half_window, half_window + 1)])
	m = np.linalg.pinv(b).A[deriv] * rate ** deriv * factorial(deriv)
	firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
	lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
	y = np.concatenate((firstvals, y, lastvals))
	return np.convolve(m[::-1], y, mode='valid')


def spec_correction(opd, normalize=True):
	spec = np.asarray(opd['Spectrometer'])
	spec_2D = spec.reshape(-1, 8)

	if (spec_2D[:, 1] > 20000).any():
		res_spec = spec_2D[:, 1:5]
	else:
		res_spec = spec_2D[:, 0:4]

	for i in range(res_spec.shape[1]):
		res_spec[:, i] -= np.minimum(spec_2D[:, 6], spec_2D[:, 7])

	for i in range(4):
		res_spec[:, i] = savitzky_golay(res_spec[:, i], 5, 3)  # Spectrum is smoothed
	res_spec = np.transpose(res_spec)
	if normalize == True:
		A = res_spec
		if (A.max() > 0):
			return A / A.max()
	else:
		return res_spec


def size_particle(opd, scale_factor=1):
	image = np.asarray(opd['Scattering']['Image']).reshape(-1, 24)
	x = (np.asarray(image, dtype='float32').reshape(-1, 24)[:, :]).sum()
	if x < 5500000:
		return 0.5
	elif (x >= 5500000) and (x < 500000000):
		return 9.95e-01 * np.log(3.81e-05 * x) - 4.84e+00
	else:
		return 0.0004 * x ** 0.5 - 3.9


def confusion_matrix(save, cm, classes, normalize, klasa, title='Confusion matrix', cmap=plt.cm.Blues):
	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

	plt.figure(figsize=((30, 30)))
	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45, fontsize=20)
	plt.yticks(tick_marks, classes, fontsize=20)

	fmt = '.2f' if normalize else 'd'
	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, format(cm[i, j], fmt),
				 horizontalalignment="center",
				 color="white" if cm[i, j] > thresh else "black")

	matplotlib.rcParams.update({'font.size': 20})
	plt.tight_layout()
	plt.ylabel('True label', fontsize=20)
	plt.xlabel('Predicted label', fontsize=20)
	if save:
		plt.savefig('/home/pedjamat/Documents/models/cm/' + str(klasa) + '.png')
	else:
		plt.show()


def plot_tr(epoch_list, all_train_losses, all_validation_losses):
	plt.figure(figsize=((10, 8)))
	plt.plot(epoch_list, all_train_losses)
	plt.plot(epoch_list, all_validation_losses)
	plt.xlabel("Epoch number")
	plt.ylabel("Cost")
	plt.legend(["Train", "Validation"], fontsize=20)
	matplotlib.rcParams.update({'font.size': 20})
	# plt.savefig('/home/pedjamat/Pictures/overfitting.png')
	plt.show()


if __name__ == '__main__':

	targets = []

	os.chdir('/mnt/hdd/data/')
	files = sorted(os.listdir())
	print(files)
	data = [[], [], [], [], []]

	class_to_num = label_to_index(files)

	for file in files:
		if file.split(".")[-1] != 'json':
			continue

		print(f'Current file is {file} and time is {datetime.now().time()}')
		raw_data = json.loads(open(file).read())
		data_list = [[], [], [], [], []]

		if file.split(".")[0] == "Skrob_1sat" or file.split(".")[0] == "Skrob_2dana" or file.split(".")[0] == "Alnus" or \
			file.split(".")[0] == "Corylus" or file.split(".")[0] == "Cupressus" or file.split(".")[
			0] == "Fraxinus excelsior" or file.split(".")[0] == "Ulmus":
			for i in range(len(raw_data)):
				specmax = np.max(raw_data["Data"][i]["Spectrometer"])
				if specmax > 2500:  # preprocessing

					scat = normalize_image(raw_data[i], cut=60, normalize=False, smooth=True)
					life1 = normalize_lifitime(raw_data[i], normalize=True)
					spec = spec_correction(raw_data[i], normalize=True)
					size = size_particle(raw_data[i], scale_factor=1)

					if scat is not None and spec is not None and life1 is not None:
						data_list[0].append(scat)
						data_list[1].append(size)
						data_list[2].append(life1[0])
						data_list[3].append(spec)
						data_list[4].append(life1[1])

		else:
			for i in range(len(raw_data["Data"])):
				specmax = np.max(raw_data["Data"][i]["Spectrometer"])
				if specmax > 2500:  # preprocessing

					scat = normalize_image(raw_data["Data"][i], cut=60, normalize=False, smooth=True)
					life1 = normalize_lifitime(raw_data["Data"][i], normalize=True)
					spec = spec_correction(raw_data["Data"][i], normalize=True)
					size = size_particle(raw_data["Data"][i], scale_factor=1)

					if scat is not None and spec is not None and life1 is not None:
						data_list[0].append(scat)
						data_list[1].append(size)
						data_list[2].append(life1[0])
						data_list[3].append(spec)
						data_list[4].append(life1[1])

		for p in range(len(data)):
			data[p].append(data_list[p])

	lista = []
	for i in range(len(data[2])):
		for j in range(len(data[2][i])):
			if np.where(data[2][i][j][0, :] == np.max(data[2][i][j][0, :]))[0].shape[0] > 1:
				lista.append([i, j])

	b = 0
	for i in range(len(data[0])):
		b += len(data[0][i])
	print(b)

	lista = []
	for i in range(len(data[2])):
		pom_l = []
		for j in range(len(data[2][i])):

			if i == 0:
				l = ["Cupressus"]
			elif i == 1:
				l = ["Fraxinus excelsior"]
			else:
				l = ["Ulmus"]

			if np.where(data[2][i][j][0, :] == np.max(data[2][i][j][0, :]))[0].shape[0] > 1:
				l.append("Yes")
			else:
				l.append("No")

			for k in range(4):
				if k != 2:
					l.append(np.max(data[2][i][j][k, :]) / np.e)
			pom_l.append(l)
		lista.append(pom_l)

	features = ["Pollen type", "Saturated", "Time of band 1", "Time of band 2", "Time of band 3"]
	amb1 = pd.DataFrame(columns=features)

	for i in range(len(lista)):
		for j in range(len(lista[i])):
			amb1.loc[len(amb1)] = lista[i][j]

	amb1.to_csv("./../data/Time of lifetime.csv", index=False)
