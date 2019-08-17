#####################################################################
# aim of this script is to have a class                  			#
# that gives you three main features preprocessed and converted to  #
# a fixed format													#
#####################################################################

import json
import os
import os.path as osp
import pickle
from math import factorial

import numpy as np


#################################################################################################
# Scattering processing
# Returns a 2D vector 20 x 80, 2 pixels from top and from bottom are eliminated due to device dependence
# If bad signal, returns zeros
#################################################################################################


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
    if N > 450:
        return np.zeros([20, cut * 2])
    else:
        N2 = int(N / 2)
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

    #################################################################################################


# Lifetime processing
# Returns a vector of 24 values = sum over all channels - least significant (for noise reduction)
# Returns also normalized to max over all integrals of four channels
# If bad signal, returns zeros
####################################################################################################################
def normalize_lifitime(opd, normalize=True):
    liti = np.asarray(opd['Lifetime']).reshape(-1, 64)
    lt_im = np.zeros((4, 24))
    liti_low = np.sum(liti, axis=0)
    maxx = np.max(liti_low)
    ind = np.argmax(liti_low)
    if (ind < 10) or (ind > 44):
        return np.zeros((4, 24))
    else:
        lt_im[:, :] = liti[:, ind - 4:20 + ind]
        weights = []
        for i in range(4):
            weights.append(np.sum(liti[i, ind - 4:12 + ind]) - np.sum(liti[i, 0:16]))
        # A = np.asarray(liti_low[0:24])
        B = np.asarray(weights)
        A = lt_im
        if (normalize == True):
            if (maxx > 0) and (B.max() > 0):
                return [A / maxx, B / B.max()]
            else:
                return [np.zeros((4, 24)), np.zeros(4)]
        else:
            return [A, B]

    #################################################################################################


# Smooth function
####################################################################################################################
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


########################################################################################################################
# Fluorescence spectrum processing
# Returns a vector of 32 values = sum over 2, 3, 4 and 5th acquisitions -
# last or before last acquisition (for noise reduction)
# Fluorescence spectrum is only used in normalized shape
# If bad signal, returns zeros
####################################################################################################################
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
            return A / A.max()  # Spectrum is normalized
        else:
            return np.zeros((4, 32))
    else:
        return res_spec

    #################################################################################################


# Particle equivalent otpical size
# Returns size estimation in micrometers (um)
# Empirical function
########################################################################################################################
def size_particle(opd, scale_factor=1):
    image = np.asarray(opd['Scattering']['Image']).reshape(-1, 24)
    x = (np.asarray(image, dtype='float32').reshape(-1, 24)[:, :]).sum()
    if x < 5500000:
        return 0.5
    elif (x >= 5500000) and (x < 500000000):
        return 9.95e-01 * np.log(3.81e-05 * x) - 4.84e+00
    else:
        return 0.0004 * x ** 0.5 - 3.9


def save_files(path):
    """
    Save preprocessed file into pickle object
    :param path: path to json file
    :return:
    """
    # separate path into filepath and extension
    filename, file_extension = os.path.splitext(path)
    # take only filename
    file_name = osp.basename(filename)

    spectrum_list = []
    scatter_list = []
    lifetime_list1 = []
    lifetime_list2 = []
    size_list = []

    # label = path.split('.')[0]
    data = json.loads(open(path).read())

    numbof = len(data["Data"])
    # print(numbof)
    i = 0
    while i < numbof:
        x = np.sum([np.float64(j) for j in data["Data"][i]["Scattering"]["Image"]])
        specmax = np.max(data["Data"][i]["Spectrometer"])
        if x < 5500000:
            integral_D = 0.5
        elif (x >= 5500000) and (x < 500000000):
            integral_D = 9.95e-01 * np.log(3.81e-05 * x) - 4.84e+00
        else:
            integral_D = 0.0004 * x ** 0.5 - 3.9

        if specmax < 2500:  # filter with max spectrum (can also filter with lifetime or scattering integral (which represents the size))
            del data["Data"][i]
            numbof = len(data["Data"])
        else:
            i += 1

    for i in range(len(data["Data"])):
        scat = normalize_image(data["Data"][i], cut=60, normalize=False, smooth=True)
        life1 = normalize_lifitime(data["Data"][i], normalize=True)
        spec = spec_correction(data["Data"][i], normalize=True)
        # size = size_particle(data["Data"][i], scale_factor=1)

        scatter_list.append(scat)
        # size_list.append(size)
        lifetime_list1.append(life1[0])
        lifetime_list2.append(life1[1])
        spectrum_list.append(spec)

    folders_to_create = ['scatter/cut', 'spectrum/images',
                         'spectrum/images_life1', 'spectrum/images_life2']
    extensions = ['_scatter.pckl', '_spectrum.pckl', '_lifetime1.pckl', '_lifetime2.pckl']
    files_to_save = [scatter_list, spectrum_list, lifetime_list1, lifetime_list2]

    # check if folders exists, otherwise crate it
    for folder in folders_to_create:
        if not osp.exists(osp.join(TRANSFORMED_DIR, folder)):
            os.makedirs(osp.join(TRANSFORMED_DIR, folder))

    for index, filee in enumerate(files_to_save):
        str_name = TRANSFORMED_DIR + '/' + \
                   folders_to_create[index] + '/' + \
                   file_name + \
                   extensions[index]
        f = open(str_name, 'wb')
        pickle.dump(filee, f)
        f.close()


if __name__ == "__main__":

    # iterate through all data folders, get all files in data folder and transform each file
    for folder in LIST_OF_RAW_DIR:
        files = os.listdir(folder)
        for path in files:
            save_files(osp.join(folder, path))
