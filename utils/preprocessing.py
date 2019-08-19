from math import factorial

import numpy as np


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
        if smooth:
            for i in range(20):
                im[:, i] = savitzky_golay(im[:, i] ** 0.5, 5, 3)
        # im[:,0:2] = 0
        # im[:,22:24] = 0
        im = np.transpose(im)
        if normalize:
            return np.asarray(im / im.sum())
        else:
            return np.asarray(im).astype('float32')


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
        if normalize:
            if (maxx > 0) and (B.max() > 0):
                return (A / maxx).astype('float32'), (B / B.max()).astype('float32')
        else:
            return A.astype('float32'), B.astype('float32')


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
    if normalize:
        A = res_spec
        if A.max() > 0:
            return (A / A.max()).astype('float32')
    else:
        return res_spec


def size_particle(opd, scale_factor=1):
    image = np.asarray(opd['Scattering']['Image']).reshape(-1, 24)
    x = (np.asarray(image, dtype='float32').reshape(-1, 24)[:, :]).sum()
    if x < 5500000:
        return 0.5
    elif (x >= 5500000) and (x < 500000000):
        return (9.95e-01 * np.log(3.81e-05 * x) - 4.84e+00).astype('float32')
    else:
        return (0.0004 * x ** 0.5 - 3.9).astype('float32')


def convert_filename_to_label(label):
    filename = label.split(".")[0].split("_")

    if len(filename) == 1:
        return filename[0]
    else:
        return filename[1]


def label_to_index(files):
    labels = []
    for file in files:
        if file.split(".")[-1] != 'json':
            continue
        labels.append(convert_filename_to_label(file))
    labels = sorted(list(set(labels)))

    num_to_class = dict((label, index) for (index, label) in enumerate(labels))
    return num_to_class


def check_shapes(file, shape_type, is_list=False):
    if is_list:
        if len(file) == 0:
            return False

    if shape_type == 0:
        if file.shape != (20, 120):
            return False
        else:
            return True

    elif shape_type == 1:
        if type(float(file)) != float:
            return False
        else:
            return True

    elif shape_type == 2:
        if file.shape != (4, 24):
            return False
        else:
            return True

    elif shape_type == 3:
        if file.shape != (4, 32):
            return False
        else:
            return True

    elif shape_type == 4:
        if file.shape != (4,):
            return False
        else:
            return True


def calculate_and_check_shapes(file_data, file_name, spec_max, data, targets, class_to_num):
    if spec_max > 2500:
        scat = normalize_image(file_data, cut=60, normalize=False, smooth=True)
        life1 = normalize_lifitime(file_data, normalize=True)
        spec = spec_correction(file_data, normalize=True)
        size = size_particle(file_data, scale_factor=1)

        if scat is not None and life1 is not None and \
                spec is not None and size is not None and \
                check_shapes(scat, is_list=True, shape_type=0) and \
                check_shapes(size, shape_type=1) and \
                check_shapes(life1[0], is_list=True, shape_type=2) and \
                check_shapes(spec, is_list=True, shape_type=3) and \
                check_shapes(life1[1], shape_type=4):

            data[0].append(scat.astype('float32'))
            data[1].append(size.astype('float32'))
            data[2].append(life1[0].astype('float32'))
            data[3].append(spec.astype('float32'))
            data[4].append(life1[1].astype('float32'))
            targets.append(class_to_num[convert_filename_to_label(file_name)])
