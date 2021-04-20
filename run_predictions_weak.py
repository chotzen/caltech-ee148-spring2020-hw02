import os
import numpy as np
import json
from tqdm import tqdm
from PIL import Image


import matplotlib.pyplot as plt
import matplotlib.patches as patches


def load_filters():
    filters_path = 'new_filters/'
    filter_names = [f for f in sorted(os.listdir(filters_path)) if '.jpg' in f]
    filters = []

    for i in range(len(filter_names)):
        f = Image.open(os.path.join(filters_path, filter_names[i]))
        f_img = np.asarray(f)
        filters.append(f_img.copy().astype(float))

    return filters[0:1]


def compute_convolution(I, T, stride=1):
    '''
    This function takes an image <I> and a template <T> (both numpy arrays) 
    and returns a heatmap where each grid represents the output produced by 
    convolution at each location. You can add optional parameters (e.g. stride, 
    window_size, padding) to create additional functionality. 
    '''
    (filt_width, filt_height, _) = np.shape(T)

    n_horizontal_prods = I.shape[0] - filt_width + 1
    n_vertical_prods = I.shape[1] - filt_height + 1
    heatmap = np.empty((n_horizontal_prods, n_vertical_prods))

    filt = T.flatten()

    for x in range(0, n_horizontal_prods, stride):
        for y in range(0, n_vertical_prods, stride):
            section = I[x:x + filt_width, y:y + filt_height, :].copy().astype(float)
            section = section.flatten()
            section = section / np.linalg.norm(section)
            heatmap[x, y] = np.dot(filt / np.linalg.norm(filt), section)

    # plt.imshow(heatmap, cmap='hot', interpolation='nearest')
    # plt.show()


    return heatmap


def predict_boxes(bounding_boxes, heatmap, filt_width=7, filt_height=11):
    '''
    This function takes heatmap and returns the bounding boxes and associated
    confidence scores.
    '''

    for threshold in np.arange(1.0, 0.75, -0.01):

        heatmap_thresh = heatmap.copy()
        heatmap_thresh[heatmap < threshold] = 0

        for x, y in zip(*heatmap_thresh.nonzero()):
            x = x
            y = y
            x2 = x + filt_width
            y2 = y + filt_height


            merged = False
            # check for mergeable bounding boxes and merge them if we can. this isn't perfect, especially when one
            # box surrounds the other.
            for other_rect in bounding_boxes:
                for xc, yc in [(x, y), (x2, y), (x, y2), (x2, y2)]:
                    if other_rect[0] <= xc <= other_rect[2] and other_rect[1] <= yc <= other_rect[3]:
                        merged = True
                        if other_rect[4] == threshold:
                            other_rect[0] = min(int(x), other_rect[0])
                            other_rect[1] = min(int(y), other_rect[1])
                            other_rect[2] = max(int(x2), other_rect[2])
                            other_rect[3] = max(int(y2), other_rect[3])

                        # don't merge if we found a better prediction
                        elif other_rect[4] < threshold:
                            merged = False


            if not merged:
                bounding_boxes.append([int(k) for k in [x, y, x2, y2]] + [threshold])




def cubic_downsample(I, scale):
    n_rows = int(np.ceil(I.shape[0] * scale))
    n_cols = int(np.ceil(I.shape[1] * scale))

    # First, do ~~bilinear~~ cubic interpolation on rows
    row_result = np.empty((n_rows, I.shape[1], 3))
    for j in range(n_rows):
        np.arange(np.floor(i) - 1, np.floor(i) + 3, 1)
        i = I.shape[0] * j / n_rows
        i0 = int(np.floor(i))
        i1 = int(np.ceil(i))
        if i == i0:
            i1 = i0 + 1
        row_result[j, :, :] = (i - i0) * I[i0, :, :] + (i1 - i) * I[i1, :, :]

    result = np.empty((n_rows, n_cols, 3))

    # Then, do the same to columns
    for j in range(n_cols):
        i = I.shape[1] * j / n_cols
        i0 = int(np.floor(i))
        i1 = int(np.ceil(i))
        if i == i0:
            i1 = i0 + 1
        result[:, j, :] = (i - i0) * row_result[:, i0, :] + (i1 - i) * row_result[:, i1, :]

    return result.astype(int)


def normalize_image(I):
    I = I.copy().astype(float)
    norm_means = []
    norm_stds = []

    # normalize image around [-1, 1] ish
    for i in range(3):
        mean = np.mean(I[:, :, i])
        std = np.std(I[:, :, i])
        norm_means.append(mean)
        norm_stds.append(std)

        I[:, :, i] = (I[:, :, i] - mean) / (2 * std)

    return I, norm_means, norm_stds


def normalize_filt(filt, means, stds):
    for i in range(3):
        filt[:, :, i] = (filt[:, :, i] - means[i]) / stds[i]

    return filt


def detect_red_light_mf(I):
    '''
    This function takes a numpy array <I> and returns a list <output>.
    The length of <output> is the number of bounding boxes predicted for <I>. 
    Each entry of <output> is a list <[row_TL,col_TL,row_BR,col_BR,score]>. 
    The first four entries are four integers specifying a bounding box 
    (the row and column index of the top left corner and the row and column 
    index of the bottom right corner).
    <score> is a confidence score ranging from 0 to 1. 

    Note that PIL loads images in RGB order, so:
    I[:,:,0] is the red channel
    I[:,:,1] is the green channel
    I[:,:,2] is the blue channel
    '''

    filters = load_filters()

    I_norm, norm_means, norm_stds = normalize_image(I)
    output = []

    for filt_num, filt in enumerate(filters):
        norm_filter = normalize_filt(filt, norm_means, norm_stds)

        conv_heatmap = compute_convolution(I_norm, norm_filter, stride=2)
        predict_boxes(output, conv_heatmap, norm_filter.shape[0], norm_filter.shape[1])

    # You may use multiple stages and combine the results
    # T = np.random.random((template_height, template_width))
    #
    # heatmap = compute_convolution(I, T)
    # output = predict_boxes(heatmap)

    # Comment the following lines out to toggle bounding box visualization:
    #---
    # fig, ax = plt.subplots()
    # ax.imshow(I.astype(int))
    #
    # for box in output:
    #     rect = patches.Rectangle((box[1], box[0]), box[3] - box[1], box[2] - box[0], linewidth=1, edgecolor='r',
    #                              facecolor='none')
    #     ax.add_patch(rect)
    #
    # plt.show()
    #---

    for i in range(len(output)):
        assert len(output[i]) == 5
        assert (output[i][4] >= 0.0) and (output[i][4] <= 1.0)

    return output



# Note that you are not allowed to use test data for training.
# set the path to the downloaded data:
data_path = '../data/redlightdata'

# load splits: 
split_path = '../data/hw02_splits'
file_names_train = np.load(os.path.join(split_path,'file_names_train.npy'))
file_names_test = np.load(os.path.join(split_path,'file_names_test.npy'))

# set a path for saving predictions:
preds_path = '../data/hw02_preds'
os.makedirs(preds_path, exist_ok=True) # create directory if needed

# Set this parameter to True when you're done with algorithm development:
done_tweaking = True

'''
Make predictions on the training set.
'''
preds_train = {}
for i in tqdm(range(len(file_names_train))):

    img = Image.open(os.path.join(data_path, file_names_train[i]))
#     # scale = 1
#     # scale_factor = 1/0.5
#
#     # convert to numpy array:
    I = np.asarray(img)
    preds = detect_red_light_mf(I)
#     # preds = [[r[0] // scale,
#     #           r[1] // scale,
#     #           r[2] // scale,
#     #           r[3] // scale,
#     #           r[4]] for r in preds]
#
    if file_names_train[i] in preds_train.keys():
        for p in preds:
            preds_train[file_names_train[i]].append(p)
    else:
        preds_train[file_names_train[i]] = preds

    # img = img.resize((int(img.width // scale_factor), int(img.height // scale_factor)), Image.BICUBIC)
    # scale *= scale_factor
    # print(preds_train[file_names_train[i]])


# save preds (overwrites any previous predictions!)
with open(os.path.join(preds_path,'weak_preds_train.json'),'w') as f:
    json.dump(preds_train,f)

if done_tweaking:
    '''
    Make predictions on the test set.
    '''
    preds_test = {}
    for i in tqdm(range(len(file_names_test))):

        # read image using PIL:
        I = Image.open(os.path.join(data_path,file_names_test[i]))

        # convert to numpy array:
        I = np.asarray(I)

        preds_test[file_names_test[i]] = detect_red_light_mf(I)

    # save preds (overwrites any previous predictions!)
    with open(os.path.join(preds_path,'weak_preds_test.json'),'w') as f:
        json.dump(preds_test,f)
