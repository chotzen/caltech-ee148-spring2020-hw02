import os
import json
import numpy as np
import matplotlib.pyplot as plt

def compute_iou(box_1, box_2):
    '''
    This function takes a pair of bounding boxes and returns intersection-over-
    union (IoU) of two bounding boxes.
    '''

    x1 = max(box_1[0], box_2[0])
    y1 = max(box_1[1], box_2[1])
    x2 = min(box_1[2], box_2[2])
    y2 = min(box_1[3], box_2[3])

    if x1 < x2 and y1 < y2:
        inter = (x2 - x1) * (y2 - y1)
    else:
        inter = 0

    union = area(box_1) + area(box_2) - inter

    iou = inter / union

    assert (iou >= 0) and (iou <= 1.0)

    return iou

def area(box):
    return (box[2] - box[0]) * (box[3] - box[1])



def compute_counts(preds, gts, iou_thr=0.5, conf_thr=0.5):
    '''
    This function takes a pair of dictionaries (with our JSON format; see ex.) 
    corresponding to predicted and ground truth bounding boxes for a collection
    of images and returns the number of true positives, false positives, and
    false negatives. 
    <preds> is a dictionary containing predicted bounding boxes and confidence
    scores for a collection of images.
    <gts> is a dictionary containing ground truth bounding boxes for a
    collection of images.
    '''
    TP = 0
    FP = 0
    FN = 0

    for k, (pred_file, pred) in enumerate(preds.items()):
        gt = gts[pred_file]
        for i in range(len(gt)):
            detected = False
            for j in range(len(pred)):
                matched = False
                iou = compute_iou(pred[j][:4], gt[i])
                conf = pred[j][4]

                if iou > iou_thr and conf > conf_thr:
                    TP += 1
                    detected = True
                    matched = True

                if conf > conf_thr and not matched:
                    FP += 1

            if not detected:
                FN += 1

    return TP, FP, FN

# set a path for predictions and annotations:
preds_path = '../data/hw02_preds'
gts_path = '../data/hw02_annotations'

# load splits:
split_path = '../data/hw02_splits'
file_names_train = np.load(os.path.join(split_path, 'file_names_train.npy'))
file_names_test = np.load(os.path.join(split_path, 'file_names_test.npy'))

# Set this parameter to True when you're done with algorithm development:
done_tweaking = False

'''
Load training data. 
'''
with open(os.path.join(preds_path,'preds_train.json'),'r') as f:
    preds_train = json.load(f)
    
with open(os.path.join(gts_path, 'annotations_train.json'),'r') as f:
    gts_train = json.load(f)

if done_tweaking:
    
    '''
    Load test data.
    '''

    with open(os.path.join(preds_path,'preds_test.json'),'r') as f:
        preds_test = json.load(f)
        
    with open(os.path.join(gts_path, 'annotations_test.json'),'r') as f:
        gts_test = json.load(f)


# For a fixed IoU threshold, vary the confidence thresholds.
# The code below gives an example on the training set for one IoU threshold.

def plot_pr(preds, gts):
    colors = ['teal', 'seagreen', 'darkgoldenrod', 'darkmagenta']
    fig, ax = plt.subplots()
    for c, iou in enumerate([0.25, 0.5, 0.75]):

        confidence_thrs = np.sort(np.array(list(set([pred[4] for fname in preds for pred in preds[fname]]))))  # using (ascending) list of confidence scores as thresholds

        tp = np.zeros(len(confidence_thrs))
        fp = np.zeros(len(confidence_thrs))
        fn = np.zeros(len(confidence_thrs))
        for i, conf_thr in enumerate(confidence_thrs):
            tp[i], fp[i], fn[i] = compute_counts(preds, gts, iou_thr=iou, conf_thr=conf_thr)

        n_preds = tp + fp
        n_objs = tp + fn

        precision = tp / n_preds
        recall = tp / n_objs

        plt.scatter(precision, recall, color=colors[c], label="iou = {}".format(iou))

    ax.legend()
    ax.grid(True)
    plt.title("Precision vs. Recall for Red Traffic Light Detection")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.show()


plot_pr(preds_train, gts_train)

if done_tweaking:
    plot_pr(preds_test, gts_test)
