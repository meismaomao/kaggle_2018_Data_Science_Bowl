import numpy as np
from scipy import misc
from skimage.morphology import label
import tensorflow as tf

def random_rotate_image(image,mask):
    angle = np.random.uniform(low=-10.0, high=10.0)
    return misc.imrotate(image, angle, 'bicubic'), misc.imrotate(mask, angle, 'bicubic')

def flip(image, mask):
    if np.random.choice([True, False]):
        image = np.fliplr(image)
        mask = np.fliplr(mask)
    return image, mask

def rle_encoding(x):
    '''
    x: numpy array of shape (height, width), 1 - mask, 0 - background
    Returns run length as list
    '''
    dots = np.where(x.T.flatten()==1)[0] # .T sets Fortran order down-then-right
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b+1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

def prob_to_rles(x, cut_off = 0.5):
    lab_img = label(x>cut_off)
    if lab_img.max()<1:
        lab_img[0,0] = 1 # ensure at least one prediction per image
    for i in range(1, lab_img.max()+1):
        yield rle_encoding(lab_img==i)

def check_rle(ground_truth, fake):
    train_row_rles = prob_to_rles(ground_truth)
    tl_rles = prob_to_rles(fake)

    match, mismatch = 0, 0
    for img_rle, train_rle in zip(sorted(train_row_rles, key=lambda x: x[0]),
                                  sorted(tl_rles, key=lambda x: x[0])):

        for i_x, i_y in zip(img_rle, train_rle):
            if i_x == i_y:
                match += 1
            else:
                mismatch += 1
    print('Matches: %d, Mismatches: %d, Accuracy: %2.1f%%' %
          (match, mismatch, 100.0 * match / (match + mismatch)))


def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        prec.append(score)
    return tf.reduce_mean(prec, axis=0)