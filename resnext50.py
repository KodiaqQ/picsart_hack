import keras.backend as K
from segmentation_models import Unet
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import argparse
import pandas as pd
from skimage.morphology import remove_small_objects
from sklearn.model_selection import StratifiedKFold

SEED = 42
smooth = 1e-10


def my_generator(x_train, y_train, batch_size):
    data_generator = ImageDataGenerator(
        width_shift_range=0.25,
        height_shift_range=0.25,
        zoom_range=0.25,
        horizontal_flip=True,
        rotation_range=30,
        rescale=1. / 255).flow(x_train, x_train, batch_size, seed=SEED)
    mask_generator = ImageDataGenerator(
        width_shift_range=0.25,
        height_shift_range=0.25,
        zoom_range=0.25,
        horizontal_flip=True,
        rotation_range=30,
        rescale=1. / 255).flow(y_train, y_train, batch_size, seed=SEED)
    while True:
        x_batch, _ = data_generator.next()
        y_batch, _ = mask_generator.next()
        yield x_batch, y_batch


def val_generator(x_train, y_train, batch_size=1):
    data_generator = ImageDataGenerator(
        rescale=1. / 255).flow(x_train, x_train, batch_size, seed=SEED)
    mask_generator = ImageDataGenerator(
        rescale=1. / 255).flow(y_train, y_train, batch_size, seed=SEED)
    while True:
        x_batch, _ = data_generator.next()
        y_batch, _ = mask_generator.next()
        yield x_batch, y_batch


def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def jaccard_coef(y_true, y_pred):
    # __author__ = Vladimir Iglovikov
    intersection = K.sum(y_true * y_pred, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])

    jac = (intersection + smooth) / (sum_ - intersection + smooth)

    return K.mean(jac)


def prepare_images(train_images_path):
    # get names of jpg files inside folder and create a list
    train_images = list(filter(lambda x: x.endswith('.jpg'), os.listdir(train_images_path)))

    # input data array
    x_data = np.empty((len(train_images), image_h, image_w, 3), dtype='uint8')
    for i, file_name in enumerate(train_images):
        img = cv2.imread(train_images_path + file_name, cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, dsize=(image_w, image_h))
        x_data[i] = img

    return x_data


def prepare_masks(train_masks_path):
    # get names of png files inside folder and create a list
    train_masks = list(filter(lambda x: x.endswith('.png'), os.listdir(train_masks_path)))

    # output data array
    y_data = np.empty((len(train_masks), image_h, image_w, 1), dtype='uint8')
    for i, file_name in enumerate(train_masks):
        img = cv2.imread(train_masks_path + file_name, cv2.IMREAD_UNCHANGED)
        img = cv2.resize(img, dsize=(image_w, image_h))
        img = img[:, :, np.newaxis]
        y_data[i] = img

    return y_data


def argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument('name', help='Name for model')
    args = ap.parse_args()
    return args


def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1. - score


def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    from: https://www.kaggle.com/kmader/baseline-u-net-model-part-1
    '''
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def save_history(history, j):
    plt.plot(history.history['dice_coef'])
    plt.plot(history.history['val_dice_coef'])
    plt.plot(history.history['jaccard_coef'])
    plt.plot(history.history['val_jaccard_coef'])
    plt.legend(['dice', 'val_dice', 'jaccard_coef', 'val_jaccard_coef'], loc='upper left')
    plt.ylabel('accuracy')
    plt.savefig('../graph/' + exargs.name + '_acc_cv' + str(j) + '.png')

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['loss', 'val_loss'], loc='upper left')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.savefig('../graph/' + exargs.name + '_loss_cv' + str(j) + '.png')


if __name__ == '__main__':
    exargs = argparser()
    train_images_path = '../data/train/'
    test_images_path = '../data/test/'
    train_masks_path = '../data/train_mask/'
    image_h = 288
    image_w = 288
    sample = pd.read_csv('../data/sample_submission.csv', index_col=['image'])

    x_data = prepare_images(train_images_path)
    test_data = prepare_images(test_images_path)
    y_data = prepare_masks(train_masks_path)

    x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.2, random_state=SEED)

    model = Unet(backbone_name='resnext50',
                 input_shape=(image_h, image_w, 3),
                 encoder_weights='imagenet',
                 decoder_block_type='transpose',
                 freeze_encoder=True,
                 activation='sigmoid')
    model.summary()

    callbacks_list = [ReduceLROnPlateau(monitor='val_loss', factor=0.25, patience=1, min_lr=1e-6)]

    # model.load_weights('../weights/resnet34_RLE_72_loss.h5')

    model.compile(optimizer=Adam(1e-4), loss=dice_loss, metrics=[dice_coef, jaccard_coef])

    save_name = '../weights/' + exargs.name  + '.h5'
    save_name_loss = '../weights/' + exargs.name  + '_loss.h5'
    callbacks_list.append(
        ModelCheckpoint(save_name_loss,
                        verbose=1,
                        monitor='loss',
                        save_best_only=True,
                        mode='min',
                        save_weights_only=True))
    callbacks_list.append(
        ModelCheckpoint(save_name,
                        verbose=1,
                        save_best_only=True,
                        mode='min',
                        save_weights_only=True))
    history = model.fit_generator(my_generator(x_train, y_train, 32),
                                  steps_per_epoch=len(x_train),
                                  validation_data=val_generator(x_val, y_val),
                                  validation_steps=len(x_val),
                                  epochs=10,
                                  verbose=1,
                                  shuffle=True,
                                  callbacks=callbacks_list)
    save_history(history, 1)

    test_image = model.predict(test_data[1].reshape(1, image_h, image_w, 3))
    test_image = test_image[0, :, :, 0]

    cv2.imshow('test_image', test_image)
    cv2.waitKey(0)

    images_list = os.listdir(test_images_path)
    for i, img in enumerate(images_list):
        image = cv2.imread(os.path.join(test_images_path, img), cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, dsize=(image_w, image_h))
        result = model.predict(image.reshape(1, image_h, image_w, 3))
        res = result[0, :, :, 0]
        # if image sise different from 240x320
        mask = cv2.resize(res, dsize=(240, 320))
        kernel = np.ones((5, 5), np.float32) / 25
        mask = cv2.blur(mask, (5, 5))
        # mask = cv2.GaussianBlur(mask,(3,3),0)
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
        # cv2.imshow('image', mask)
        # cv2.waitKey(0)
        # exit()
        mask = np.asarray(mask, bool)
        res = remove_small_objects(mask, min_size=256, connectivity=1)
        res = np.asarray(res, 'uint8')
        enc = rle_encode(res)
        print('Saving ' + str(i) + ' image..')
        sample.set_value(int(img.split('.')[0]), 'rle_mask', enc)

    sample.to_csv(exargs.name + '.csv')

    model_json = model.to_json()
    json_file = open('../models/' + exargs.name + '.json', 'w')
    json_file.write(model_json)
    json_file.close()
    print('Model saved!')


    K.clear_session()
