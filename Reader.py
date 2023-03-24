import pandas as pd
import os
from sklearn.model_selection import train_test_split
import imgaug as ia
import imgaug.augmenters as iaa
import numpy as np
import Techniques as T
import tqdm
import cv2


def create_dataset(df, type=0, mode=0):  # 0->float32 1->uint8
    # Creating dataset
    images = []
    labels = []
    index = 0
    for path in df['paths']:
        # According to parameter, we apply some preprocesses here. default=0
        img = T.deleteBlackAreas(path)  # deleting black areas. Initial preprocess
        if mode == 1:
            img = T.color_normalization(img)
        elif mode == 2:
            img = T.canny_edge(img)
        elif mode == 3:
            img = T.convertToGray(img)
        elif mode == 4:
            img = T.convertColorSpace2XYZ(img)
        elif mode == 5:
            img = T.convertColorSpace2HSV(img)
        elif mode == 6:
            img = T.binarization(img)
        label = [0, 0, 0, 0]
        label[df.iloc[index]["class_label"]] += 1
        index += 1
        images.append(img)
        labels.append(label)
    if type == 0:
        images = np.array(images, dtype='float32') / 255
    elif type == 1:
        images = np.array(images, dtype='uint8')
    labels = np.array(labels)
    return augmentation(images, labels)


def augmentation(images, labels):
    ia.seed(42)

    contrast = iaa.Sequential([
        iaa.Sometimes(
            0.3,
            iaa.GaussianBlur(sigma=(0, 0.25))
        ),
        iaa.LinearContrast((0.75, 1.5)),
    ])

    mix = iaa.Sequential([
        iaa.Sometimes(
            0.5,
            iaa.GaussianBlur(sigma=(0, 0.25))
        ),
        iaa.Sometimes(
            0.5,
            iaa.LinearContrast((0.75, 1.5)),
        ),
        iaa.Affine(
            scale={"x": (0.8, 1), "y": (0.8, 1)},
            # translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-20, 20),
            # shear=(-2, 2)
        )
    ], random_order=True)  # apply augmenters in random order

    augmentation_dict = {0: contrast, 1: mix}
    images_result = None
    labels_result = None
    seeds = [6, 37]
    for i in range(len(augmentation_dict)):
        func = augmentation_dict[i]
        if i == 0:
            images_augmented = func(images=images)
            images_result = np.concatenate((images, images_augmented))
            labels_result = np.concatenate((labels, labels))
        elif i == 1:
            for k in range(1):
                ia.seed(seeds[k])
                images_augmented = func(images=images)
                images_result = np.concatenate((images_result, images_augmented))
                labels_result = np.concatenate((labels_result, labels))
    return images_result, labels_result


def read_dataset(mode=0):
    DATASET_PATH = r"dataset/"
    SEED = 42
    dataset_folders = []  # to keep main folder names
    total = 0
    # print(f"There are {len(os.listdir(DATASET_PATH))} folder in dataset.")
    for path in sorted(os.listdir(DATASET_PATH)):
        # print(f"\t-There are {len(os.listdir(DATASET_PATH + path))} images in {path} folder.")
        total += len(os.listdir(DATASET_PATH + path))  # add element size of the current folder to total variable
        dataset_folders.append(DATASET_PATH + path)  # add current folder path to dataset_folders

    # Create an empty dataframe
    df = pd.DataFrame(0,
                      columns=['paths',
                               'class_label'],
                      index=range(total))
    # store each image path in the dataframe
    # class labels -> 0:Normal 1:Cataract 2:Glaucoma 3:RetinaDisease
    i = 0
    for p, path in enumerate(dataset_folders):  # main folders
        for sub_path in sorted(os.listdir(path)):  # images
            df.iloc[i, 0] = path + "/" + sub_path
            df.iloc[i, 1] = p
            i += 1
    # Display some examples for the created DataFrame
    # print(df.sample(frac=1, random_state=SEED).head(10))
    train_df, test_df = train_test_split(df,
                                         test_size=0.2,
                                         random_state=SEED,
                                         stratify=df['class_label'])

    # Creating dataset and split the data
    X_train, y_train = create_dataset(train_df, mode=mode)
    X_test, y_test = create_dataset(test_df, mode=mode)

    return X_train, y_train, X_test, y_test
