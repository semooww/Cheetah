import pandas as pd
import os
from sklearn.model_selection import train_test_split
import imgaug as ia
import imgaug.augmenters as iaa
import numpy as np
import Techniques as T
import tqdm
import matplotlib.pyplot as plt
import cv2


def create_dataset(df, mode=0, type=0, is_Test=0):  # 0->float32 1->uint8
    # Creating dataset
    images = []
    labels = []
    index = 0
    for path in df['paths']:
        if "3_retina_disease" in path:
            index += 1
            continue
        # According to parameter, we apply some preprocesses here. default=0
        img = T.deleteBlackAreas(path)  # deleting black areas. Initial preprocess
        if mode == 1:
            img = T.color_normalization(img)
        elif mode == 2:
            img = T.CLAHE(img)
            img = cv2.merge((img, img, img))
        elif mode == 3:
            img = T.convertToGray(img)
            img = cv2.merge((img, img, img))
        elif mode == 4:
            img = T.convertColorSpace2XYZ(img)
        label = [0, 0, 0]
        label[df.iloc[index]["class_label"]] += 1
        index += 1
        images.append(img)
        labels.append(label)
        if not "1_normal" in path:
            img2 = cv2.flip(img, 1)
            images.append(img2)
            labels.append(label)
            img3 = T.sharpening_image(img)
            images.append(img3)
            labels.append(label)
    if type == 0:
        images = np.array(images, dtype='float32') / 255
    elif type == 1:
        images = np.array(images, dtype='uint8')
    labels = np.array(labels)
    return augmentation(images, labels, mode, is_Test)


def augmentation(images, labels, mode, is_Test):
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
            iaa.AdditiveGaussianNoise(scale=(0, 0.2 * 255))
        ),
        iaa.Sometimes(
            0.5,
            iaa.pillike.EnhanceSharpness()
        ),
        iaa.Sometimes(
            0.5,
            iaa.LinearContrast((0.75, 1.5)),
        ),
        iaa.Affine(
            scale={"x": (0.7, 1), "y": (0.7, 1)},
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
            for k in range(2):
                ia.seed(seeds[k])
                images_augmented = func(images=images)
                images_result = np.concatenate((images_result, images_augmented))
                labels_result = np.concatenate((labels_result, labels))
    save_images(mode, images_result, labels_result, is_Test)
    return images_result, labels_result


def read_dataset(mode=0, type=0):
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
    X_train, y_train = create_dataset(train_df, mode=mode, type=type)
    X_test, y_test = create_dataset(test_df, mode=mode, type=type, is_Test=1)

    return X_train, y_train, X_test, y_test


def save_images(mode, images, labels, is_Test):
    train_or_test = ""
    if not is_Test:
        train_or_test = "train"
    else:
        train_or_test = "test"
    techniques = ["Default", "Color Normalization", "CLAHE", "Gray Scale", "XYZ"]
    MAIN_PATH = r"Processes"
    os.makedirs(MAIN_PATH, exist_ok=True)
    MODE_PATH = os.path.join(MAIN_PATH, techniques[mode])
    os.makedirs(MODE_PATH, exist_ok=True)
    TT_PATH = os.path.join(MODE_PATH, train_or_test)
    os.makedirs(TT_PATH, exist_ok=True)
    label_dict = {0: "1_normal", 1: "2_cataract", 2: "2_glaucoma", 3: "3_retina_disease"}
    for i in range(4):
        os.makedirs(os.path.join(TT_PATH, label_dict[i]), exist_ok=True)
    label_index = np.argmax(labels, axis=1)
    counter = [0, 0, 0, 0]
    for i in range(len(images)):
        class_label = label_index[i]
        CLASS_PATH = os.path.join(TT_PATH, label_dict[class_label])
        current_counter = counter[class_label]
        length = len(str(current_counter))
        last = (4 - length) * "0" + str(current_counter) + ".png"
        counter[class_label] += 1
        SAVE_PATH = os.path.join(CLASS_PATH, last)
        # cv2.imwrite(SAVE_PATH,images[i])
        plt.imsave(SAVE_PATH, images[i])
    print(f"{train_or_test} set : {counter} -> {np.sum(counter)}")
