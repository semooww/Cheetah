
import cv2
import numpy as np

# Cropping image to avoid from unnecessary black areas
def deleteBlackAreas(filename):
    IMAGE_WIDTH = 224
    IMAGE_HEIGHT = 224
    image_size = (IMAGE_WIDTH, IMAGE_HEIGHT)
    img = cv2.imread(filename)  # read image from file
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert to grayscale
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)  # turn it into a binary image
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # find contours
    if len(contours) != 0:
        # find the biggest area
        cnt = max(contours, key=cv2.contourArea)

        # find the bounding rect
        x, y, w, h = cv2.boundingRect(cnt)

        crop = img[y:y + h, x:x + w]  # crop image
        crop1 = cv2.cvtColor(cv2.resize(crop, image_size, interpolation=cv2.INTER_AREA),
                             cv2.COLOR_BGR2RGB)  # resize to image_size and change color space from BGR to RGB for matplotlib
        return crop1
    else:
        return cv2.cvtColor(cv2.resize(img, image_size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)



#color_normalization of images
def color_normalization(img):
    image_copy = img.copy()
    for i in range(3):
        imi = img[:, :, i]
        minval = np.min(imi)
        maxval = np.max(imi)
        imrange = maxval - minval
        # imi-minval will turn the color range between 0-imrange, and the scaling will stretch the range between 0-255
        image_copy[:, :, i] = (255 / (imrange + 0.0001) * (imi - minval))
    return image_copy

def canny_edge(img):
    t_lower = 20
    t_upper = 120
    edges = cv2.Canny(img, t_lower, t_upper, apertureSize=3, L2gradient=True)
    return edges

def convertToGray(img):
    img_copy = img.copy()
    img_copy = cv2.cvtColor(img_copy, cv2.COLOR_RGB2GRAY)
    return img_copy

def hist_equalization(img):
    array = np.asarray(img)
    bin_cont = np.bincount(array.flatten(), minlength=256)
    pixels = np.sum(bin_cont)
    bin_cont = bin_cont / pixels
    cumulative_sumhist = np.cumsum(bin_cont)
    map = np.floor(255 * cumulative_sumhist).astype(np.uint8)
    arr_list = list(array.flatten())
    eq_arr = [map[p] for p in arr_list]
    arr_back = np.reshape(np.asarray(eq_arr), array.shape)
    return arr_back


def ahe(img, rx=136, ry=185):
    img_eq = np.empty((img.shape[0], img.shape[1]), dtype=np.uint8)
    for i in range(0, img.shape[1], rx):
        for j in range(0, img.shape[0], ry):
            t = img[j:j + ry, i:i + rx]
            c = hist_equalization(t)
            img_eq[j:j + ry, i:i + rx] = c
    return img_eq

def convertColorSpace2XYZ(img):
    img_copy = cv2.cvtColor(img, cv2.COLOR_RGB2XYZ)
    return img_copy

def convertColorSpace2HSV(img):
    img_copy = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    return img_copy

def binarization(img):
    img_copy = img.copy()
    img_copy = cv2.cvtColor(img_copy, cv2.COLOR_RGB2GRAY)
    img_copy = cv2.adaptiveThreshold(img_copy, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 3, 2)
    return img_copy