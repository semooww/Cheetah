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


# color_normalization of images
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


def convertToGray(img):
    img_copy = img.copy()
    img_copy = cv2.cvtColor(img_copy, cv2.COLOR_RGB2GRAY)
    return img_copy


def convertColorSpace2XYZ(img):
    img_copy = cv2.cvtColor(img, cv2.COLOR_RGB2XYZ)
    return img_copy


def binarization(img):
    img_copy = img.copy()
    img_copy = cv2.cvtColor(img_copy, cv2.COLOR_RGB2GRAY)
    img_copy = cv2.adaptiveThreshold(img_copy, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 3, 2)
    return img_copy


def mahal(img, select=None, mean_pix=None):
    arr = np.reshape(img, (img.shape[0] * img.shape[1], 3))
    # no sampling.  use the entire image
    if select is None:
        select = arr
    else:
        # if 'select' is a number, generate an array of size 'select' containing
        # random pixels in 'arr'.
        # otherwise it should be a list of indices of pixels to choose.
        select = arr[np.random.randint(0, arr.shape[0], select), :] if isinstance(select, int) else arr[select]

    # calculate the covariance matrix inverse using the sampled array
    invcovar = np.linalg.inv(np.cov(np.transpose(select)))

    if mean_pix is None:
        # no provided mean RGB vector.  assume we are using the images own
        # mean RGB value
        meandiff = arr - np.mean(select, axis=0)
    else:
        meandiff = arr - mean_pix

    # calculate the difference between every pixel in 'arr' and the mean RGB vector.
    # if provided, use the given mean RGB vector, otherwise calculate the mean RGB
    # value of 'select'
    meandiff = arr - (mean_pix if mean_pix is not None else np.mean(select, axis=0))

    # calculate the first multiplication.
    output = np.dot(meandiff, invcovar)

    # do literally everything else all in this step, then reshape back to image dimensions and return
    return np.sqrt(np.einsum('ij,ij->i', output, meandiff)).reshape(img.shape[:-1])

def CLAHE(img):
    img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=5)
    img = clahe.apply(img)
    return img

def sharpening_image(img):
    # create a sharpening kernel
    sharpen_filter = np.array([[-1, -1, -1],
                               [-1, 9, -1],
                               [-1, -1, -1]])
    # applying kernels to the input image to get the sharpened image
    sharp_image = cv2.filter2D(img, -1, sharpen_filter)
    return sharp_image