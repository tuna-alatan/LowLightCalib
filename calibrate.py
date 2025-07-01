import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import csv

def calculate_patch_means(input_img):
    img = input_img.astype(np.float32) / 255.0 # Write normalized values to CSV

    means = []

    # First row
    x = 212
    y = 786
    w = 43
    h = 29
    rMean = np.mean(img[y:y+h, x:x+w, 2])
    gMean = np.mean(img[y:y+h, x:x+w, 1])
    bMean = np.mean(img[y:y+h, x:x+w, 0])
    means.append((rMean, gMean, bMean))

    x = 240
    y = 604
    w = 45
    h = 33
    rMean = np.mean(img[y:y+h, x:x+w, 2])
    gMean = np.mean(img[y:y+h, x:x+w, 1])
    bMean = np.mean(img[y:y+h, x:x+w, 0])
    means.append((rMean, gMean, bMean))

    x = 270
    y = 426
    w = 44
    h = 31
    rMean = np.mean(img[y:y+h, x:x+w, 2])
    gMean = np.mean(img[y:y+h, x:x+w, 1])
    bMean = np.mean(img[y:y+h, x:x+w, 0])
    means.append((rMean, gMean, bMean))

    x = 290
    y = 267
    w = 46
    h = 29
    rMean = np.mean(img[y:y+h, x:x+w, 2])
    gMean = np.mean(img[y:y+h, x:x+w, 1])
    bMean = np.mean(img[y:y+h, x:x+w, 0])
    means.append((rMean, gMean, bMean))
    
    # Second row
    x = 460
    y = 801
    w = 42
    h = 29
    rMean = np.mean(img[y:y+h, x:x+w, 2])
    gMean = np.mean(img[y:y+h, x:x+w, 1])
    bMean = np.mean(img[y:y+h, x:x+w, 0])
    means.append((rMean, gMean, bMean))

    x = 486
    y = 600
    w = 52
    h = 33
    rMean = np.mean(img[y:y+h, x:x+w, 2])
    gMean = np.mean(img[y:y+h, x:x+w, 1])
    bMean = np.mean(img[y:y+h, x:x+w, 0])
    means.append((rMean, gMean, bMean))

    x = 511
    y = 431
    w = 52
    h = 36
    rMean = np.mean(img[y:y+h, x:x+w, 2])
    gMean = np.mean(img[y:y+h, x:x+w, 1])
    bMean = np.mean(img[y:y+h, x:x+w, 0])
    means.append((rMean, gMean, bMean))

    x = 521
    y = 260
    w = 51
    h = 33
    rMean = np.mean(img[y:y+h, x:x+w, 2])
    gMean = np.mean(img[y:y+h, x:x+w, 1])
    bMean = np.mean(img[y:y+h, x:x+w, 0])
    means.append((rMean, gMean, bMean))

    # Third row
    x = 726
    y = 810
    w = 49
    h = 36
    rMean = np.mean(img[y:y+h, x:x+w, 2])
    gMean = np.mean(img[y:y+h, x:x+w, 1])
    bMean = np.mean(img[y:y+h, x:x+w, 0])
    means.append((rMean, gMean, bMean))

    x = 713
    y = 611
    w = 53
    h = 37
    rMean = np.mean(img[y:y+h, x:x+w, 2])
    gMean = np.mean(img[y:y+h, x:x+w, 1])
    bMean = np.mean(img[y:y+h, x:x+w, 0])
    means.append((rMean, gMean, bMean))

    x = 741
    y = 422
    w = 54
    h = 39
    rMean = np.mean(img[y:y+h, x:x+w, 2])
    gMean = np.mean(img[y:y+h, x:x+w, 1])
    bMean = np.mean(img[y:y+h, x:x+w, 0])
    means.append((rMean, gMean, bMean))

    x = 769
    y = 270
    w = 54
    h = 40
    rMean = np.mean(img[y:y+h, x:x+w, 2])
    gMean = np.mean(img[y:y+h, x:x+w, 1])
    bMean = np.mean(img[y:y+h, x:x+w, 0])
    means.append((rMean, gMean, bMean))

    # Fourth row	
    x = 970
    y = 800
    w = 52
    h = 43
    rMean = np.mean(img[y:y+h, x:x+w, 2])
    gMean = np.mean(img[y:y+h, x:x+w, 1])
    bMean = np.mean(img[y:y+h, x:x+w, 0])
    means.append((rMean, gMean, bMean))

    x = 980
    y = 600
    w = 55
    h = 45
    rMean = np.mean(img[y:y+h, x:x+w, 2])
    gMean = np.mean(img[y:y+h, x:x+w, 1])
    bMean = np.mean(img[y:y+h, x:x+w, 0])
    means.append((rMean, gMean, bMean))

    x = 1000
    y = 440
    w = 56
    h = 46
    rMean = np.mean(img[y:y+h, x:x+w, 2])
    gMean = np.mean(img[y:y+h, x:x+w, 1])
    bMean = np.mean(img[y:y+h, x:x+w, 0])
    means.append((rMean, gMean, bMean))

    x = 990
    y = 280
    w = 58
    h = 46
    rMean = np.mean(img[y:y+h, x:x+w, 2])
    gMean = np.mean(img[y:y+h, x:x+w, 1])
    bMean = np.mean(img[y:y+h, x:x+w, 0])
    means.append((rMean, gMean, bMean))

    # Fifth row
    x = 1230
    y = 800
    w = 53
    h = 45
    rMean = np.mean(img[y:y+h, x:x+w, 2])
    gMean = np.mean(img[y:y+h, x:x+w, 1])
    bMean = np.mean(img[y:y+h, x:x+w, 0])
    means.append((rMean, gMean, bMean))

    x = 1220
    y = 620
    w = 61
    h = 50
    rMean = np.mean(img[y:y+h, x:x+w, 2])
    gMean = np.mean(img[y:y+h, x:x+w, 1])
    bMean = np.mean(img[y:y+h, x:x+w, 0])
    means.append((rMean, gMean, bMean))

    x = 1220
    y = 440
    w = 60
    h = 53
    rMean = np.mean(img[y:y+h, x:x+w, 2])
    gMean = np.mean(img[y:y+h, x:x+w, 1])
    bMean = np.mean(img[y:y+h, x:x+w, 0])
    means.append((rMean, gMean, bMean))

    x = 1220
    y = 285
    w = 64
    h = 52
    rMean = np.mean(img[y:y+h, x:x+w, 2])
    gMean = np.mean(img[y:y+h, x:x+w, 1])
    bMean = np.mean(img[y:y+h, x:x+w, 0])
    means.append((rMean, gMean, bMean))

    # Sixth row
    x = 1500
    y = 820
    w = 54
    h = 54
    rMean = np.mean(img[y:y+h, x:x+w, 2])
    gMean = np.mean(img[y:y+h, x:x+w, 1])
    bMean = np.mean(img[y:y+h, x:x+w, 0])
    means.append((rMean, gMean, bMean))

    x = 1500
    y = 610
    w = 61
    h = 56
    rMean = np.mean(img[y:y+h, x:x+w, 2])
    gMean = np.mean(img[y:y+h, x:x+w, 1])
    bMean = np.mean(img[y:y+h, x:x+w, 0])
    means.append((rMean, gMean, bMean))

    x = 1470
    y = 450
    w = 64
    h = 56
    rMean = np.mean(img[y:y+h, x:x+w, 2])
    gMean = np.mean(img[y:y+h, x:x+w, 1])
    bMean = np.mean(img[y:y+h, x:x+w, 0])
    means.append((rMean, gMean, bMean))

    x = 1450
    y = 290
    w = 63
    h = 57
    rMean = np.mean(img[y:y+h, x:x+w, 2])
    gMean = np.mean(img[y:y+h, x:x+w, 1])
    bMean = np.mean(img[y:y+h, x:x+w, 0])
    means.append((rMean, gMean, bMean))
   
    return means


def computeCCM(XYZ_values, RGB_values, white_point):
    d65 = np.array([0.9504, 1, 1.0888], dtype=np.float32)

    XYZ_to_sRGB =  np.array([[3.2404542, -1.5371385, -0.4985314],
                             [-0.9692660, 1.8760108, 0.0415560],
                             [0.0556434, -0.2040259, 1.0572252]], dtype=np.float32)
    
    M_CA = np.diag(d65 / white_point)

    XYZ_vector = XYZ_values.reshape(-1, 1)

    A = np.zeros((72, 9), dtype=np.float32)

    k = 0
    for i in range(24):
        A[k, 0:3] = RGB_values[i, :]
        A[k + 1, 3:6] = RGB_values[i, :]
        A[k + 2, 6:9] = RGB_values[i, :]
        k += 3
    
    CAM_to_XYZ = np.linalg.lstsq(A, XYZ_vector, rcond=None)[0].reshape(3, 3)
    CAM_to_sRGB = XYZ_to_sRGB @ M_CA @ CAM_to_XYZ

    return CAM_to_sRGB
    

def calibrate(input_img):
    # Assumes the input image to be in BGR order
    # Returns the result in BGR order as well
    XYZ_values = np.loadtxt('measurement_results.csv', delimiter=',').astype(np.float32)
    RGB_values = np.loadtxt('image_data.csv', delimiter=',').astype(np.float32)

    white_point = np.array([1, 1, 1], dtype=np.float32)
    CAM_to_sRGB = computeCCM(XYZ_values, RGB_values, white_point)

    img = input_img.astype(np.float32) / 255.0 # Use the [0, 1] range
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    orig_shape = img.shape
    img = img.reshape(-1, 3)
    
    # Color correction happens here
    img = img @ CAM_to_sRGB.T
    img /= 60 # Some kind of digital gain
    img = np.clip(img, 0, 1)

    img = img.reshape(orig_shape)

    # Gamma correction
    #img = linear_to_srgb(img)  # Convert to sRGB

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = (img * 255).astype(np.uint8)

    return img
   
def srgb_to_linear(rgb):
    rgb = np.array(rgb)
    mask = rgb <= 0.04045
    linear = np.where(mask, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    return linear

def linear_to_srgb(rgb):
    rgb = np.clip(rgb, 0, 1)
    mask = rgb <= 0.0031308
    srgb = np.where(mask, 12.92 * rgb, 1.055 * (rgb ** (1/2.4)) - 0.055)
    return srgb


if __name__ == "__main__":

    # Use a TIFF image as input instead of a raw .bin file
    tiff_img_path = "./input_images/16mm_f1_4_exp1_15_gain25.tiff"  # Change this path as needed


    input_img = cv2.imread(tiff_img_path, cv2.IMREAD_COLOR_BGR)
    if input_img is None or input_img.size == 0:
        raise FileNotFoundError(f"Could not read TIFF image at {tiff_img_path}")

    # If the image is grayscale, convert to 3-channel
    if len(input_img.shape) == 2:
        input_img = np.stack([input_img]*3, axis=-1)

    #linear_input_img = srgb_to_linear(input_img / 255.0)  # Convert to linear RGB in [0, 1] range
    means = calculate_patch_means(input_img)
    print("Means: ", means)
    np.savetxt("image_data.csv", means, delimiter=",")

    finalResult = calibrate(input_img)

    cv2.imwrite('final_result.png', finalResult)
    cv2.imshow('Final result', finalResult)
    cv2.waitKey(0)
    print("HALT")
