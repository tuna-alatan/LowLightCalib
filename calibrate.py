import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2

def calculate_patch_means(input_img):
    img = input_img.astype(np.float32) / 255.0 # Write normalized values to CSV
    #img = input_img

    means = []
    patch_centers = []
    patch_coords = []

    # Helper to add patch info
    def add_patch(x, y, w, h):
        rMean = np.mean(img[y:y+h, x:x+w, 2])
        gMean = np.mean(img[y:y+h, x:x+w, 1])
        bMean = np.mean(img[y:y+h, x:x+w, 0])
        means.append((rMean, gMean, bMean))
        cx = x + w // 2
        cy = y + h // 2
        patch_centers.append((cx, cy))
        patch_coords.append((x, y, w, h))

    # First row
    add_patch(212, 786, 43, 29)
    add_patch(240, 604, 45, 33)
    add_patch(270, 426, 44, 31)
    add_patch(290, 267, 46, 29)
    # Second row
    add_patch(460, 801, 42, 29)
    add_patch(486, 600, 52, 33)
    add_patch(511, 431, 52, 36)
    add_patch(521, 260, 51, 33)
    # Third row
    add_patch(726, 810, 49, 36)
    add_patch(713, 611, 53, 37)
    add_patch(741, 422, 54, 39)
    add_patch(769, 270, 54, 40)
    # Fourth row
    add_patch(970, 800, 52, 43)
    add_patch(980, 600, 55, 45)
    add_patch(1000, 440, 56, 46)
    add_patch(990, 280, 58, 46)
    # Fifth row
    add_patch(1230, 800, 53, 45)
    add_patch(1220, 620, 61, 50)
    add_patch(1220, 440, 60, 53)
    add_patch(1220, 285, 64, 52)
    # Sixth row
    add_patch(1500, 820, 54, 54)
    add_patch(1500, 610, 61, 56)
    add_patch(1470, 450, 64, 56)
    add_patch(1450, 290, 63, 57)

    return means, patch_centers


def computeCCM(XYZ_values, RGB_values, ground_truth_RGB_values,white_point):
    d65 = np.array([0.9504, 1, 1.0888], dtype=np.float32)

    XYZ_to_sRGB =  np.array([[3.2404542, -1.5371385, -0.4985314],
                             [-0.9692660, 1.8760108, 0.0415560],
                             [0.0556434, -0.2040259, 1.0572252]], dtype=np.float32)
    
    M_CA = np.diag(d65 / white_point)

    XYZ_vector = XYZ_values.reshape(-1, 1)
    RGB_vector = ground_truth_RGB_values.reshape(-1, 1)

    A = np.zeros((72, 9), dtype=np.float32)

    k = 0
    for i in range(24):
        A[k, 0:3] = RGB_values[i, :]
        A[k + 1, 3:6] = RGB_values[i, :]
        A[k + 2, 6:9] = RGB_values[i, :]
        k += 3
    
    CAM_to_XYZ = np.linalg.lstsq(A, XYZ_vector, rcond=None)[0].reshape(3, 3)
    CAM_to_sRGB = XYZ_to_sRGB @ M_CA @ CAM_to_XYZ
    #CAM_to_sRGB = np.linalg.lstsq(A, RGB_vector, rcond=None)[0].reshape(3, 3)

    return CAM_to_sRGB
    

def calibrate(input_img):
    # Assumes the input image to be in BGR order
    # Returns the result in BGR order as well
    XYZ_values = np.loadtxt('measurement_results.csv', delimiter=',').astype(np.float32)
    ground_truth_RGB_values = np.loadtxt('ground_truth_srgb.csv', delimiter=',').astype(np.float32)
    RGB_values = np.loadtxt('image_data.csv', delimiter=',').astype(np.float32)

    white_point = np.array([1, 1, 1], dtype=np.float32)
    CAM_to_sRGB = computeCCM(XYZ_values, RGB_values, ground_truth_RGB_values, white_point)

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
    img = simple_linear_to_srgb(img)  # Convert to sRGB

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = (img * 255).astype(np.uint8)

    return img
   
def srgb_to_linear(rgb):
    rgb = np.array(rgb)
    mask = rgb <= 0.04045
    linear = np.where(mask, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    return linear

def simple_srgb_to_linear(rgb):
    linear = np.power(rgb, 2.2)
    return linear

def simple_linear_to_srgb(rgb):
    srgb = np.power(rgb, 1/2.2)
    return srgb

def linear_to_srgb(rgb):
    rgb = np.clip(rgb, 0, 1)
    mask = rgb <= 0.0031308
    srgb = np.where(mask, 12.92 * rgb, 1.055 * (rgb ** (1/2.4)) - 0.055)
    return srgb


if __name__ == "__main__":
    # Use a TIFF image as input instead of a raw .bin file
    tiff_img_path = "./input_images/16mm_f1_4_exp1_15_gain25.tiff"  # Change this path as needed

    input_img = cv2.imread(tiff_img_path, cv2.IMREAD_COLOR)
    if input_img is None or input_img.size == 0:
        raise FileNotFoundError(f"Could not read TIFF image at {tiff_img_path}")

    # If the image is grayscale, convert to 3-channel
    if len(input_img.shape) == 2:
        input_img = np.stack([input_img]*3, axis=-1)

    linear_img = simple_srgb_to_linear(input_img.astype(np.float32) / 255.0) * 255.0
    means, patch_centers = calculate_patch_means(linear_img)
    print("Means: ", means)
    np.savetxt("image_data.csv", means, delimiter=",")

    finalResult = calibrate(linear_img)

    # Show input and output images side by side using matplotlib
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    # Draw squares on input image
    input_img_disp = cv2.cvtColor(input_img.copy(), cv2.COLOR_BGR2RGB)
    for cx, cy in patch_centers:
        x0 = int(cx - 20)
        y0 = int(cy - 20)
        x1 = int(cx + 20)
        y1 = int(cy + 20)
        input_img_disp = cv2.rectangle(input_img_disp, (x0, y0), (x1, y1), (0, 255, 0), 2)
    axes[0].imshow(input_img_disp)
    axes[0].set_title('Input Image')
    axes[0].axis('off')
    axes[1].imshow(cv2.cvtColor(finalResult, cv2.COLOR_BGR2RGB))
    axes[1].set_title('Output Image')
    axes[1].axis('off')
    plt.tight_layout()
    plt.savefig('comparison.png')  # Save the comparison image
    plt.show()

    cv2.imwrite('final_result.png', finalResult)
    print("HALT")
