import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import colour

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
    add_patch(290, 267, 46, 29) # Dark Skin
    add_patch(521, 260, 51, 33) # Light Skin
    add_patch(769, 270, 54, 40) # Blue Sky
    add_patch(990, 280, 58, 46) # Foliage
    add_patch(1220, 285, 64, 52) # Blue Flower
    add_patch(1450, 290, 63, 57) # Bluish Green
    
    # Second row
    add_patch(270, 426, 44, 31) # Orange
    add_patch(511, 431, 52, 36) # Purplish Blue
    add_patch(741, 422, 54, 39) # Moderate Red
    add_patch(1000, 440, 56, 46) # Purple
    add_patch(1220, 440, 60, 53) # Yellow Green
    add_patch(1470, 450, 64, 56) # Orange Yellow
    
    # Third row
    add_patch(240, 604, 45, 33) # Blue
    add_patch(486, 600, 52, 33) # Green
    add_patch(713, 611, 53, 37) # Red
    add_patch(980, 600, 55, 45) # Yellow 
    add_patch(1220, 620, 61, 50) # Magenta
    add_patch(1500, 610, 61, 56) # Cyan
    
    # Fourth row
    add_patch(212, 786, 43, 29) # White
    add_patch(460, 801, 42, 29) # Neutral 8
    add_patch(726, 810, 49, 36) # Neutral 6.5
    add_patch(970, 800, 52, 43) # Neutral 5
    add_patch(1230, 800, 53, 45) # Neutral 3.5
    add_patch(1500, 820, 54, 54) # Black 2

    return means, patch_centers


def computeCCM(RGB_values, white_point, gamma_corrected):
    d65 = np.array([0.9504, 1, 1.0888], dtype=np.float32)

    XYZ_to_sRGB =  np.array([[3.2404542, -1.5371385, -0.4985314],
                             [-0.9692660, 1.8760108, 0.0415560],
                             [0.0556434, -0.2040259, 1.0572252]], dtype=np.float32)
    
    sRGB_to_XYZ = np.linalg.inv(XYZ_to_sRGB)
     
    M_CA = np.diag(d65 / white_point)

    A = np.zeros((72, 9), dtype=np.float32)

    k = 0
    for i in range(24):
        A[k, 0:3] = RGB_values[i, :]
        A[k + 1, 3:6] = RGB_values[i, :]
        A[k + 2, 6:9] = RGB_values[i, :]
        k += 3

    data = colour.CCS_COLOURCHECKERS['cc2005'][1]  # [1] = data (OrderedDict)
    xyY = np.array(list(data.values()))  # shape (24, 3)

    # Step 2: Convert xyY to XYZ
    XYZ = np.array([colour.xyY_to_XYZ(xyy) for xyy in xyY])  # shape (24, 3)

    XYZ_vector = XYZ.reshape(-1, 1).astype(np.float32)


    CAM_to_XYZ = np.linalg.lstsq(A, XYZ_vector, rcond=None)[0].reshape(3, 3)
    CAM_to_sRGB = XYZ_to_sRGB @ M_CA @ CAM_to_XYZ

    patch_RGB_values = RGB_values @ CAM_to_sRGB.T
    
    if(gamma_corrected):
        # Gamma Correction
        patch_RGB_values = linear_to_srgb(patch_RGB_values)

    return CAM_to_sRGB
    
def calibrate(input_img, gamma_corrected):
    # measurement_results.csv contains the measured XYZ values but for this implementation, we use the data from colour library
    XYZ_values = np.loadtxt('measurement_results.csv', delimiter=',').astype(np.float32)
    RGB_values = np.loadtxt('image_data.csv', delimiter=',').astype(np.float32)

    white_point = np.array([1, 1, 1], dtype=np.float32)
    CAM_to_sRGB = computeCCM(RGB_values, white_point, gamma_corrected)

    img = input_img.astype(np.float32) / 255.0 # Use the [0, 1] range
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    orig_shape = img.shape
    img = img.reshape(-1, 3)
    
    # Color correction happens here
    img = img @ CAM_to_sRGB.T
    img = np.clip(img, 0, 1)

    #img /= 60
    if (gamma_corrected):
        # Gamma correction
        img = simple_linear_to_srgb(img)  # Convert to sRGB

    img = img.reshape(orig_shape)
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
    gamma_corrected = True
    hist_equalization = True

    input_img = cv2.imread(tiff_img_path, cv2.IMREAD_COLOR)
    if input_img is None or input_img.size == 0:
        raise FileNotFoundError(f"Could not read TIFF image at {tiff_img_path}")

    # If the image is grayscale, convert to 3-channel
    if len(input_img.shape) == 2:
        input_img = np.stack([input_img]*3, axis=-1)

    if(hist_equalization):
    # HE (histogram equalization) on V channel in HSV
        img_hsv = cv2.cvtColor(input_img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(img_hsv)
        v_equalized = cv2.equalizeHist(v)
        img_hsv_equalized = cv2.merge((h, s, v_equalized))
        input_img = cv2.cvtColor(img_hsv_equalized, cv2.COLOR_HSV2BGR)

    if (gamma_corrected):
        input_img = simple_srgb_to_linear(input_img.astype(np.float32) / 255.0) * 255.0

    means, patch_centers = calculate_patch_means(input_img)
    #print("Means: ", means)
    np.savetxt("image_data.csv", means, delimiter=",")

    finalResult = calibrate(input_img, gamma_corrected)

    # Show input and output images side by side using matplotlib
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    # Draw squares on input image
    input_img_disp = cv2.cvtColor(input_img.copy(), cv2.COLOR_BGR2RGB).astype(np.uint8)
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
