import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import csv

def calculate_patch_means(img):
    img = img.astype(np.float32) / 65535.0 # Write normalized values to CSV

    means = []

    # First row
    x = 948
    y = 805
    w = 43
    h = 29
    rMean = np.mean(img[y:y+h, x:x+w, 2])
    gMean = np.mean(img[y:y+h, x:x+w, 1])
    bMean = np.mean(img[y:y+h, x:x+w, 0])
    means.append((rMean, gMean, bMean))

    x = 1012
    y = 807
    w = 45
    h = 33
    rMean = np.mean(img[y:y+h, x:x+w, 2])
    gMean = np.mean(img[y:y+h, x:x+w, 1])
    bMean = np.mean(img[y:y+h, x:x+w, 0])
    means.append((rMean, gMean, bMean))

    x = 1099
    y = 813
    w = 44
    h = 31
    rMean = np.mean(img[y:y+h, x:x+w, 2])
    gMean = np.mean(img[y:y+h, x:x+w, 1])
    bMean = np.mean(img[y:y+h, x:x+w, 0])
    means.append((rMean, gMean, bMean))

    x = 1174
    y = 819
    w = 46
    h = 29
    rMean = np.mean(img[y:y+h, x:x+w, 2])
    gMean = np.mean(img[y:y+h, x:x+w, 1])
    bMean = np.mean(img[y:y+h, x:x+w, 0])
    means.append((rMean, gMean, bMean))
    
    # Second row
    x = 937
    y = 859
    w = 42
    h = 29
    rMean = np.mean(img[y:y+h, x:x+w, 2])
    gMean = np.mean(img[y:y+h, x:x+w, 1])
    bMean = np.mean(img[y:y+h, x:x+w, 0])
    means.append((rMean, gMean, bMean))

    x = 1009
    y = 859
    w = 52
    h = 33
    rMean = np.mean(img[y:y+h, x:x+w, 2])
    gMean = np.mean(img[y:y+h, x:x+w, 1])
    bMean = np.mean(img[y:y+h, x:x+w, 0])
    means.append((rMean, gMean, bMean))

    x = 1090
    y = 863
    w = 52
    h = 36
    rMean = np.mean(img[y:y+h, x:x+w, 2])
    gMean = np.mean(img[y:y+h, x:x+w, 1])
    bMean = np.mean(img[y:y+h, x:x+w, 0])
    means.append((rMean, gMean, bMean))

    x = 1171
    y = 870
    w = 51
    h = 33
    rMean = np.mean(img[y:y+h, x:x+w, 2])
    gMean = np.mean(img[y:y+h, x:x+w, 1])
    bMean = np.mean(img[y:y+h, x:x+w, 0])
    means.append((rMean, gMean, bMean))

    # Third row
    x = 921
    y = 912
    w = 49
    h = 36
    rMean = np.mean(img[y:y+h, x:x+w, 2])
    gMean = np.mean(img[y:y+h, x:x+w, 1])
    bMean = np.mean(img[y:y+h, x:x+w, 0])
    means.append((rMean, gMean, bMean))

    x = 1001
    y = 914
    w = 53
    h = 37
    rMean = np.mean(img[y:y+h, x:x+w, 2])
    gMean = np.mean(img[y:y+h, x:x+w, 1])
    bMean = np.mean(img[y:y+h, x:x+w, 0])
    means.append((rMean, gMean, bMean))

    x = 1083
    y = 919
    w = 54
    h = 39
    rMean = np.mean(img[y:y+h, x:x+w, 2])
    gMean = np.mean(img[y:y+h, x:x+w, 1])
    bMean = np.mean(img[y:y+h, x:x+w, 0])
    means.append((rMean, gMean, bMean))

    x = 1167
    y = 924
    w = 54
    h = 40
    rMean = np.mean(img[y:y+h, x:x+w, 2])
    gMean = np.mean(img[y:y+h, x:x+w, 1])
    bMean = np.mean(img[y:y+h, x:x+w, 0])
    means.append((rMean, gMean, bMean))

    # Fourth row	
    x = 907
    y = 970
    w = 52
    h = 43
    rMean = np.mean(img[y:y+h, x:x+w, 2])
    gMean = np.mean(img[y:y+h, x:x+w, 1])
    bMean = np.mean(img[y:y+h, x:x+w, 0])
    means.append((rMean, gMean, bMean))

    x = 989
    y = 974
    w = 55
    h = 45
    rMean = np.mean(img[y:y+h, x:x+w, 2])
    gMean = np.mean(img[y:y+h, x:x+w, 1])
    bMean = np.mean(img[y:y+h, x:x+w, 0])
    means.append((rMean, gMean, bMean))

    x = 1075
    y = 981
    w = 56
    h = 46
    rMean = np.mean(img[y:y+h, x:x+w, 2])
    gMean = np.mean(img[y:y+h, x:x+w, 1])
    bMean = np.mean(img[y:y+h, x:x+w, 0])
    means.append((rMean, gMean, bMean))

    x = 1163
    y = 987
    w = 58
    h = 46
    rMean = np.mean(img[y:y+h, x:x+w, 2])
    gMean = np.mean(img[y:y+h, x:x+w, 1])
    bMean = np.mean(img[y:y+h, x:x+w, 0])
    means.append((rMean, gMean, bMean))

    # Fifth row
    x = 889
    y = 1040
    w = 53
    h = 45
    rMean = np.mean(img[y:y+h, x:x+w, 2])
    gMean = np.mean(img[y:y+h, x:x+w, 1])
    bMean = np.mean(img[y:y+h, x:x+w, 0])
    means.append((rMean, gMean, bMean))

    x = 975
    y = 1043
    w = 61
    h = 50
    rMean = np.mean(img[y:y+h, x:x+w, 2])
    gMean = np.mean(img[y:y+h, x:x+w, 1])
    bMean = np.mean(img[y:y+h, x:x+w, 0])
    means.append((rMean, gMean, bMean))

    x = 1066
    y = 1048
    w = 60
    h = 53
    rMean = np.mean(img[y:y+h, x:x+w, 2])
    gMean = np.mean(img[y:y+h, x:x+w, 1])
    bMean = np.mean(img[y:y+h, x:x+w, 0])
    means.append((rMean, gMean, bMean))

    x = 1158
    y = 1056
    w = 64
    h = 52
    rMean = np.mean(img[y:y+h, x:x+w, 2])
    gMean = np.mean(img[y:y+h, x:x+w, 1])
    bMean = np.mean(img[y:y+h, x:x+w, 0])
    means.append((rMean, gMean, bMean))

    # Sixth row
    x = 872
    y = 1109
    w = 54
    h = 54
    rMean = np.mean(img[y:y+h, x:x+w, 2])
    gMean = np.mean(img[y:y+h, x:x+w, 1])
    bMean = np.mean(img[y:y+h, x:x+w, 0])
    means.append((rMean, gMean, bMean))

    x = 972
    y = 1117
    w = 61
    h = 56
    rMean = np.mean(img[y:y+h, x:x+w, 2])
    gMean = np.mean(img[y:y+h, x:x+w, 1])
    bMean = np.mean(img[y:y+h, x:x+w, 0])
    means.append((rMean, gMean, bMean))

    x = 1059
    y = 1125
    w = 64
    h = 56
    rMean = np.mean(img[y:y+h, x:x+w, 2])
    gMean = np.mean(img[y:y+h, x:x+w, 1])
    bMean = np.mean(img[y:y+h, x:x+w, 0])
    means.append((rMean, gMean, bMean))

    x = 1158
    y = 1136
    w = 63
    h = 57
    rMean = np.mean(img[y:y+h, x:x+w, 2])
    gMean = np.mean(img[y:y+h, x:x+w, 1])
    bMean = np.mean(img[y:y+h, x:x+w, 0])
    means.append((rMean, gMean, bMean))
   
    return means

def extractBayer10(pixel_serialized):
    # Get the higher 8 bits of the 10-bit pixel values
    C1_8 = (pixel_serialized[:, 0]).astype(np.uint16)
    C2_8 = (pixel_serialized[:, 1]).astype(np.uint16)
    C3_8 = (pixel_serialized[:, 2]).astype(np.uint16)
    C4_8 = (pixel_serialized[:, 3]).astype(np.uint16)

    # Get the lower 2 bits of the 10-bit pixel values
    lower = pixel_serialized[:, 4]

    # Combine the higher and lower bits to get the 10-bit pixel values
    C1_10 = (C1_8 << 2) | ((lower & 0xC0) >> 6)
    C2_10 = (C2_8 << 2) | ((lower & 0x30) >> 4)
    C3_10 = (C3_8 << 2) | ((lower & 0x0C) >> 2)
    C4_10 = (C4_8 << 2) | (lower & 0x03)

    bayer10 = np.stack([C1_10, C2_10, C3_10, C4_10], axis=1)

    return bayer10

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

    img = input_img.astype(np.float32) / 65535.0 # Use the [0, 1] range
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.reshape(-1, 3)
    
    # Color correction happens here
    img = img @ CAM_to_sRGB.T
    img /= 60 # Some kind of digital gain
    img = np.clip(img, 0, 1)

    img = img.reshape((1520, 2028, 3))

    #img[img < 0] = 0 # clip out-of-gamut values
    #maxval = np.max(img)
    #minval = np.min(img)

    # normalize to [0, 1]
    #img = (img - minval) / (maxval - minval)

    # Gamma correction
    img[img <= 0.0031308] = 12.92 * img[img <= 0.0031308]
    img[img > 0.0031308] = 1.055 * (img[img > 0.0031308] ** (1./2.4)) - 0.055
    #img[input_img >= 60000] = 1.0
    img[input_img >= 65000] = input_img[input_img >= 65000] / 65535
    
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = (img * 65535).astype(np.uint16)

    return img
   

def bayer_parse_and_process(byte_array, width_w_padding, width, height, num_padding_pixels, apply_wb=False, apply_demosaic=False, calculate_means=False):
    #! WARNING: THIS WORKS ONLY IN v2h2 MODE!
    try:
        # Remove padding
        byte_array_no_padding = byte_array.reshape(height, int(width_w_padding * 10 / 8))[:, :int(-num_padding_pixels * 10 / 8)]
        pixel_serialized = byte_array_no_padding.reshape(-1, 5)

        # Subtract the black level and rescale to [0, 1023]
        # The black level should be image independent but we
        # compute it as the minimum pixel value here.
        bayer10 = extractBayer10(pixel_serialized)
        blackLevel = np.min(bayer10)
        bayer10 = bayer10 - blackLevel
        saturationPoint = 1023 - blackLevel

        chunkSize = int(width / 4)
        
        # plot histograms
        start_indices_rg = np.arange(0, height * chunkSize, 2*chunkSize)
        start_indices_gb = np.arange(chunkSize, height * chunkSize, 2*chunkSize)
        selector_rg = start_indices_rg[:,None] + np.arange(chunkSize)
        selector_gb = start_indices_gb[:,None] + np.arange(chunkSize)

        # R = np.stack([bayer10[selector_rg, 0], bayer10[selector_rg, 2]], axis=0)
        # G = np.stack([bayer10[selector_rg, 1], bayer10[selector_rg, 3]], axis=0)
        # G2 = np.stack([bayer10[selector_gb, 0], bayer10[selector_gb, 2]], axis=0)
        # B = np.stack([bayer10[selector_gb, 1], bayer10[selector_gb, 3]], axis=0)
        # G = np.stack([G, G2], axis=0)

        # plt.hist(R.flatten(), bins=range(0, 1024, 1), alpha=0.5, label='R', color='red')
        # plt.hist(G.flatten(), bins=range(0, 1024, 1), alpha=0.5, label='G', color='green')
        # plt.hist(B.flatten(), bins=range(0, 1024, 1), alpha=0.5, label='B', color='blue')
        # plt.show()

        if apply_wb:
            # Apply white balance as found from the color checker
            rFactor = 1.0 / 0.94
            bFactor = 1.0 / 0.88
           
            bayer10 = bayer10.astype(np.float32)
            bayer10[selector_rg, 0] *= rFactor
            bayer10[selector_rg, 2] *= rFactor
            bayer10[selector_gb, 1] *= bFactor
            bayer10[selector_gb, 3] *= bFactor
            bayer10 = np.clip(bayer10, 0, saturationPoint) # prevent exceeding the 10-bit range

        # use the maximum 16-bit range
        # maxVal = np.max(bayer10)
        # factor = 65535.0 / maxVal
        factor = 65535.0 / saturationPoint
        bayer10 = (bayer10 * factor).astype(np.uint16) # saturation point maps to 65535

        bayer_vals_reshaped = (bayer10.reshape(height, width))

        if apply_demosaic:
            # Demosaic: Experimentally verified to be RGGB pattern
            bayer_vals_reshaped = cv2.demosaicing(bayer_vals_reshaped, cv2.COLOR_BAYER_RGGB2BGR)

            if calculate_means:
                means = calculate_patch_means(bayer_vals_reshaped)
                print("Means: ", means)
                np.savetxt("image_data.csv", means, delimiter=",")

    except Exception as e:
        print("Error: ", e)
        return np.zeros((height, width_w_padding - num_padding_pixels), dtype=np.uint8)
    
    return bayer_vals_reshaped


if __name__ == "__main__":
    raw_img_path = "./imx500_outs/2_better/frame_000000420.bin"
    #raw_img_path = "./imx500_outs/2_better/frame_000000435.bin"
    #raw_img_path = "./imx500_outs/1/frame_000006975.bin"
    #raw_img_path = "./imx500_outs/1/frame_000006990.bin"

    apply_wb = True
    apply_demosaic = True
    calculate_means = False
    
    with open(raw_img_path, "rb") as bin_file:
        data = np.array(list(bin_file.read())).astype('uint8')
    result = bayer_parse_and_process(data, 2048, 2028, 1520, 20, apply_wb, apply_demosaic, calculate_means)
    finalResult = calibrate(result)
    cv2.imwrite('final_result.png', finalResult)
    cv2.imshow('Final result', finalResult)
    cv2.waitKey(0)

    if apply_wb:
        if apply_demosaic:
            cv2.imwrite('bayer_wb_demosaic_16.png', result)
            cv2.imwrite('bayer_wb_demosaic_16.hdr', result)
            cv2.imwrite('bayer_wb_demosaic_float.hdr', result.astype(np.float32))
            cv2.imwrite('bayer_wb_demosaic_float.exr', result.astype(np.float32))
        else:
            cv2.imwrite('bayer_wb.png', result)
    else:
        if apply_demosaic:
            cv2.imwrite('bayer_demosaic.png', result)
        else:
            cv2.imwrite('bayer.png', result)
    print("HALT")
