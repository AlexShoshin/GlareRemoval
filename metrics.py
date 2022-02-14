import math
import numpy as np
import imageio as io
import cv2
from skimage.color import rgb2gray
from skimage import feature
from foggy.FADE import *





def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

import cv2
def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

from skimage import io, color

def UCIQE(img):
    lab = color.rgb2lab(img)
    
    c1 = 0.4680
    c2 = 0.2745
    c3 = 0.2576
    
    l = lab[:, :, 0] / 100
    a = lab[:, :, 1] / 110
    b = lab[:, :, 2] / 110
    
    
    chroma = np.sqrt(a**2 + b**2)
    u_c = np.mean(chroma)
    sigma_c = np.sqrt(np.mean(chroma**2 - u_c**2))
    
    saturation = np.divide(chroma, l + 1e-20)
    u_s = np.mean(saturation)
    
    contrast_l = (np.amax(l) - np.amin(l))
    
    UCIQE = c1 * sigma_c / 0.05 + c2 * contrast_l / 0.25 + c3 * u_s / 0.17
    
    
    return UCIQE

def CFF(img):
    imColor = (img).astype(float)
    
    R = imColor[:, :, 0]
    G = imColor[:, :, 1]
    B = imColor[:, :, 2]
    
    RR = np.log(R + 1e-6) - np.mean(np.log(R + 1e-6))
    GG = np.log(G + 1e-6) - np.mean(np.log(G + 1e-6))
    BB = np.log(B + 1e-6) - np.mean(np.log(B + 1e-6))
    
    alpha = RR - GG
    beta = 0.5 * (RR + GG) - BB
    
    mu_alpha = np.mean(alpha)
    mu_beta = np.mean(beta)
    
    var_alpha = np.std(alpha)
    var_beta = np.std(beta)
    
    colorfulness = 1000 * ((np.sqrt(var_alpha * var_alpha + var_beta * var_beta) + 0.3 * np.sqrt(mu_alpha * mu_alpha + mu_beta * mu_beta)) / 85.59)
    
    ######
    
    gray = rgb2gray(img)
    
    contrast = CFF_contrast(gray)
    
    ######
    
    foggy = FADE(img)
    
    
    ######
    
    colorfulness = colorfulness / 4
    contrast = contrast / 0.04
    foggy = 2 - foggy / 0.313
    
    
    c = [0.17593, 0.61759, 0.33988]
    quality = c[0] * (colorfulness) + c[1] * (contrast) + c[2] * (foggy)  
    
    return quality

def CFF_contrast(gray):
    T = 0.002
    (m, n) = gray.shape
    rb = 64
    rc = 64
    CON = []
    
    for i in range(0, int(np.floor(m / rb))):
        for j in range(0, int(np.floor(n / rc))):
            A_temp = gray[rb * i : rb * (i + 1), rc * j : rc * (j + 1)]

            desicion = get_edgeblocks_mod(A_temp, T)
            
            if(desicion):
                CON.append(np.sqrt(np.sum(np.multiply(A_temp - np.mean(A_temp), A_temp - np.mean(A_temp))) / (rb * rc)))
    brow = np.floor(m / rb)
    bcol = np.floor(n / rc)
            
    return np.sum(CON) / (brow * bcol)

def get_edgeblocks_mod(mat, T):
    edges = feature.canny(mat)
    (m, n) = edges.shape
    
    L = m * n
    
    num_pix = np.sum(edges)
    return num_pix > (L * T)
    


    
    