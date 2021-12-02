import cv2
import numpy as np
from scipy import signal

def skeleton(image):
    return cv2.ximgproc.thinning(image)

def ced(image):
    m1 = np.array([[5, 5, 5],[-3,0,-3],[-3,-3,-3]])
    m8 = np.array([[-3, 5,5],[-3,0,5],[-3,-3,-3]])
    m7 = np.array([[-3,-3,5],[-3,0,5],[-3,-3,5]])
    m6 = np.array([[-3,-3,-3],[-3,0,5],[-3,5,5]])
    m5 = np.array([[-3, -3, -3],[-3,0,-3],[5,5,5]])
    m4 = np.array([[-3, -3, -3],[5,0,-3],[5,5,-3]])
    m3 = np.array([[5, -3, -3],[5,0,-3],[5,-3,-3]])
    m2 = np.array([[5, 5, -3],[5,0,-3],[-3,-3,-3]])
    list_m = [m1,m2,m3,m4,m5,m6,m7,m8]

    list_e = []
    count = 1
    
    for m in list_m:
        imgk = signal.convolve2d(image, m,boundary='symm')
        list_e.append(np.abs(imgk))
        out = imgk.astype(np.uint8)
        count += 1
    count
    e = list_e[0]
    for i in range(len(list_e)):
        e = e*(e>=list_e[i]) + list_e[i]*(e<list_e[i])
        
    e[e>255] = 255
    e=e.astype(np.uint8)
    return e

def binary(image):
    (thresh, im_bw) = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    thresh = 127
    im_bw = cv2.threshold(image, thresh, 255, cv2.THRESH_BINARY)[1]
    return im_bw

def noise(bw):
    den = cv2.fastNlMeansDenoising(bw, None, 10, 7, 15)
    return den

def adjust_to_see(img):

    (h, w) = img.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D((cX, cY), -90, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    img = cv2.warpAffine(img, M, (nW + 1, nH + 1))
    img = cv2.warpAffine(img.transpose(), M, (nW, nH))

    return img


def augmentation(imgs,
                 rotation_range=0,
                 scale_range=0,
                 height_shift_range=0,
                 width_shift_range=0,
                 dilate_range=1,
                 erode_range=1):

    imgs = imgs.astype(np.float32)
    _, h, w = imgs.shape

    dilate_kernel = np.ones((int(np.random.uniform(1, dilate_range)),), np.uint8)
    erode_kernel = np.ones((int(np.random.uniform(1, erode_range)),), np.uint8)
    height_shift = np.random.uniform(-height_shift_range, height_shift_range)
    rotation = np.random.uniform(-rotation_range, rotation_range)
    scale = np.random.uniform(1 - scale_range, 1)
    width_shift = np.random.uniform(-width_shift_range, width_shift_range)

    trans_map = np.float32([[1, 0, width_shift * w], [0, 1, height_shift * h]])
    rot_map = cv2.getRotationMatrix2D((w // 2, h // 2), rotation, scale)

    trans_map_aff = np.r_[trans_map, [[0, 0, 1]]]
    rot_map_aff = np.r_[rot_map, [[0, 0, 1]]]
    affine_mat = rot_map_aff.dot(trans_map_aff)[:2, :]

    for i in range(len(imgs)):
        imgs[i] = cv2.warpAffine(imgs[i], affine_mat, (w, h), flags=cv2.INTER_NEAREST, borderValue=255)
        imgs[i] = cv2.erode(imgs[i], erode_kernel, iterations=1)
        imgs[i] = cv2.dilate(imgs[i], dilate_kernel, iterations=1)

    return imgs


def normalization(imgs):

    imgs = np.asarray(imgs).astype(np.float32)
    imgs = np.expand_dims(imgs / 255, axis=-1)
    return imgs


"""
Preprocess metodology based in:
    H. Scheidl, S. Fiel and R. Sablatnig,
    Word Beam Search: A Connectionist Temporal Classification Decoding Algorithm, in
    16th International Conference on Frontiers in Handwriting Recognition, pp. 256-258, 2018.
"""


def preprocess(img, input_size):
   
    def imread(path):
        if path.find('Zialcita_Risperidone_1 CHECK.png') != -1:
            img = cv2.imread('C:/Users/Camille/source/repos/CED-CRNN/raw/doctors/images/Zialcita_Risperidone_1 CHECK.png', cv2.IMREAD_GRAYSCALE)
        elif path.find('Yenson_Prednisolone_3.png') != -1:
            img = cv2.imread('C:/Users/Camille/source/repos/CED-CRNN/raw/doctors/images/Yenson_Prednisolone_3.png', cv2.IMREAD_GRAYSCALE)
        elif path.find('PayuranGatchalian_Dobutamine_1_0_0_2778.png') != -1:
            img = cv2.imread('C:/Users/Camille/source/repos/CED-CRNN/raw/doctors/images/PayuranGatchalian_Dobutamine_1_0_0_2778.png', cv2.IMREAD_GRAYSCALE)
        elif path.find('PayuranGatchalian_Azathioprine_3 CHECK_0_0_1447.png') != -1:
            img = cv2.imread('C:/Users/Camille/source/repos/CED-CRNN/raw/doctors/images/PayuranGatchalian_Azathioprine_3 CHECK_0_0_1447.png', cv2.IMREAD_GRAYSCALE)
        else:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        u, i = np.unique(np.array(img).flatten(), return_inverse=True)
        background = int(u[np.argmax(np.bincount(i))])
        return img, background
    
    if isinstance(img, str):
        img, bg = imread(img)
        
    if isinstance(img, tuple):
        image, boundbox = img
        img, bg = imread(image)

        for i in range(len(boundbox)):
            if isinstance(boundbox[i], float):
                total = len(img) if i < 2 else len(img[0])
                boundbox[i] = int(total * boundbox[i])
            else:
                boundbox[i] = int(boundbox[i])

        img = np.asarray(img[boundbox[0]:boundbox[1], boundbox[2]:boundbox[3]], dtype=np.uint8)
    
    # img = noise(img)
    img = binary(img)
    # img = np.invert(img)
    img = cv2.blur(img,(3,3))
    img = ced(img)
    # img = skeleton(img)

    wt, ht, _ = input_size
    h, w = np.asarray(img).shape
    f = max((w / wt), (h / ht))

    new_size = (max(min(wt, int(w / f)), 1), max(min(ht, int(h / f)), 1))
    img = cv2.resize(img, new_size)

    target = np.ones([ht, wt], dtype=np.uint8) * bg
    target[0:new_size[1], 0:new_size[0]] = img
    img = cv2.transpose(target)

    return img