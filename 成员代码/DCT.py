import numpy as np
from typing import Tuple
import cv2
import matplotlib.pyplot as plt
import os

BLOCK_SHAPE = (8, 8)

def img_to_blocks(img: np.ndarray, block_shape: Tuple[int, int], dtype=np.float32) -> np.ndarray:
    height, width = img.shape[:2]
    block_height, block_width = block_shape
    shape = (height // block_height, width // block_width, block_height, block_width)
    strides = img.itemsize * np.array([width * block_height, block_width, width, 1])
    img_blocks = np.lib.stride_tricks.as_strided(img, shape, strides).astype(dtype)
    img_blocks = np.reshape(img_blocks, (shape[0] * shape[1], block_height, block_width))
    return img_blocks

def blocks_to_img(img_blocks: np.ndarray, img_shape: Tuple[int, int]) -> np.ndarray:
    height, width = img_shape[:2]
    block_height, block_width = img_blocks.shape[-2:]
    shape = (height // block_height, width // block_width, block_height, block_width)
    img_blocks = np.reshape(img_blocks, shape)

    lines = []
    for line in img_blocks:
        lines.append(np.concatenate(line, axis=1))
    img = np.concatenate(lines, axis=0)
    return img

STUDENT_INFO = ""
ALPHA = 1.3
POS_A = (4, 1)
POS_B = (3, 2)
MEAN = 0
SIGMA = 0.002

def PSNR(template, img):
    mse = np.mean((template / 255. - img / 255.)**2)
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

def gaussian_attack(img, mean, sigma):
    img = img.astype(np.float32) / 255
    noise = np.random.normal(mean, sigma, img.shape)
    img_gaussian = img + noise
    img_gaussian = np.clip(img_gaussian, 0, 1)
    img_gaussian = np.uint8(img_gaussian * 255)
    return img_gaussian

def extract_watermark_from_blocks(img_r_blocks, wm_len, POS_A, POS_B):
    extracted_wm_bits = []
    for block in img_r_blocks:
        if len(extracted_wm_bits) >= wm_len:
            break
        block_dct = cv2.dct(block)
        C_A = block_dct[POS_A]
        C_B = block_dct[POS_B]
        if C_A > C_B:
            extracted_wm_bits.append(0)
        else:
            extracted_wm_bits.append(1)
    return np.array(extracted_wm_bits)

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

if not os.path.exists('bupt.bmp') or not os.path.exists('watermark.bmp'):
    print("错误: 找不到 bupt.bmp 或 watermark.bmp 文件。请确保文件存在于脚本的同一目录。")
else:
    wm_img_orig = cv2.imread('watermark.bmp', cv2.IMREAD_GRAYSCALE)
    wm_shape = wm_img_orig.shape
    wm = wm_img_orig.flatten() > 128

    img = cv2.imread('bupt.bmp')
    height, width = img.shape[:2]

    assert (height * width) // (BLOCK_SHAPE[0] * BLOCK_SHAPE[1]) >= len(wm)

    img_b, img_g, img_r = cv2.split(img)
    img_r_blocks = img_to_blocks(img_r, BLOCK_SHAPE)
    img_r_blocks_embedded = img_r_blocks.copy()

    for i in range(len(wm)):
        block = img_r_blocks[i]
        block_dct = cv2.dct(block)
        block_dct_embedded = block_dct.copy()

        C_A_idx, C_B_idx = POS_A, POS_B

        if wm[i] == 0 and block_dct_embedded[C_A_idx] <= block_dct_embedded[C_B_idx]:
            block_dct_embedded[C_A_idx], block_dct_embedded[C_B_idx] = block_dct_embedded[C_B_idx], block_dct_embedded[C_A_idx]
            block_dct_embedded[C_B_idx] -= ALPHA

        elif wm[i] == 1 and block_dct_embedded[C_A_idx] >= block_dct_embedded[C_B_idx]:
            block_dct_embedded[C_A_idx], block_dct_embedded[C_B_idx] = block_dct_embedded[C_B_idx], block_dct_embedded[C_A_idx]
            block_dct_embedded[C_A_idx] -= ALPHA

        block_embedded = cv2.idct(block_dct_embedded)
        img_r_blocks_embedded[i] = block_embedded

    img_r_embedded_float = blocks_to_img(img_r_blocks_embedded, img.shape[:2])
    img_embedded = cv2.merge([img_b, img_g, img_r_embedded_float.astype(np.uint8)])

    STEGO_R_PATH = 'buptstegoR.bmp'
    cv2.imwrite(STEGO_R_PATH, img_embedded)

    psnr_val_stego = PSNR(img, img_embedded)
    print(f"buptstegoR.bmp峰值信噪比为：{psnr_val_stego:.3f} dB")

    plt.figure(figsize=(10, 6))

    plt.subplot(121)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f'{STUDENT_INFO}+原始图像', fontsize=10)
    plt.xticks([]), plt.yticks([])

    plt.subplot(122)
    plt.imshow(cv2.cvtColor(img_embedded, cv2.COLOR_BGR2RGB))
    plt.title(f'{STUDENT_INFO}+嵌入图像', fontsize=10)
    plt.xticks([]), plt.yticks([])

    plt.subplots_adjust(wspace=0.3, hspace=0.1)

    plt.show()

    np.save('img_r_embedded_float.npy', img_r_embedded_float)

    orig_stego = cv2.imread(STEGO_R_PATH)
    if orig_stego is None:
        raise FileNotFoundError(f"原始携密图像 {STEGO_R_PATH} 未找到")

    attacked_img = gaussian_attack(orig_stego, MEAN, SIGMA)

    psnr_val = PSNR(orig_stego, attacked_img)
    print(f'buptstegoR1.bmp峰值信噪比为：{psnr_val:.3f} dB')

    ATTACKED_IMG_PATH = 'buptstegoR1.bmp'
    cv2.imwrite(ATTACKED_IMG_PATH, attacked_img)

    plt.figure(figsize=(10, 6))

    plt.subplot(121)
    plt.imshow(cv2.cvtColor(orig_stego, cv2.COLOR_BGR2RGB))
    plt.title(f'{STUDENT_INFO}+原始携密图像', fontsize=10)
    plt.xticks([]), plt.yticks([])

    plt.subplot(122)
    plt.imshow(cv2.cvtColor(attacked_img, cv2.COLOR_BGR2RGB))
    plt.title(f'{STUDENT_INFO}+高斯噪声攻击后图像', fontsize=10)
    plt.xticks([]), plt.yticks([])

    plt.subplots_adjust(wspace=0.3, hspace=0.1)

    plt.show()

    img_b1, img_g1, img_r1 = cv2.split(attacked_img)
    img_r1_blocks = img_to_blocks(img_r1, BLOCK_SHAPE)
    wm_len = wm_shape[0] * wm_shape[1]
    extracted_wm_bits = extract_watermark_from_blocks(img_r1_blocks, wm_len, POS_A, POS_B)
    extracted_wm_bits_arr = np.array(extracted_wm_bits[:wm_len])
    extracted_wm_data = extracted_wm_bits_arr * 255
    extracted_wm_arr = extracted_wm_data.reshape(wm_shape)

    WM_EXTRACT2_PATH = 'watermark2.bmp'
    cv2.imwrite(WM_EXTRACT2_PATH, extracted_wm_arr)

    def NC(template, img):
        template = template.astype(np.uint8)
        img = img.astype(np.uint8)
        return cv2.matchTemplate(img, template, cv2.TM_CCORR_NORMED)[0][0]

    nc = NC(wm_img_orig, extracted_wm_arr)
    print(f'余弦相似度值为：{nc:.3f}')

    plt.figure(figsize=(10, 6))

    plt.subplot(121)
    plt.imshow(cv2.cvtColor(wm_img_orig, cv2.COLOR_BGR2RGB))
    plt.title(f'{STUDENT_INFO}+原始水印', fontsize=10)
    plt.xticks([]), plt.yticks([])

    plt.subplot(122)
    plt.imshow(cv2.cvtColor(extracted_wm_arr, cv2.COLOR_BGR2RGB))
    plt.title(f'{STUDENT_INFO}+提取水印', fontsize=10)
    plt.xticks([]), plt.yticks([])

    plt.subplots_adjust(wspace=0.3, hspace=0.1)

    plt.show()
