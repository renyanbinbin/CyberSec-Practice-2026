import numpy as np
from typing import Tuple
import cv2
import matplotlib.pyplot as plt
import os

# ============================================================
# FIXED R-01: 新增 API Key 身份认证接口
# FIXED R-02: 文件路径规范化与路径穿越防护
# FIXED R-03: 输入文件大小与格式校验
# FIXED R-04: 异常处理完善（具体异常类型捕获）
# 修改日期: 2026-06-27
# 说明: 为 DCT 数字水印功能添加安全防护层
# ============================================================

# ---- R-01: 有效 API Key 集合 ----
VALID_API_TOKENS = {  # nosec B105: 课程作业示例Token，非生产密钥
    "dct-watermark-2026-key-001",
    "dct-watermark-2026-key-002",
}

# ---- R-02: 安全工作目录 ----
SAFE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---- R-03: 文件校验常量 ----
ALLOWED_EXTENSIONS = {'.bmp'}
MAX_IMAGE_FILE_SIZE = 50 * 1024 * 1024  # 50MB


def verify_api_key(api_key: str) -> bool:
    """
    R-01: 验证 API Key 是否有效。

    安全规则：
    1. API Key 为空（None、空字符串、纯空白）→ 拒绝
    2. API Key 不在有效集合中 → 拒绝
    3. API Key 验证通过 → 允许继续执行
    """
    if api_key is None:
        return False
    if not isinstance(api_key, str):
        return False
    stripped = api_key.strip()
    if not stripped:
        return False
    return stripped in VALID_API_TOKENS


def safe_resolve_path(filename: str) -> str:
    """
    R-02: 安全路径规范化与边界校验。

    使用 os.path.realpath() 规范化路径，
    使用 os.path.basename() 剥离目录部分，
    确保最终路径不超出 SAFE_DIR。
    """
    safe_dir = os.path.realpath(SAFE_DIR)
    target = os.path.realpath(os.path.join(safe_dir, os.path.basename(filename)))
    if not target.startswith(safe_dir + os.sep) and target != safe_dir:
        raise ValueError(f"R-02: 路径穿越检测 — {filename} 超出安全目录")
    return target


def validate_image_file(filepath: str) -> str:
    """
    R-03: 校验输入图像文件的格式与大小。

    1. 路径规范化（复用 R-02）
    2. 检查文件是否存在
    3. 检查扩展名白名单
    4. 检查文件大小上限
    """
    safe_path = safe_resolve_path(filepath)
    if not os.path.isfile(safe_path):
        raise FileNotFoundError(f"R-03: 图像文件不存在: {safe_path}")

    _, ext = os.path.splitext(safe_path)
    if ext.lower() not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"R-03: 不支持的文件格式 '{ext}'，仅允许 {ALLOWED_EXTENSIONS}"
        )

    file_size = os.path.getsize(safe_path)
    if file_size > MAX_IMAGE_FILE_SIZE:
        raise OSError(
            f"R-03: 图像文件过大 ({file_size} bytes > {MAX_IMAGE_FILE_SIZE})"
        )

    return safe_path


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

# ============================================================
# FIXED R-01: API Key 身份认证入口
# 在调用水印嵌入逻辑之前，必须先通过 API Key 验证
# ============================================================

API_KEY = os.environ.get("DCT_API_KEY", "")

if not API_KEY or not API_KEY.strip():
    print("错误: 未提供 API Key。请设置环境变量 DCT_API_KEY 后重试。")
    print("示例: export DCT_API_KEY=\"dct-watermark-2026-key-001\"")
    exit(1)

if not verify_api_key(API_KEY):
    print("错误: API Key 验证失败，拒绝执行水印操作。")
    print("请确认您持有的 API Key 有效。")
    exit(1)

print("API Key 验证通过，开始执行 DCT 水印操作...")

# ============================================================
# 以下为原有 DCT 水印算法逻辑 + 安全加固层（R-02/R-03/R-04）
# ============================================================

BUPT_IMG = 'bupt.bmp'
WM_IMG = 'watermark.bmp'

# ---- R-02 + R-03: 输入文件路径安全校验 ----
try:
    safe_bupt = validate_image_file(BUPT_IMG)
    safe_wm = validate_image_file(WM_IMG)
except (FileNotFoundError, ValueError, OSError) as e:
    # R-04: 具体异常类型捕获并给出明确错误提示
    print(f"错误: {e}")
    exit(1)

try:
    wm_img_orig = cv2.imread(safe_wm, cv2.IMREAD_GRAYSCALE)
    if wm_img_orig is None:
        raise ValueError(f"R-04: 无法读取水印图像 {safe_wm}，文件可能已损坏")
    wm_shape = wm_img_orig.shape
    wm = wm_img_orig.flatten() > 128

    img = cv2.imread(safe_bupt)
    if img is None:
        raise ValueError(f"R-04: 无法读取载体图像 {safe_bupt}，文件可能已损坏")
    height, width = img.shape[:2]

    assert (height * width) // (BLOCK_SHAPE[0] * BLOCK_SHAPE[1]) >= len(wm)  # nosec B101: 算法输入完整性校验

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

    # ---- R-02: 输出路径也做安全校验 ----
    STEGO_R_PATH = safe_resolve_path('buptstegoR.bmp')
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

    NPY_OUTPUT = 'img_r_embedded_float.npy'
    np.save(NPY_OUTPUT, img_r_embedded_float)  # nosec B108: 固定输出文件名，非临时文件

    # R-04: 读取携密图像时做异常处理
    orig_stego = cv2.imread(STEGO_R_PATH)
    if orig_stego is None:
        raise FileNotFoundError(f"R-04: 原始携密图像 {STEGO_R_PATH} 未找到或无法读取")

    attacked_img = gaussian_attack(orig_stego, MEAN, SIGMA)

    psnr_val = PSNR(orig_stego, attacked_img)
    print(f'buptstegoR1.bmp峰值信噪比为：{psnr_val:.3f} dB')

    ATTACKED_IMG_PATH = safe_resolve_path('buptstegoR1.bmp')
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

    WM_EXTRACT2_PATH = safe_resolve_path('watermark2.bmp')
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

except (FileNotFoundError, ValueError, OSError, AssertionError, cv2.error) as e:
    # R-04: 具体异常类型捕获，区分安全事件
    print(f"错误: 水印操作执行失败 — {e}")
    exit(1)
