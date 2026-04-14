# encoding=utf-8
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import math

def text_to_binary(text):
    # 将文本转换为二进制字符串
    binary_str = ''
    for char in text:
        binary_str += format(ord(char), '08b')
    return binary_str

def binary_to_text(binary_str):
    # 将二进制字符串转换为文本
    text = ''
    for i in range(0, len(binary_str), 8):
        byte = binary_str[i:i+8]
        text += chr(int(byte, 2))
    return text

def calculate_psnr_grayscale(img1_path, img2_path):
    # 计算两幅灰度图像的峰值信噪比
    # 读取图像并确保是灰度图
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    
    if img1 is None or img2 is None:
        raise ValueError("无法读取图像文件")
    
    # 确保图像尺寸相同
    if img1.shape != img2.shape:
        # 调整图像尺寸使其相同
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    
    img1 = np.float64(img1)
    img2 = np.float64(img2)
    mse = np.mean((img1 - img2) ** 2)
    
    if mse < 1.0e-10:
        return 100
    
    PIXEL_MAX = 255.0
    psnr_value = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    return round(psnr_value, 2)

class ImprovedLSB:
    def __init__(self):
        pass
    
    def hide_message_improved_sequential(self, original_path, secret_msg, output_path):
        # 改进的LSB隐写 - 次低有效位嵌入 + 随机位置选择
        # 读取图像
        img = Image.open(original_path)
        
        # 确保图像是RGB格式
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        img_array = np.array(img)
        height, width, channels = img_array.shape
        
        # 转换为二进制消息
        binary_msg = text_to_binary(secret_msg)
        msg_length = len(binary_msg)
        
        # 检查容量（只使用R通道）
        if msg_length > height * width:
            raise ValueError("消息太长，无法隐藏在此图像中")
        
        # 次低有效位嵌入（bit 1）+ 随机位置
        stego_array = img_array.copy()
        flat_r_channel = stego_array[:,:,0].flatten()
        
        # 生成随机嵌入位置（固定 seed 保证可提取）
        np.random.seed(2021)
        positions = np.random.permutation(height * width)[:msg_length]
        
        for i, idx in enumerate(positions):
            # 次低有效位（bit 1）嵌入：清除 bit 1，再设置秘密位左移1位
            flat_r_channel[idx] = (flat_r_channel[idx] & 0xFD) | (int(binary_msg[i]) << 1)
        
        # 重塑R通道并保存
        stego_array[:,:,0] = flat_r_channel.reshape((height, width))
        stego_img = Image.fromarray(stego_array)
        stego_img.save(output_path)
        
        return img_array, stego_array, msg_length
    
    def extract_message_improved_sequential(self, stego_path, msg_length_bits):
        # 从改进的LSB隐写中提取信息（随机位嵌入）
        stego_img = Image.open(stego_path)
        stego_array = np.array(stego_img)
        
        # 从次低有效位（bit 1）提取
        flat_r_channel = stego_array[:,:,0].flatten()
        binary_msg = ''
        
        # 重建相同的随机位置序列
        height, width = stego_array.shape[:2]
        np.random.seed(2021)
        positions = np.random.permutation(height * width)[:msg_length_bits]
        
        for idx in positions:
            bit_value = (flat_r_channel[idx] >> 1) & 1  # 提取 bit 1
            binary_msg += str(bit_value)
        
        secret_msg = binary_to_text(binary_msg)
        return binary_msg, secret_msg

def display_improved_comparison(original_array, stego_array):
    # 显示改进算法的图像对比
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.imshow(original_array)
    plt.title('原始图像')
    plt.axis('off')
    
    plt.subplot(122)
    plt.imshow(stego_array)
    plt.title('改进LSB携密图像')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# 执行实验思考部分
if __name__ == "__main__":
    print("\n=== 改进的LSB隐写实验（随机位嵌入） ===")
    try:
        improved_lsb = ImprovedLSB()
        
        # 使用次低有效位（bit 1）进行随机嵌入
        original_improved, stego_improved, msg_length_improved = improved_lsb.hide_message_improved_sequential(
            'buptgray.bmp', "BUPTshahexiaoqu", 'buptgraystego1.bmp'
        )
        
        display_improved_comparison(original_improved, stego_improved)
        
        # 提取信息
        binary_improved, extracted_improved = improved_lsb.extract_message_improved_sequential(
            'buptgraystego1.bmp', msg_length_improved
        )
        
        print("提取隐藏的秘密信息二进制为: " + binary_improved)
        print("提取隐藏的秘密信息为: " + extracted_improved)
        
        # 计算PSNR
        psnr_improved = calculate_psnr_grayscale('buptgray.bmp', 'buptgraystego1.bmp')
        print("本组计算的峰值信噪比为: " + str(psnr_improved))
        
        print("完成！")
    except Exception as e:
        print(f"执行出错: {e}")