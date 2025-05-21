import cv2
import os
import numpy as np

input_folder = "./results/X-Ray_gen/Pleural_Thickening"
output_folder = "./results/X-Ray_gen/enhenced_Pleural_Thickening"

# 创建输出目录
os.makedirs(output_folder, exist_ok=True)

def enhance_image(img):
    """图像增强处理流程"""
    # 转换为灰度图（如果输入是彩色）
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # CLAHE增强（保留原始参数）
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(img)
    
    # 非锐化掩模（Unsharp Masking）（保留原始参数）
    blurred = cv2.GaussianBlur(enhanced, (0, 0), 3)
    sharpened = cv2.addWeighted(enhanced, 1.2, blurred, -0.2, 0)
    
    # 更平滑的对比度拉伸 - 参数已调低
    # 使用更宽的百分位数范围，减少对比度增强的强度
    low_percentile = np.percentile(sharpened, 15)   # 百分位下限（越大越平滑）
    high_percentile = np.percentile(sharpened, 85)  # 百分位上限（越小越平滑）
    
    # 为防止除零错误，确保分母不为零
    if high_percentile > low_percentile:
        # 对图像进行非常轻微的线性拉伸
        alpha = 0.5  # 平滑系数，越小对比度越低（0.3-0.7是较好范围）
        beta = 0.5   # 混合系数，控制原图与拉伸图的混合比例（0.5表示各占一半）
        
        # 先计算拉伸后的图像
        stretched_part = np.clip(((sharpened - low_percentile) / (high_percentile - low_percentile) * 255.0), 0, 255)
        # 然后与原图混合（这是控制对比度最关键的参数）
        stretched = np.clip((alpha * stretched_part + (1-alpha) * sharpened), 0, 255).astype(np.uint8)
        # 再与原始图像按beta比例混合，进一步降低对比度
        stretched = cv2.addWeighted(stretched, beta, sharpened, (1-beta), 0)
    else:
        # 如果高低阈值相同，就不进行拉伸
        stretched = sharpened
    
    # 轻微高斯模糊以平滑噪点
    final = cv2.GaussianBlur(stretched, (3, 3), 0.3)
    
    # 返回最终结果
    return final

# 处理所有图像文件
for filename in os.listdir(input_folder):
    # 支持常见图像格式
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp')):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        
        # 读取图像
        img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        
        # 执行增强处理
        processed = enhance_image(img)
        
        # 保存结果
        cv2.imwrite(output_path, processed)

print("图像增强处理完成！")