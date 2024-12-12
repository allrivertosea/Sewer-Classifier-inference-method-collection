import numpy as np
import openvino.runtime as ov
import time
import sys
import os
import argparse
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
import torch

# 添加上级目录到系统路径
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

# 定义命令行参数
parse = argparse.ArgumentParser(description='opv model infer!')
parse.add_argument('--type', type=str, default='one_image', help='推理类型支持：one_image')
parse.add_argument('--xml_path', type=str, default=r".\weights\ir\sewer_cls.xml", help='xml存放路径')
parse.add_argument('--one_image_path', type=str, default=r'.\infer_data\00000081.png', help='单张图片路径')
parse.add_argument('--device', type=str, default='CPU', help='默认设备cpu')
parse.add_argument('--savepath', type=str, default=r'.\infer_results\result_pics', help='标记图片存储路径')

# 定义标签名称列表
label_names = ["BX", "CJ", "CK", "CQ", "CR", "DK", "DP", "JB", "JG", "NO", "PL", "SG", "YW", "ZG", "ZW"]


# 定义函数：使用 OpenVINO 模型对单张图片进行推理
def opv_infer_one_image(args):
    """
    使用 OpenVINO 模型对单张图片进行推理。

    参数:
        args: 命令行参数，包含模型路径、图片路径、设备等信息。
    """
    # 初始化 OpenVINO 核心对象
    core = ov.Core()
    ie = core
    model = ie.read_model(model=args.xml_path)  # 加载模型
    compiled_model = ie.compile_model(model=model, device_name=args.device)  # 编译模型
    output_layer = compiled_model.output(0)  # 获取输出层

    # 定义图像预处理步骤
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.514, 0.450, 0.349], std=[0.211, 0.203, 0.164])
    ])

    # 加载并预处理图片
    image = Image.open(args.one_image_path)
    image_data = transform(image)
    image_data = torch.unsqueeze(image_data, dim=0)

    # 执行推理并记录时间
    start = time.time()
    result_infer = compiled_model([image_data])[output_layer]
    end = time.time()
    timecost = round(end - start, 4)


# 主函数：调用推理函数
if __name__ == "__main__":
    args = parse.parse_args()  # 解析命令行参数
    opv_infer_one_image(args)  # 调用单张图片推理函数
