from flask import Flask, request, make_response
from PIL import Image
from io import BytesIO
import time
import json
import torch
import argparse
from torchvision import transforms
from utils import load_model

# 创建 Flask 应用实例
app = Flask(__name__)

# 定义参数解析器，用于解析命令行参数
parse = argparse.ArgumentParser(description='pth model infer!')
parse.add_argument('--weights_path', type=str, default=r'.\weights\sewer_cls.pth', help='pth模型存放路径')  # 模型权重路径
parse.add_argument("--model_name", type=str, default="resnet18")  # 使用的模型名称
parse.add_argument("--num_classes", type=int, default=15)  # 模型分类数量
args = parse.parse_args()

# 设置设备，优先使用 GPU，否则使用 CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载模型并设置为评估模式
model = load_model(args, device)
model.eval()

# 定义分类标签名称
label_names = ["BX", "CJ", "CK", "CQ", "CR", "DK", "DP", "JB", "JG", "NO", "PL", "SG", "YW", "ZG", "ZW"]


# 定义 Flask 路由，用于处理 POST 请求
@app.route('/resnet18', methods=['POST'])
def hello():
    # 定义图像的预处理步骤
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 调整图像大小为 224x224
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize(mean=[0.514, 0.450, 0.349], std=[0.211, 0.203, 0.164])  # 标准化
    ])

    # 从请求中读取图像数据
    img = request.stream.read()  # 获取图像的二进制数据
    f = BytesIO(img)  # 将图像数据转换为文件对象
    image = Image.open(f)  # 使用 PIL 打开图像
    image_data = transform(image)  # 对图像进行预处理
    image_data = torch.unsqueeze(image_data, dim=0)  # 增加 batch 维度
    image_data = image_data.to(device)  # 将图像数据移动到计算设备

    # 记录推理的开始时间
    start = time.time()
    output = model(image_data)  # 执行模型推理
    end = time.time()  # 记录推理结束时间

    # 构造结果
    result = []
    result.append(end - start)  # 推理耗时
    out = torch.argmax(output)  # 获取输出的最大值对应的索引
    result.append(label_names[int(out)])  # 将索引映射为标签名称

    # 构建响应对象
    rsp = make_response(json.dumps(result))  # 将结果转换为 JSON 格式
    rsp.mimetype = 'application/json'  # 设置响应的 MIME 类型
    rsp.headers['Connection'] = 'close'  # 设置连接为关闭状态
    return rsp  # 返回响应


# 启动 Flask 服务
if __name__ == '__main__':
    app.run(processes=1, threaded=False)  # 设置单进程、非线程化模式运行
