from net.net import ClassifierNet
import os
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 标签名称列表
label_names = ["BX", "CJ", "CK", "CQ", "CR", "DK", "DP", "JB", "JG", "NO", "PL", "SG", "YW", "ZG", "ZW"]


# 定义函数：保持图像长宽比例缩放到指定尺寸
def keep_shape_resize(frame, size=128):
    """
    保持图像的宽高比，填充至正方形并缩放到指定尺寸。

    参数:
        frame: 输入的 PIL 图像。
        size: 缩放后的边长。
    返回:
        mask: 经过缩放和填充的 PIL 图像。
    """
    w, h = frame.size
    temp = max(w, h)
    mask = Image.new('RGB', (temp, temp), (0, 0, 0))  # 创建黑色填充背景
    if w >= h:
        position = (0, (w - h) // 2)  # 计算垂直填充位置
    else:
        position = ((h - w) // 2, 0)  # 计算水平填充位置
    mask.paste(frame, position)  # 将原图粘贴到填充背景中
    mask = mask.resize((size, size))  # 缩放到目标尺寸
    return mask


# 定义函数：将 PyTorch 张量转换为 NumPy 数组
def to_numpy(tensor):
    """
    将 PyTorch 张量转换为 NumPy 数组。

    参数:
        tensor: 输入的 PyTorch 张量。
    返回:
        NumPy 数组。
    """
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


# 定义自定义数据集类：用于多分类推理
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader


class MCinfer_data(Dataset):
    """
    多分类推理的数据集类。

    用于加载指定路径下的图片文件并进行预处理。
    """

    def __init__(self, imgRoot, transform=None):
        """
        初始化数据集。

        参数:
            imgRoot: 图片文件夹路径。
            transform: 数据预处理步骤。
        """
        super(MCinfer_data, self).__init__()
        self.imgRoot = imgRoot
        self.transform = transform
        self.loader = default_loader  # 使用默认的图像加载器

        self.imgPaths = []  # 存储图片路径
        print(self.imgRoot)
        for file in os.listdir(self.imgRoot):  # 遍历文件夹中的文件
            image_file = os.path.join(self.imgRoot, file)
            self.imgPaths.append([image_file])

    def __len__(self):
        """
        获取数据集长度。
        """
        return len(self.imgPaths)

    def __getitem__(self, index):
        """
        获取指定索引的数据。

        参数:
            index: 索引值。
        返回:
            图像张量和图像路径。
        """
        path = self.imgPaths[index][0]
        img_org = self.loader(path)  # 加载图片
        if self.transform is not None:
            img = self.transform(img_org)  # 应用预处理
        return img, path


# 定义函数：在推理图片上输出带标记的图片
def output_pics_label(softmaxOutput, label_names, img_org, img_name, outpath, prob_threshold):
    """
    在推理结果的基础上标记图片并保存。

    参数:
        softmaxOutput: 模型输出的 softmax 结果。
        label_names: 标签名称列表。
        img_org: 原始图像（PIL 格式）。
        img_name: 图片名称。
        outpath: 保存路径。
        prob_threshold: 置信度阈值。
    """
    result_index = np.argmax(softmaxOutput[0], axis=0)  # 获取最大概率的索引
    result_label = label_names[result_index]  # 获取对应的标签
    prob = max(softmaxOutput[0])  # 获取最大概率
    if result_label != 'NO' and prob >= prob_threshold:  # 如果标签不是 'NO' 且概率大于阈值
        draw = ImageDraw.Draw(img_org)  # 创建绘图对象
        font = ImageFont.truetype(r"C:\Windows\Fonts\BRITANIC.TTF", 50)  # 设置字体
        draw.text((10, 10), result_label, font=font, fill='red')  # 绘制文本到图像
        img_org.save(outpath + r'\%s' % img_name)  # 保存带标记的图片


# 定义函数：加载模型
def load_model(args, device):
    """
    加载分类模型。

    参数:
        args: 命令行参数，包含模型路径等信息。
        device: 模型运行的设备（'cpu' 或 'cuda'）。
    返回:
        加载好的模型。
    """
    net = ClassifierNet(args.model_name, num_classes=args.num_classes)  # 初始化模型
    if args.weights_path is not None:
        checkpoint = torch.load(args.weights_path)  # 加载模型权重
        net.load_state_dict(checkpoint['model_state_dict'])  # 加载模型参数
        net = net.to(device)  # 将模型移动到指定设备
    return net
