import os
import sys
import argparse
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
import torch
from utils import load_model, MCinfer_data, output_pics_label
from torch.utils.data import DataLoader
import torch.nn as nn
import cv2
import time

# 将上级目录添加到系统路径，以便导入模块
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

# 定义命令行参数解析器
parse = argparse.ArgumentParser(description='pth model infer!')
parse.add_argument('--type', type=str, default='video', help='推理类型支持：one_image/images/video')
parse.add_argument('--weights_path', type=str, default=r'.\weights\sewer_cls.pth', help='pth模型存放路径')
parse.add_argument('--one_image_path', type=str, default=r'.\infer_data\00000081.png', help='单张图片路径')
parse.add_argument('--image_path', type=str, default=r'.\infer_data\pics', help='图片文件夹路径')
parse.add_argument('--video_path', type=str, default=r'.\infer_data\sewer_1.mp4', help='视频路径')
parse.add_argument('--batch_size', type=int, default=1, help='batchsize')
parse.add_argument('--savepath', type=str, default=r'.\infer_results\result_pics', help='推理图片后，标记图片存储路径')
parse.add_argument('--savepath_video_pic', type=str, default=r'.\infer_results\video_pics', help='推理视频后，标记图片存储路径')
parse.add_argument("--model_name", type=str, default="resnet18")
parse.add_argument("--num_classes", type=int, default=15)
parse.add_argument("--pic_label_out", type=bool, default=True)
parse.add_argument("--prob_threshold", type=float, default=0.6)
args = parse.parse_args()

# 定义标签名称列表
label_names = ["BX", "CJ", "CK", "CQ", "CR", "DK", "DP", "JB", "JG", "NO", "PL", "SG", "YW", "ZG", "ZW"]
# 设置设备为GPU或CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 定义函数：用于对单张图片进行推理
def pth_infer_one_image(args):
    """
    对单张图像进行模型推理，并输出结果。

    参数:
        args: 命令行参数，包含图片路径和模型等信息。
    """
    # 加载模型并设置为评估模式
    model = load_model(args, device)
    model.eval()
    # 定义图像预处理步骤
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.514, 0.450, 0.349], std=[0.211, 0.203, 0.164])
    ])
    # 加载并预处理图像
    image = Image.open(args.one_image_path)
    image_data = transform(image)
    image_data = torch.unsqueeze(image_data, dim=0)
    image_data = image_data.to(device)
    image_name = os.path.basename(args.one_image_path)
    # 记录推理时间
    start = time.time()
    output = model(image_data)
    end = time.time()
    # 获取预测结果
    out = torch.argmax(output)
    result = label_names[int(out)]
    timecost = round(end - start, 4)
    # 输出结果（可以根据需要添加打印或保存功能）
    print(f"Image: {image_name}, Prediction: {result}, Time: {timecost}s")

# 定义函数：用于对文件夹中的多张图片进行批量推理
def pth_infer_image(args):
    """
    对文件夹中的图像进行批量推理，并在图像上标注预测结果。

    参数:
        args: 命令行参数，包含图片文件夹路径、模型等信息。
    """
    # 加载模型并设置为评估模式
    model = load_model(args, device)
    model.eval()
    # 定义图像预处理步骤
    infer_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.514, 0.450, 0.349], std=[0.211, 0.203, 0.164])
    ])
    # 创建数据集和数据加载器
    dataset = MCinfer_data(args.image_path, transform=infer_transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=2, pin_memory=True)
    softmax = nn.Softmax(dim=1)
    dataLen = len(dataloader)

    # 禁用梯度计算，加速推理
    with torch.no_grad():
        for i, (images, imgPaths) in enumerate(dataloader):
            if i % 100 == 0:
                print(f"{i} / {dataLen}")
            images = images.to(device)
            # 记录推理时间
            start = time.time()
            output = model(images)
            end = time.time()
            softmaxOutput = softmax(output).detach().cpu().numpy()
            if args.pic_label_out:
                print('每批图片推理时间：', end - start)
                # 对每张图片进行结果标注并保存
                for j in range(len(images)):
                    img_org = Image.open(imgPaths[j])
                    img_name = os.path.basename(imgPaths[j])
                    output_pics_label(softmaxOutput[j], label_names, img_org, img_name, args.savepath, args.prob_threshold)

# 定义函数：用于对视频进行逐帧推理
def pth_infer_video(args):
    """
    对视频进行逐帧推理，并在满足条件时保存标注后的帧。

    参数:
        args: 命令行参数，包含视频路径、模型等信息。
    """
    # 加载模型并设置为评估模式
    model = load_model(args, device)
    model.eval()
    # 定义图像预处理步骤
    infer_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.514, 0.450, 0.349], std=[0.211, 0.203, 0.164])
    ])
    # 打开视频文件
    cap = cv2.VideoCapture(args.video_path)
    softmax = nn.Softmax(dim=1)
    sum_num = 0
    while True:
        ret, frame = cap.read()
        if ret:
            # 将帧转换为RGB格式并预处理
            image_data = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_data = Image.fromarray(image_data)
            image_data = infer_transform(image_data)
            image_data = torch.unsqueeze(image_data, dim=0)
            image_data = image_data.to(device)
            # 记录推理时间
            start = time.time()
            output = model(image_data)
            end = time.time()
            softmaxOutput = softmax(output).detach().cpu().numpy()
            prob = max(softmaxOutput[0])
            print('每帧推理时间：', end - start)
            out = torch.argmax(output)
            result = label_names[int(out)]
            # 在帧上绘制预测结果
            cv2.putText(frame, result, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=2)
            # 根据条件保存帧
            if result != 'NO' and args.pic_label_out and prob >= args.prob_threshold:
                cv2.imwrite(os.path.join(args.savepath_video_pic, f'{sum_num:08d}.png'), frame)
            sum_num += 1
        else:
            break
    cap.release()

# 主函数，根据命令行参数调用对应的推理函数
if __name__ == '__main__':
    args = parse.parse_args()
    if args.type == 'one_image':
        pth_infer_one_image(args)
    elif args.type == 'images':
        pth_infer_image(args)
    elif args.type == 'video':
        pth_infer_video(args)
    else:
        exit(0)
    exit(0)
