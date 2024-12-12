import os
import sys
import argparse
import requests
import json
from PIL import Image, ImageDraw, ImageFont

# 将上级目录添加到系统路径，以便导入模块
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

# 定义命令行参数解析器
parse = argparse.ArgumentParser(description='pth model infer!')
parse.add_argument('--type', type=str, default='one_image', help='推理类型支持：one_image')  # 指定推理类型
parse.add_argument('--one_image_path', type=str, default=r'.\infer_data\00000081.png', help='单张图片路径')  # 图片路径
parse.add_argument('--savepath', type=str, default=r'.\infer_results\result_pics', help='标记图片存储路径')  # 推理后图片保存路径
args = parse.parse_args()

# 定义函数：通过 API 对单张图片进行推理
def pth_infer_one_image_API(args):
    """
    使用 Flask 提供的 API 服务对单张图片进行推理，并将结果标注到图片上。

    参数:
        args: 命令行参数，包含图片路径、保存路径等信息。
    """
    # 定义 API 地址和请求头
    url = 'http://127.0.0.1:5000/resnet18'  # 推理 API 地址
    headers = {'Content-Type': 'image/png'}  # 指定数据类型为 PNG

    # 以二进制方式打开图片
    files = {'media': open(args.one_image_path, 'rb')}  # 用于测试 API 的文件形式
    data = open(args.one_image_path, 'rb').read()  # 读取图片内容为字节流

    # 向 API 发送 POST 请求
    requests.post(url, files=files)  # 发送文件形式的请求（可选）
    r = requests.post(url, data=data, headers=headers, verify=False)  # 发送二进制数据请求

    # 解析 API 返回的推理结果
    infer_results = json.loads(r.text)  # 将返回的 JSON 格式结果解析为 Python 对象
    infer_time, result = infer_results[0], infer_results[1]  # 推理时间和分类结果

    # 如果推理结果不是 "NO"，则进行处理
    if result != 'NO':
        image_name = os.path.basename(args.one_image_path)  # 获取图片名称
        print(image_name + ':', result + '缺陷,' + ' 推理时间为：', round(infer_time, 4))  # 输出推理结果

        # 打开图片并绘制推理结果
        image = Image.open(args.one_image_path)  # 打开原始图片
        draw = ImageDraw.Draw(image)  # 创建绘图对象
        font = ImageFont.truetype(r"C:\Windows\Fonts\simsun.ttc", 35, encoding="unic")  # 设置字体
        draw.text((15, 10), result, font=font, fill='red')  # 在图片左上角绘制推理结果
        image.save(args.savepath + '\%s' % image_name)  # 保存带有标注的图片到指定路径

# 主函数：根据命令行参数调用对应的推理函数
if __name__ == '__main__':
    args = parse.parse_args()  # 解析命令行参数
    if args.type == 'one_image':  # 如果指定类型是单张图片推理
        pth_infer_one_image_API(args)  # 调用单张图片推理函数
    else:
        exit(0)  # 如果类型不支持，退出程序
    exit(0)
