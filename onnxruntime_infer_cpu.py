
import os
import sys
import numpy as np
import onnxruntime
import argparse
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
import torch
from utils import to_numpy

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

parse = argparse.ArgumentParser(description='onnx model infer!')
parse.add_argument('--type', type=str,default='one_image', help='推理类型支持：one_image')
parse.add_argument('--onnx_path', type=str, default=r'.\weights\sewer_cls.onnx', help='onnx包存放路径')
parse.add_argument('--one_image_path', type=str, default=r'.\infer_data\00000081.png', help='单张图片路径')#resnet101_pth,model_lightnining_save
parse.add_argument('--device', type=str, default='cpu', help='默认设备cpu')
parse.add_argument('--savepath',type=str, default=r'.\infer_results\result_pics',help='标记图片存储路径')
args = parse.parse_args()
label_names = ["BX","CJ","CK","CQ","CR","DK","DP","JB","JG","NO","PL","SG","YW","ZG","ZW"]

def onnx_infer_one_image(args):

    ort_session = onnxruntime.InferenceSession(args.onnx_path)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.514, 0.450, 0.349], std=[0.211, 0.203, 0.164])
    ])
    image = Image.open(args.one_image_path)
    image_data = transform(image)
    image_data = torch.unsqueeze(image_data, dim=0)
    if args.device == 'cpu':
        image_name = os.path.basename(args.one_image_path)
        ort_input = {ort_session.get_inputs()[0].name: to_numpy(image_data)}
        start = time.time()
        ort_out = ort_session.run(None, ort_input)
        end = time.time()
        out = np.argmax(ort_out[0], axis=1)
        result = label_names[int(out)]
        timecost = round(end - start, 4)
    elif args.device == 'cuda':
        pass
    else:
        exit(0)


import time
if __name__ == '__main__':
    args = parse.parse_args()
    onnx_infer_one_image(args)