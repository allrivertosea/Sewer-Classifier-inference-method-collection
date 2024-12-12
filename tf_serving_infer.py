import os
import sys
import numpy as np
import argparse
from PIL import Image
from torchvision import transforms
import torch
from utils import keep_shape_resize, to_numpy
import cv2
from queue import Queue
import random, threading, time
import requests
import json

# 添加上级目录到系统路径
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

# 定义 TensorFlow Serving 服务的 URL
SERVER_URL_MC = 'http://localhost:8501/v1/models/sewer_cls_14c_model:predict'  # 多分类模型服务地址
SERVER_URL_BN = 'http://localhost:8501/v1/models/sewer_cls_2c_model:predict'  # 二分类模型服务地址

# 定义函数：向多分类模型发送推理请求
def prediction_mc(input):
    """
    调用多分类模型的推理接口。
    参数:
        input: 图像数据的输入。
    返回:
        prediction: 模型的预测结果。
    """
    predict_request = {"instances": input}
    response = requests.post(SERVER_URL_MC, json.dumps(predict_request))
    prediction = response.json()['predictions'][0]
    return prediction

# 定义函数：向二分类模型发送推理请求
def prediction_bn(input):
    """
    调用二分类模型的推理接口。
    参数:
        input: 图像数据的输入。
    返回:
        prediction: 模型的预测结果。
    """
    predict_request = {"instances": input}
    response = requests.post(SERVER_URL_BN, json.dumps(predict_request))
    prediction = response.json()['predictions'][0]
    return prediction

# 定义线程类：从视频中提取帧并进行二分类
class tf_infer_video(threading.Thread):
    """
    从视频中提取帧并进行二分类的线程类。
    """
    def __init__(self, name, queue):
        threading.Thread.__init__(self, name=name)
        self.data = queue  # 共享队列用于存储帧

    def run(self):
        transform = transforms.Compose([transforms.ToTensor()])  # 定义图像预处理步骤
        cap = cv2.VideoCapture(args.video_path)  # 打开视频文件
        fps = int(cap.get(cv2.CAP_PROP_FPS))  # 获取视频帧率
        sum_num = 0

        while True:
            _, frame = cap.read()  # 读取视频帧
            if _:
                sum_num += 1
                # 每隔指定时间提取一帧进行推理
                if sum_num % (fps * args.period) == 0:
                    image_data = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image_data = Image.fromarray(image_data)
                    image_data = keep_shape_resize(image_data, args.image_size)
                    image_data = transform(image_data)
                    image_data = torch.unsqueeze(image_data, dim=0)
                    if args.device == 'cpu':
                        input = to_numpy(image_data)
                        output = prediction_bn(input.tolist())
                        out = np.argmax(output)
                        result = args.binary_class_names[int(out)]
                        if result != 'NO':
                            print("生产者 %s 将产品加入队列" % self.getName())
                            self.data.put(frame)  # 将帧加入队列
                            time.sleep(random.random())
                    elif args.device == 'cuda':
                        pass  # GPU推理逻辑未实现
                    else:
                        exit(0)
                else:
                    continue
            else:
                # 视频读取完成，通知队列消费线程结束
                print("生产者 %s 完成" % self.getName())
                self.data.put('finished')
                exit(0)

# 定义线程类：从队列中消费帧并进行多分类
class tf_infer_image(threading.Thread):
    """
    从队列中消费帧并进行多分类的线程类。
    """
    def __init__(self, name, queue):
        threading.Thread.__init__(self, name=name)
        self.data = queue  # 共享队列用于存储帧

    def run(self):
        sum_num = 0
        while True:
            frame = self.data.get()  # 从队列中获取帧
            sum_num += 1
            if frame == "finished":
                print("消费者 %s 完成" % self.getName())
                exit(0)
            else:
                print("消费者 %s 将产品从队列中取出" % self.getName())
                time.sleep(random.random())
                transform = transforms.Compose([transforms.ToTensor()])
                image_data = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_data = Image.fromarray(image_data)
                image_data = keep_shape_resize(image_data, args.image_size)
                image_data = transform(image_data)
                image_data = torch.unsqueeze(image_data, dim=0)

                if args.device == 'cpu':
                    input = to_numpy(image_data)
                    output = prediction_mc(input.tolist())
                    out = np.argmax(output)
                    result = args.defect_class_names[int(out)]
                    cv2.putText(frame, result, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=2)
                    save_path = os.path.join(args.defect_pics_savepath, '%08d.png' % sum_num)
                    print("Saving image to %s" % save_path)
                    cv2.imwrite(save_path, frame)  # 保存带标注的帧
                    print("Image saved successfully!")
                elif args.device == 'cuda':
                    pass  # GPU推理逻辑未实现
                else:
                    exit(0)

# 主函数
if __name__ == '__main__':
    # 定义命令行参数解析器
    parse = argparse.ArgumentParser(description='tf_serving infer!')
    parse.add_argument('--type', type=str, default='video', help='推理类型支持：video')
    parse.add_argument('--defect_pics_savepath', type=str, default=r'.\infer_results\result_pics\\', help='标记图片存储路径')
    parse.add_argument('--video_path', type=str, default=r'.\infer_data\sewer_1.mp4', help='视频路径')
    parse.add_argument('--device', type=str, default='cpu', help='默认设备cpu')
    parse.add_argument('--image_size', type=int, default=128)
    parse.add_argument('--binary_class_names', type=list, default=['DT', 'NO'])
    parse.add_argument('--defect_class_names', type=list, default=['ZW', 'YW', 'MS', 'CQ', 'PL'])
    parse.add_argument('--period', type=int, default=1/20, help='检测间隔时间')
    args = parse.parse_args()

    print("---主线程开始---")
    queue = Queue()  # 创建共享队列
    binary_classify = tf_infer_video('binary_classify', queue)  # 创建二分类线程
    multi_classify = tf_infer_image('multi_classify', queue)  # 创建多分类线程
    binary_classify.start()  # 启动二分类线程
    multi_classify.start()  # 启动多分类线程
    binary_classify.join()  # 等待二分类线程结束
    multi_classify.join()  # 等待多分类线程结束
    print("---主线程结束---")
