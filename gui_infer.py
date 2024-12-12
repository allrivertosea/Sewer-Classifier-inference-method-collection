import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QFileDialog,QFrame
from PyQt5.QtGui import QPixmap,QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import cv2
from net.net import ClassifierNet
from collections import OrderedDict

label_names = ["BX","CJ","CK","CQ","CR","DK","DP","JB","JG","NO","PL","SG","YW","ZG","ZW"]

def load_model(weight_path):
    net = ClassifierNet('resnet101', num_classes=15)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if weight_path is not None:
        best_model_state_dict = torch.load(weight_path)
        updated_state_dict = OrderedDict()
        for k, v in best_model_state_dict.items():
            name = k
            if "criterion" in name:
                continue
            updated_state_dict[name] = v
        net.load_state_dict(updated_state_dict)
        net = net.to(device)
        print('successfully loading model weights!')
    return net
# 定义一个线程类，用于在后台运行模型推理
class InferenceThread(QThread):
    update_signal = pyqtSignal(QPixmap)  # 自定义信号，用于将结果显示在界面上
    global current_frame
    def __init__(self, video_path):
        super().__init__()
        self.video_path = video_path
        self.model = load_model('sewer_cls.pth')
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.514, 0.450, 0.349],
                                 std=[0.211, 0.203, 0.164])])
    def run(self):
        # TODO: 实现模型推理逻辑
        cap = cv2.VideoCapture(self.video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img_tensor = self.transform(img).unsqueeze(0)
            img_tensor = img_tensor.to(device)
            with torch.no_grad():
                output = self.model(img_tensor)
                out = torch.argmax(output)
                result = label_names[int(out)]
                cv2.putText(frame, result, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=2)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result_img =QPixmap.fromImage(QImage(frame.data, frame.shape[1], frame.shape[0],QImage.Format_RGB888))
            # TODO: 处理模型输出结果
            # 如果检测到缺陷，则将结果通过update_signal信号发送到界面显示
            self.update_signal.emit(result_img)
            cv2.waitKey(int(1000 / fps))
        cap.release()


# 定义一个主窗口类
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('管道缺陷检测工具')
        self.resize(1400, 800)

        # 创建操作任务框
        operation_widget = QFrame(self)
        operation_widget.setFrameShape(QFrame.StyledPanel)
        operation_widget.setLineWidth(1)
        self.task1_label = QLabel(self)
        self.import_btn = QPushButton('导入视频')
        self.import_btn.clicked.connect(self.select_video_file)
        self.start_btn = QPushButton('开始检测')
        self.start_btn.clicked.connect(self.start_inference)
        self.pause_btn = QPushButton('停止检测')
        self.pause_btn.clicked.connect(self.pause_detection)
        operation_layout = QVBoxLayout()
        operation_layout.addWidget(self.import_btn)
        operation_layout.addWidget(self.start_btn)
        operation_layout.addWidget(self.pause_btn)
        operation_widget.setLayout(operation_layout)

        # 创建管道信息任务框
        pipeline_widget = QFrame(self)
        pipeline_widget.setFrameShape(QFrame.StyledPanel)
        pipeline_widget.setLineWidth(1)
        self.task2_label = QLabel(self)
        self.task_name_label = QLabel('任务名称：')
        self.unit_label = QLabel('检测单位：')
        self.inspector_label = QLabel('检测员：')
        pipeline_layout = QVBoxLayout()
        pipeline_layout.addWidget(self.task_name_label)
        pipeline_layout.addWidget(self.unit_label)
        pipeline_layout.addWidget(self.inspector_label)
        pipeline_widget.setLayout(pipeline_layout)

        # 创建显示任务框
        display_widget = QFrame(self)
        display_widget.setFrameShape(QFrame.StyledPanel)
        display_widget.setLineWidth(1)
        self.task3_label = QLabel(self)
        self.result_label = QLabel()
        display_layout = QVBoxLayout()
        display_layout.addWidget(self.result_label)
        display_widget.setLayout(display_layout)

        # 设置主布局
        main_layout = QHBoxLayout()
        main_layout.addWidget(operation_widget)
        main_layout.addWidget(pipeline_widget)
        main_layout.addWidget(display_widget)
        main_widget = QWidget()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

    def select_video_file(self):
        # 弹出文件选择
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_name, _ = QFileDialog.getOpenFileName(self, "选择视频文件", "", "视频文件 (*.mp4 *.avi);;All Files (*)",
                                                   options=options)
        if file_name:
            self.video_path = file_name
            # 从文件名中提取管道信息
            file_name = os.path.basename(file_name)
            file_parts = file_name.split('_')
            self.task_name_label.setText(f'任务名称：{file_parts[0]}')
            self.unit_label.setText(f'检测单位：{file_parts[1]}')
            self.inspector_label.setText(f'检测员：{file_parts[2].split(".")[0]}')

    def start_inference(self):
        # 启动模型推理线程
        self.inference_thread = InferenceThread(self.video_path)
        self.inference_thread.update_signal.connect(self.update_result)
        self.inference_thread.start()
        # 更新按钮状态
        self.start_btn.setDisabled(True)
        self.pause_btn.setDisabled(False)

    def pause_detection(self):
        exit()
    def update_result(self, result_img):
        # 在显示任务框中显示检测结果
        self.result_label.setPixmap(result_img)

    def closeEvent(self, event):
        # 关闭窗口前停止模型推理线程
        if hasattr(self, 'inference_thread') and self.inference_thread.isRunning():
            self.inference_thread.terminate()
        super().closeEvent(event)

if __name__ == '__main__':
    app = QApplication([])
    main_window = MainWindow()
    main_window.show()
    app.exec_()