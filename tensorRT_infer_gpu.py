from typing import Union, Optional, Sequence, Dict, Any
import tensorrt as trt
import torch
from torchvision import transforms
from PIL import Image
import time
import os
import argparse

# 定义命令行参数解析器
parse = argparse.ArgumentParser(description='trt model infer!')
parse.add_argument('--type', type=str, default='one_image', help='推理类型支持：one_image')  # 推理类型
parse.add_argument('--trt_path', type=str, default=r".\weights\best_cls.trt", help='trt存放路径')  # TensorRT 模型路径
parse.add_argument('--one_image_path', type=str, default=r'.\infer_data\00000081.png', help='单张图片路径')  # 推理图片路径
parse.add_argument('--device', type=str, default='cuda', help='默认设备GPU')  # 推理设备
parse.add_argument('--savepath', type=str, default=r'.\infer_results\result_pics', help='标记图片存储路径')  # 图片保存路径

# 定义分类标签
label_names = ["BX", "CJ", "CK", "CQ", "CR", "DK", "DP", "JB", "JG", "NO", "PL", "SG", "YW", "ZG", "ZW"]

# 定义 TensorRT 包装类
class TRTWrapper(torch.nn.Module):
    """
    TensorRT 推理包装类，用于加载 TensorRT 引擎并执行推理。
    """
    def __init__(self, engine: Union[str, trt.ICudaEngine],
                 output_names: Optional[Sequence[str]] = None) -> None:
        super().__init__()
        self.engine = engine
        # 如果引擎是路径字符串，则加载引擎
        if isinstance(self.engine, str):
            with trt.Logger() as logger, trt.Runtime(logger) as runtime:
                with open(self.engine, mode='rb') as f:
                    engine_bytes = f.read()
                self.engine = runtime.deserialize_cuda_engine(engine_bytes)
        # 创建执行上下文
        self.context = self.engine.create_execution_context()

        # 获取引擎的输入输出名称
        names = [_ for _ in self.engine]
        input_names = list(filter(self.engine.binding_is_input, names))
        self._input_names = input_names
        self._output_names = output_names

        # 如果未指定输出名称，则自动获取
        if self._output_names is None:
            output_names = list(set(names) - set(input_names))
            self._output_names = output_names

    def forward(self, inputs: Dict[str, torch.Tensor]):
        """
        执行推理。

        参数:
            inputs: 字典形式的输入张量，键为输入名称，值为输入张量。
        返回:
            outputs: 字典形式的输出张量，键为输出名称，值为输出张量。
        """
        # 检查输入输出名称是否存在
        assert self._input_names is not None
        assert self._output_names is not None

        # 初始化绑定数组
        bindings = [None] * (len(self._input_names) + len(self._output_names))
        profile_id = 0

        # 设置输入绑定
        for input_name, input_tensor in inputs.items():
            profile = self.engine.get_profile_shape(profile_id, input_name)
            assert input_tensor.dim() == len(profile[0]), 'Input dim is different from engine profile.'
            for s_min, s_input, s_max in zip(profile[0], input_tensor.shape, profile[2]):
                assert s_min <= s_input <= s_max, \
                    'Input shape should be between ' \
                    + f'{profile[0]} and {profile[2]}' \
                    + f' but get {tuple(input_tensor.shape)}.'
            idx = self.engine.get_binding_index(input_name)
            assert 'cuda' in input_tensor.device.type
            input_tensor = input_tensor.contiguous()
            if input_tensor.dtype == torch.long:
                input_tensor = input_tensor.int()
            self.context.set_binding_shape(idx, tuple(input_tensor.shape))
            bindings[idx] = input_tensor.contiguous().data_ptr()

        # 初始化输出张量并设置绑定
        outputs = {}
        for output_name in self._output_names:
            idx = self.engine.get_binding_index(output_name)
            dtype = torch.float32
            shape = tuple(self.context.get_binding_shape(idx))
            device = torch.device('cuda')
            output = torch.empty(size=shape, dtype=dtype, device=device)
            outputs[output_name] = output
            bindings[idx] = output.data_ptr()

        # 执行推理
        self.context.execute_async_v2(bindings, torch.cuda.current_stream().cuda_stream)
        return outputs

# 定义函数：使用 TensorRT 模型对单张图片进行推理
def trt_infer_one_image(args):
    """
    使用 TensorRT 模型对单张图片进行推理。

    参数:
        args: 命令行参数，包含模型路径、图片路径等信息。
    """
    img_path = args.one_image_path
    imgname = os.path.basename(img_path)

    # 定义图像预处理步骤
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.514, 0.450, 0.349], std=[0.211, 0.203, 0.164])
    ])

    # 加载并预处理图片
    image = Image.open(img_path)
    image_data = transform(image)
    image_data = torch.unsqueeze(image_data, dim=0)
    input_img = image_data.cuda()

    # 加载 TensorRT 模型
    eng = args.trt_path
    model = TRTWrapper(eng, ['output'])

    # 执行推理并记录时间
    start = time.time()
    output = model(dict(input=input_img))
    end = time.time()

    # 解析推理结果
    timecost = round(end - start, 10)
    out = torch.argmax(output['output'])
    result = label_names[int(out)]

    # 输出推理结果
    print(imgname + ':', result + '缺陷,' + ' 推理时间为：', timecost)

# 主函数：调用推理函数
if __name__ == "__main__":
    args = parse.parse_args()  # 解析命令行参数
    trt_infer_one_image(args)  # 调用单张图片推理函数
