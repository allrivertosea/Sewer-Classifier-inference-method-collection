# Sewer-Classifier-inference-method-collection
The collection of inference methods for sewer defect classification using various model formats.

## 功能说明

- pth模型直接推理
- Flask API模型推理
- Openvino模型推理
- tensorrt模型推理
- tf_serving+docker模型推理

## 分类效果

![功能测试](https://github.com/allrivertosea/Sewer-Classifier-inference-method-collection/blob/main/infer_results/result_pics/00000001.png)


## 环境配置

conda create -n infer_env python=3.8 -y

conda activate infer_env

pip install -r requirements.txt

## 使用说明

```
python xxx.py   #执行推理操作
```


