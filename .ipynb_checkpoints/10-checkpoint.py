import torch
from ultralytics import YOLO
import os

def train_phase_c(model_path, data_yaml, output_dir="./runs/finetune"):
    """
    阶段 C: 大分辨率精细化训练
    使用预训练模型进行高分辨率微调
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载预训练模型
    model = YOLO(model_path)
    
    # 训练参数
    kwargs = {
        "data": data_yaml,
        "epochs": 500,           # 训练轮数
        "imgsz": 896,            # 输入图像尺寸
        "batch": 4,              # 批次大小 (根据GPU内存调整)
        "workers": 16,            # 数据加载工作进程数
        "name": "yolov11s_896_finetune",  # 运行名称
        "val": True,             # 启用验证
        "lr0": 0.0001,            # 初始学习率
        "device": 0,             # 使用GPU 0
        "patience": 20,          # 早停耐心值
        "cos_lr": True,          # 使用余弦学习率衰减
        "warmup_epochs": 5,      # 热身轮数
        "warmup_momentum": 0.8,  # 热身动量
        "warmup_bias_lr": 0.1,   # 热身偏置学习率
        "amp": True,             # 启用自动混合精度
        # "accumulate": 2,       # 梯度累积 (如显存不足再启用)
    }
    
    print(f"开始高分辨率微调训练: {kwargs['name']}")
    print(f"分辨率: {kwargs['imgsz']}x{kwargs['imgsz']}")
    print(f"学习率: {kwargs['lr0']}")
    print(f"批次大小: {kwargs['batch']}")
    
    # 开始训练
    results = model.train(**kwargs)
    
    print("高分辨率微调训练完成 ✅")
    return results

if __name__ == "__main__":
    # 设置路径参数
    PRETRAINED_MODEL = "/hy-tmp/runs/detect/yolov11_drone_phaseC 0.725/weights/best.pt"  # 您已经训练好的权重
    DATA_YAML = "/hy-tmp/ultralytics-8.3.184/data/data/data.yaml"    # 数据集配置文件
    
    # 运行训练
    train_phase_c(PRETRAINED_MODEL, DATA_YAML)