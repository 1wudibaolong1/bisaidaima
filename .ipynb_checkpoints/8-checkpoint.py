import torch
import yaml
from ultralytics import YOLO
import os

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def transfer_weights(s_model_path, x_config_path, output_path):
    """
    将预训练的 YOLOv11s 权重转移到 YOLOv11x 架构
    
    参数:
        s_model_path: 预训练的 YOLOv11s 模型路径
        x_config_path: YOLOv11x 配置文件路径
        output_path: 输出模型路径
    """
    print("正在加载模型和配置...")
    
    # 加载预训练的 YOLOv11s 模型 - 修复 PyTorch 2.6 兼容性问题
    try:
        # 首先尝试使用 weights_only=True (PyTorch 2.6+ 默认)
        s_model = torch.load(s_model_path, weights_only=True)
    except:
        # 如果失败，回退到 weights_only=False (PyTorch 2.6 之前的默认行为)
        print("使用 weights_only=True 加载失败，尝试使用 weights_only=False...")
        s_model = torch.load(s_model_path, weights_only=False)
    
    s_state_dict = s_model['model'].state_dict() if 'model' in s_model else s_model.state_dict()
    
    # 根据 x 网络的配置文件创建一个新模型
    x_model = YOLO(x_config_path).model
    
    # 获取新模型的状态字典
    x_state_dict = x_model.state_dict()
    
    print("正在转移权重...")
    # 筛选预训练权重：只转移层名相同且维度匹配的权重
    transferred_dict = {}
    matched_layers, unmatched_layers = 0, 0
    
    for k, v in s_state_dict.items():
        if k in x_state_dict and v.shape == x_state_dict[k].shape:
            transferred_dict[k] = v
            matched_layers += 1
        else:
            unmatched_layers += 1
    
    print(f"权重转移完成: 匹配 {matched_layers} 层, 不匹配 {unmatched_layers} 层")
    
    # 更新新模型的状态字典
    x_state_dict.update(transferred_dict)
    x_model.load_state_dict(x_state_dict)
    
    # 保存嫁接后的模型
    torch.save({'model': x_model}, output_path)
    print(f"新模型已保存至: {output_path}")
    
    return output_path

def main():
    # 加载配置文件
    config_path = "finetune_config.yaml"
    if not os.path.exists(config_path):
        # 如果配置文件不存在，创建一个默认的
        create_default_config(config_path)
        print(f"已创建默认配置文件: {config_path}")
        print("请编辑此文件后重新运行脚本")
        return
    
    config = load_config(config_path)
    
    # 创建输出目录
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # 第一步：权重转移
    transferred_model_path = os.path.join(config['output_dir'], 'yolov11x_pretrained.pt')
    transfer_weights(config['s_model'], config['x_config'], transferred_model_path)
    
    # 第二步：开始训练
    print("开始微调训练...")
    
    # 加载模型
    model = YOLO(transferred_model_path)
    
    # 准备训练参数
    train_args = {
        'data': config['data'],
        'epochs': config['epochs'],
        'patience': config['patience'],
        'batch': config['batch'],
        'imgsz': config['imgsz'],
        'optimizer': 'SGD',
        'lr0': config['lr0'],
        'lrf': config['lrf'],
        'momentum': config['momentum'],
        'weight_decay': config['weight_decay'],
        'warmup_epochs': config['warmup_epochs'],
        'warmup_momentum': config['warmup_momentum'],
        'warmup_bias_lr': config['warmup_bias_lr'],
        'box': config['box'],
        'cls': config['cls'],
        'dfl': config['dfl'],
        'hsv_h': config['hsv_h'],
        'hsv_s': config['hsv_s'],
        'hsv_v': config['hsv_v'],
        'degrees': config['degrees'],
        'translate': config['translate'],
        'scale': config['scale'],
        'shear': config['shear'],
        'perspective': config['perspective'],
        'flipud': config['flipud'],
        'fliplr': config['fliplr'],
        'mosaic': config['mosaic'],
        'mixup': config['mixup'],
        'copy_paste': config['copy_paste'],
        'name': config['project_name'],
        'project': config['output_dir']
    }
    
    # 开始训练
    results = model.train(**train_args)
    
    print("训练完成!")
    print(f"最佳模型保存在: {config['output_dir']}/{config['project_name']}")

def create_default_config(config_path):
    """创建默认配置文件"""
    default_config = {
        # 必需参数
        's_model': '/path/to/your/yolov11s.pt',
        'x_config': '/path/to/your/yolov11x.yaml',
        'data': '/path/to/your/dataset.yaml',
        
        # 输出目录
        'output_dir': './runs/finetune',
        
        # 训练参数
        'epochs': 50,
        'batch': 16,
        'imgsz': 640,
        'patience': 15,
        'project_name': 'yolov11s_to_x_finetune',
        
        # 优化器参数
        'lr0': 0.0001,
        'lrf': 0.01,
        'momentum': 0.9,
        'weight_decay': 0.0005,
        'warmup_epochs': 3.0,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        
        # 损失权重
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        
        # 数据增强参数
        'hsv_h': 0.015,
        'hsv_s': 0.7,
        'hsv_v': 0.4,
        'degrees': 5.0,
        'translate': 0.1,
        'scale': 0.5,
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.5,
        'mosaic': 1.0,
        'mixup': 0.0,
        'copy_paste': 0.0
    }
    
    with open(config_path, 'w') as f:
        yaml.dump(default_config, f, default_flow_style=False)

if __name__ == '__main__':
    main()