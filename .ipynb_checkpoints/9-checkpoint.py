import torch
import yaml
from ultralytics import YOLO
import os
import argparse
import numpy as np

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def transfer_weights_with_stats(pretrained_path, target_model):
    """
    加载预训练权重并显示迁移统计信息
    
    参数:
        pretrained_path: 预训练模型路径
        target_model: 目标模型
    
    返回:
        更新后的模型和迁移统计信息
    """
    print(f"正在加载预训练权重: {pretrained_path}")
    
    # 加载预训练模型
    try:
        # 首先尝试使用 weights_only=True (PyTorch 2.6+ 默认)
        pretrained_model = torch.load(pretrained_path, weights_only=True)
    except:
        # 如果失败，回退到 weights_only=False (PyTorch 2.6 之前的默认行为)
        print("使用 weights_only=True 加载失败，尝试使用 weights_only=False...")
        pretrained_model = torch.load(pretrained_path, weights_only=False)
    
    # 获取预训练模型的状态字典
    if 'model' in pretrained_model:
        pretrained_state_dict = pretrained_model['model'].state_dict()
    else:
        pretrained_state_dict = pretrained_model.state_dict()
    
    # 获取目标模型的状态字典
    target_state_dict = target_model.state_dict()
    
    print("正在分析权重迁移...")
    
    # 统计信息
    total_params = 0
    transferred_params = 0
    matched_layers = 0
    unmatched_layers = 0
    layer_stats = []
    
    # 遍历目标模型的每一层
    for name, param in target_state_dict.items():
        total_params += param.numel()
        
        # 检查预训练模型中是否有对应的层
        if name in pretrained_state_dict:
            # 检查形状是否匹配
            if pretrained_state_dict[name].shape == param.shape:
                # 形状匹配，可以迁移权重
                target_state_dict[name] = pretrained_state_dict[name]
                transferred_params += param.numel()
                matched_layers += 1
                layer_stats.append((name, "匹配", f"{param.numel():,}"))
            else:
                # 形状不匹配，无法迁移
                unmatched_layers += 1
                layer_stats.append((name, "形状不匹配", 
                                   f"预训练: {pretrained_state_dict[name].shape}, 目标: {param.shape}"))
        else:
            # 预训练模型中不存在该层
            unmatched_layers += 1
            layer_stats.append((name, "不存在于预训练模型", ""))
    
    # 打印详细的层迁移信息
    print("\n=== 权重迁移详情 ===")
    for name, status, info in layer_stats:
        print(f"{name:60} {status:20} {info}")
    
    # 打印总体统计信息
    print("\n=== 权重迁移统计 ===")
    print(f"匹配层数: {matched_layers}")
    print(f"不匹配层数: {unmatched_layers}")
    print(f"总参数量: {total_params:,}")
    print(f"迁移参数量: {transferred_params:,}")
    print(f"迁移比例: {transferred_params/total_params*100:.2f}%")
    
    # 更新模型状态字典
    target_model.load_state_dict(target_state_dict)
    
    return target_model, {
        'matched_layers': matched_layers,
        'unmatched_layers': unmatched_layers,
        'total_params': total_params,
        'transferred_params': transferred_params,
        'transfer_ratio': transferred_params/total_params
    }

def main():
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='微调 YOLOv11s 模型')
    parser.add_argument('--config', type=str, default='finetune_config.yaml', help='配置文件路径')
    args = parser.parse_args()
    
    # 加载配置文件
    config_path = args.config
    if not os.path.exists(config_path):
        # 如果配置文件不存在，创建一个默认的
        create_default_config(config_path)
        print(f"已创建默认配置文件: {config_path}")
        print("请编辑此文件后重新运行脚本")
        return
    
    config = load_config(config_path)
    
    # 创建输出目录
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # 创建新模型并加载预训练权重
    print(f"创建新模型并加载预训练权重: {config['pretrained_model']}")
    model = YOLO(config['model_config']).model  # 创建新模型
    
    # 迁移权重并显示统计信息
    model, stats = transfer_weights_with_stats(config['pretrained_model'], model)
    
    # 保存迁移后的模型
    transferred_model_path = os.path.join(config['output_dir'], 'transferred_model.pt')
    torch.save({'model': model}, transferred_model_path)
    print(f"\n迁移后的模型已保存至: {transferred_model_path}")
    
    # 准备训练参数 - 使用较小的学习率进行微调
    train_args = {
        'data': config['data'],
        'epochs': config['epochs'],
        'patience': config['patience'],
        'batch': config['batch'],
        'imgsz': config['imgsz'],
        'optimizer': 'SGD',
        'lr0': config['lr0'],  # 使用较小的学习率
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
        'project': config['output_dir'],
        'resume': False,
        'device': config.get('device', '0'),
        'workers': config.get('workers', 8),
        'close_mosaic': config.get('close_mosaic', 10),
        'save_period': config.get('save_period', -1),
        'single_cls': config.get('single_cls', False),
        'amp': config.get('amp', True),
    }
    
    # 开始训练
    print("\n开始微调训练...")
    model_wrapper = YOLO(transferred_model_path)
    results = model_wrapper.train(**train_args)
    
    print("训练完成!")
    print(f"最佳模型保存在: {config['output_dir']}/{config['project_name']}")
    
    # 验证最佳模型
    best_model_path = f"{config['output_dir']}/{config['project_name']}/weights/best.pt"
    if os.path.exists(best_model_path):
        print("验证最佳模型...")
        metrics = model_wrapper.val(
            data=config['data'],
            batch=config['batch'],
            imgsz=config['imgsz'],
            conf=0.001,
            iou=0.6,
            device=config.get('device', '0'),
            split='val'
        )
        print(f"最佳模型 mAP50: {metrics.box.map50}")

def create_default_config(config_path):
    """创建默认配置文件"""
    default_config = {
        # 必需参数
        'pretrained_model': '/path/to/your/best.pt',  # 您已经训练好的权重
        'model_config': 'yolov11s.yaml',  # 模型配置文件
        'data': '/path/to/your/dataset.yaml',
        
        # 输出目录
        'output_dir': './runs/finetune_s',
        
        # 训练参数
        'epochs': 100,
        'batch': 16,
        'imgsz': 640,
        'patience': 30,
        'project_name': 'yolov11s_finetune',
        
        # 优化器参数 - 使用较小的学习率
        'lr0': 0.00001,
        'lrf': 0.01,
        'momentum': 0.9,
        'weight_decay': 0.0005,
        'warmup_epochs': 5.0,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        
        # 损失权重
        'box': 7.5,
        'cls': 0.5,
        'dfl': 1.5,
        
        # 数据增强参数
        'hsv_h': 0.01,
        'hsv_s': 0.5,
        'hsv_v': 0.3,
        'degrees': 2.0,
        'translate': 0.05,
        'scale': 0.3,
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.3,
        'mosaic': 0.8,
        'mixup': 0.0,
        'copy_paste': 0.0,
        
        # 其他参数
        'device': '0',
        'workers': 8,
        'close_mosaic': 10,
        'amp': True,
    }
    
    with open(config_path, 'w') as f:
        yaml.dump(default_config, f, default_flow_style=False)

if __name__ == '__main__':
    main()