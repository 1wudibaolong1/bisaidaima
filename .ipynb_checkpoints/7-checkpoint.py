"""
simple_train_yolov11s.py

一个极简的 YOLOv11s 训练脚本，适合直接用于微调/训练（假定数据已按 YOLO 格式准备并有一个 data YAML）。
添加了早停机制（patience），当验证集指标在连续指定轮次内无提升时自动停止训练。

用法示例：
  python simple_train_yolov11s.py --data dataset/drone_data.yaml --epochs 100 --imgsz 1024 --batch 8 --pretrained yolo11s.pt --device 0 --patience 50

requirements (建议写入 requirements.txt)：
  ultralytics
  torch
  torchvision
  pyyaml
  opencv-python

注意：该脚本只保留最基本流程，便于快速上手。如需切片（tiling）、复杂增强或自定义超参请告知。
"""

import argparse
from pathlib import Path

try:
    from ultralytics import YOLO
except Exception:
    raise ImportError('请先安装 ultralytics（pip install ultralytics）')


def main():
    parser = argparse.ArgumentParser(description='简单的 YOLOv11s 训练脚本')
    parser.add_argument('--data', type=str, default='/hy-tmp/ultralytics-8.3.184/ultralytics-8.3.184/data/data.yaml', help='数据 YAML 文件路径（YOLO 格式）')
    parser.add_argument('--epochs', type=int, default=10000, help='最大训练轮数')
    parser.add_argument('--imgsz', type=int, default=640, help='输入尺寸')
    parser.add_argument('--batch', type=int, default=32, help='batch size')
    parser.add_argument('--pretrained', type=str, default='/hy-tmp/ultralytics-8.3.184/ultralytics-8.3.184/yolo11s.pt', help='预训练权重或模型路径')
    parser.add_argument('--device', type=str, default='0', help='训练设备，例如 "0" 或 "cpu"')
    parser.add_argument('--project', type=str, default='runs', help='保存结果的目录')
    parser.add_argument('--name', type=str, default='exp', help='实验名')
    parser.add_argument('--patience', type=int, default=100, help='早停耐心值：连续多少轮验证集指标无提升则停止训练（0表示禁用早停）')
    args = parser.parse_args()

    data_yaml = Path(args.data)
    if not data_yaml.exists():
        raise FileNotFoundError(f"找不到 data YAML: {data_yaml}. 请确保路径正确并且文件存在。")

    print('加载模型：', args.pretrained)
    model = YOLO(args.pretrained)

    print('开始训练...')
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        exist_ok=True,
        patience=args.patience,  # 添加早停参数
    )

    print('训练完成，查看结果目录：', args.project)


if __name__ == '__main__':
    main()

