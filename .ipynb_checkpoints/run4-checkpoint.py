import traceback
from ultralytics import YOLO

# ---------------- config ----------------
WEIGHTS_PTH = r'/hy-tmp/ultralytics-8.3.184/runs/detect/yolov11_drone_phaseC3/weights/best.pt'
DATA_YAML = r'/hy-tmp/ultralytics-8.3.184/data/data/data.yaml'

WORKERS = 16

# ---------- 阶段 C (优化版 v5) ----------
IMGSZ_PHASE_C = 1024
EPOCHS_PHASE_C = 150  # 增加 epochs 以允许更长时间收敛
BATCH_PHASE_C = -1
RUN_NAME_PHASE_C = 'yolov11_drone_phaseC_optimized_v7'
# ----------------------------------------

def train_phase_c_optimized(model):
    """阶段 C 优化版 v5: 进一步优化以提升 mAP50 (针对分类混淆和小物体检测)"""
    try:
        kwargs = dict(
            data=DATA_YAML,
            epochs=EPOCHS_PHASE_C,
            imgsz=IMGSZ_PHASE_C,
            batch=BATCH_PHASE_C,
            workers=WORKERS,
            name=RUN_NAME_PHASE_C,
            val=True,
            # 学习率相关优化 - 降低初始 lr 以更平稳训练
            lr0=0.0003,
            lrf=0.05,           # 最终学习率为初始的5%
            # 优化器选择 - 继续使用 SGD 但调整 momentum
            optimizer='SGD',
            momentum=0.95,      # 增加 momentum 以加速收敛
            weight_decay=0.0005,
            # 学习率调度策略
            cos_lr=True,
            warmup_epochs=7,    # 进一步增加 warmup 轮数以避免早期过拟合
            warmup_momentum=0.85,
            warmup_bias_lr=0.1,
            # 数据增强调整 - 加强针对小物体和分类混淆的增强
            hsv_h=0.02,         # 略微增加颜色扰动
            hsv_s=0.75,
            hsv_v=0.45,
            degrees=10.0,       # 增加旋转增强以处理无人机视角变化
            translate=0.15,     # 增加平移以模拟物体位置变化
            scale=0.6,          # 增加缩放以处理尺度变化
            shear=5.0,          # 增加剪切以模拟视角扭曲
            perspective=0.001,  # 轻微启用透视变换以模拟高空视角
            flipud=0.2,         # 启用上下翻转以增加多样性
            fliplr=0.5,
            mosaic=0.5,         # 增加 mosaic 强度以融合更多上下文
            mixup=0.2,          # 增加 mixup 以改善分类边界
            copy_paste=0.1,     # 增加 copy-paste 以处理类不平衡
            # 正则化与早停
            patience=15,        # 增加耐心值以允许更长训练
            label_smoothing=0.1,# 添加标签平滑以减少过自信预测
            # 设备设置
            device=0,
            # 其他
            save=True,
            exist_ok=True,
            pretrained=True,
            resume=False,
            multi_scale=True,   # 启用多尺度训练以更好地处理不同大小物体
            # 损失函数权重调整 - 进一步强调分类，针对混淆矩阵中的问题
            box=6.5,            # 略微降低 box 权重
            cls=1.0,            # 增加 cls 权重以改善分类准确性
            dfl=1.5,            # 保持 dfl 权重
            # 验证参数调整
            conf=0.2,           # 降低验证 conf 阈值以捕捉更多潜在正例
            iou=0.5,            # 调整 IoU 以匹配 mAP50 焦点
        )
        print(f"开始优化训练阶段 C (SGD v5): {RUN_NAME_PHASE_C}")
        print("优化策略: 加强数据增强 + 标签平滑 + 多尺度训练 + 调整损失权重(box:6.5, cls:1.0) + 学习率细调")
        model.train(**kwargs)
        print("阶段 C 优化训练完成 ✅")
    except Exception:
        print("阶段 C 优化训练失败 ⚠️")
        traceback.print_exc()

def main():
    try:
        # 直接加载预训练模型
        model = YOLO(WEIGHTS_PTH)
        print(f"已加载预训练权重: {WEIGHTS_PTH}")

        # 运行优化后的阶段C v5
        train_phase_c_optimized(model)

        print("训练完成 🎉")
    except Exception:
        print("训练初始化失败：")
        traceback.print_exc()

if __name__ == '__main__':
    main()