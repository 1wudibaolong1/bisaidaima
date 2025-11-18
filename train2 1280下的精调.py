import traceback
from ultralytics import YOLO

# ---------------- config ----------------
WEIGHTS_PTH = r'/hy-tmp/ultralytics-8.3.184/runs/detect/yolov11_drone_phaseC_optimized_v2/weights/0.774.pt'
DATA_YAML = r'/hy-tmp/ultralytics-8.3.184/data/data.yaml'

WORKERS = 16

# ---------- 阶段 C (优化版) ----------
IMGSZ_PHASE_C = 1024
EPOCHS_PHASE_C = 100
BATCH_PHASE_C = -1
RUN_NAME_PHASE_C = 'yolov11_drone_phaseC_optimized_v4'
# ----------------------------------------

def train_phase_c_optimized(model):
    """阶段 C 优化版: 大分辨率精细化训练 (使用SGD优化器)"""
    try:
        kwargs = dict(
            data=DATA_YAML,
            epochs=EPOCHS_PHASE_C,
            imgsz=IMGSZ_PHASE_C,
            batch=BATCH_PHASE_C,
            workers=WORKERS,
            name=RUN_NAME_PHASE_C,
            val=True,
            # 学习率相关优化
            lr0=0.00025,        # 略微提高初始学习率
            lrf=0.06,           # 最终学习率为初始的5% (之前为1%，过于激进)
            # 优化器选择 - 使用SGD
            optimizer='SGD',
            momentum=0.937,
            weight_decay=0.0005,
            # 学习率调度策略
            cos_lr=True,
            warmup_epochs=5,    # 增加warmup轮数
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            # 数据增强调整
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=5.0,        # 轻微启用旋转增强(5度)
            translate=0.1,
            scale=0.5,
            shear=2.0,          # 轻微剪切变换(2度)
            perspective=0.0,
            flipud=0.0,
            fliplr=0.5,
            mosaic=0.3,         # 轻微启用马赛克增强(0.3)
            mixup=0.1,          # 轻微启用mixup增强(0.1)
            copy_paste=0.05,    # 轻微启用copy-paste增强(0.05)
            # 正则化与早停
            patience=10,        # 增加早停耐心值
            # 设备设置
            device=0,
            # 其他
            save=True,
            exist_ok=True,
            pretrained=True,
            resume=False,
            # 损失函数权重调整 - 针对类别混淆问题
            box=7.0,            # 略微降低定位权重(从7.5→7.0)
            cls=0.8,            # 增加分类权重(从0.5→0.8)
            dfl=1.5,            # 保持dfl权重不变
            # 验证参数调整
            conf=0.25,          # 验证时使用较低的置信度阈值
            iou=0.55,           # 调整IoU阈值
        )
        print(f"开始优化训练阶段 C (SGD): {RUN_NAME_PHASE_C}")
        print("优化策略: 调整损失权重(box:7.0, cls:0.8) + 学习率优化 + 数据增强调整")
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

        # 运行优化后的阶段C
        train_phase_c_optimized(model)

        print("训练完成 🎉")
    except Exception:
        print("训练初始化失败：")
        traceback.print_exc()

if __name__ == '__main__':
    main()