import traceback
from ultralytics import YOLO

# ---------------- config ----------------
WEIGHTS_PTH = r'/hy-tmp/ultralytics-8.3.184/runs/detect/yolov11_drone_phaseC_optimized_v2/weights/best.pt'
DATA_YAML = r'/hy-tmp/ultralytics-8.3.184/data/data.yaml'

WORKERS = 16

# ---------- 阶段 C (最终优化版) ----------
IMGSZ_PHASE_C = 1024
EPOCHS_PHASE_C = 80  # 增加epoch数，给予更多时间突破
BATCH_PHASE_C = -1
RUN_NAME_PHASE_C = 'yolov11_drone_phaseC_final_v2'
# ----------------------------------------

def train_phase_c_final(model):
    """阶段 C 最终优化版: 针对0.774+的突破性优化"""
    try:
        kwargs = dict(
            data=DATA_YAML,
            epochs=EPOCHS_PHASE_C,  # 更改: 增加训练轮数至80
            imgsz=IMGSZ_PHASE_C,
            batch=BATCH_PHASE_C,
            workers=WORKERS,
            name=RUN_NAME_PHASE_C,
            val=True,
            
            # ========== 学习率策略调整 ==========
            lr0=0.00008,        # 更改: 进一步降低学习率
            lrf=0.01,           # 更改: 最终学习率为初始的1%
            cos_lr=True,
            warmup_epochs=5,    # 更改: 增加warmup轮数
            
            # ========== 优化器参数调整 ==========
            optimizer='SGD',
            momentum=0.85,       # 更改: 进一步降低动量
            weight_decay=0.00005, # 更改: 进一步降低权重衰减
            
            # ========== 正则化增强 ==========
            # 移除了不支持的 label_smoothing 参数
            
            # ========== 数据增强调整 ==========
            hsv_h=0.005,         # 更改: 极低HSV色调增强
            hsv_s=0.3,           # 更改: 极低HSV饱和度增强
            hsv_v=0.1,           # 更改: 极低HSV明度增强
            degrees=2.0,         # 更改: 极低旋转增强
            translate=0.05,      # 更改: 降低平移增强
            scale=0.3,           # 更改: 降低缩放增强
            shear=0.0,
            perspective=0.0,
            flipud=0.0,
            fliplr=0.2,          # 更改: 极低左右翻转概率
            mosaic=0.0,
            mixup=0.0,
            copy_paste=0.0,
            
            # ========== 损失函数权重调整 ==========
            # 基于各类别表现差异化调整
            box=5.5,            # 更改: 继续降低定位权重
            cls=1.5,            # 更改: 继续增加分类权重
            dfl=1.0,            # 更改: 降低DFL权重
            
            # ========== 正则化与早停 ==========
            patience=40,        # 更改: 大幅增加早停耐心值
            
            # ========== 设备设置 ==========
            device=0,
            
            # ========== 其他参数 ==========
            save=True,
            exist_ok=True,
            pretrained=True,
            resume=False,
            
            # ========== 验证参数调整 ==========
            conf=0.15,          # 更改: 进一步降低验证置信度阈值
            iou=0.65,           # 更改: 提高IoU阈值
            
            # ========== 焦点损失参数 ==========
            # 移除了不支持的 fl_gamma 参数
        )
        
        print(f"开始最终优化训练阶段 C: {RUN_NAME_PHASE_C}")
        print("优化策略: 极低学习率 + 损失权重调整 + 减少数据增强")
        print(f"重点关注: 提升people和motor类别性能，减少car/ship混淆")
        
        model.train(**kwargs)
        print("阶段 C 最终优化训练完成 ✅")
        
    except Exception:
        print("阶段 C 最终优化训练失败 ⚠️")
        traceback.print_exc()

def main():
    try:
        # 直接加载预训练模型 (当前mAP50=0.774的权重)
        model = YOLO(WEIGHTS_PTH)
        print(f"已加载预训练权重: {WEIGHTS_PTH}")
        print(f"当前模型mAP50: 0.774 (将在此基础上进行突破性优化)")

        # 运行最终优化版的阶段C
        train_phase_c_final(model)

        print("最终优化训练完成 🎉")
        
    except Exception:
        print("训练初始化失败：")
        traceback.print_exc()

if __name__ == '__main__':
    main()