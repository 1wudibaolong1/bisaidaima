import traceback
from ultralytics import YOLO

# ---------------- config ----------------
MODEL_YAML = r'/hy-tmp/ultralytics-8.3.184/ultralytics/cfg/models/11/yolo11s.yaml'
WEIGHTS_PTH = r'/hy-tmp/ultralytics-8.3.184/runs/detect/yolov11_drone_phaseA3/weights/best.pt'
DATA_YAML = r'/hy-tmp/ultralytics-8.3.184/data/data/data.yaml'

WORKERS = 16

# ---------- 阶段 A ----------
IMGSZ_PHASE_A = 512
EPOCHS_PHASE_A = 100
BATCH_PHASE_A = 20
RUN_NAME_PHASE_A = 'yolov11_drone_phaseA'

# ---------- 阶段 B ----------
IMGSZ_PHASE_B = 640
EPOCHS_PHASE_B = 150
BATCH_PHASE_B = 6
RUN_NAME_PHASE_B = 'yolov11_drone_phaseB'

# ---------- 阶段 C ----------
IMGSZ_PHASE_C = 896
EPOCHS_PHASE_C = 200
BATCH_PHASE_C = 4
RUN_NAME_PHASE_C = 'yolov11_drone_phaseC'
# ----------------------------------------


def train_phase_a(model):
    """阶段 A: 冻结 backbone 前几层"""
    try:
        kwargs = dict(
            data=DATA_YAML,
            epochs=EPOCHS_PHASE_A,
            imgsz=IMGSZ_PHASE_A,
            batch=BATCH_PHASE_A,
            workers=WORKERS,
            name=RUN_NAME_PHASE_A,
            val=True,
            lr0=1e-3,
            freeze=14,   # 冻结前几层
            device=0,
            patience=20,
        )
        print(f"开始训练阶段 A: {RUN_NAME_PHASE_A}")
        model.train(**kwargs)
        print("阶段 A 训练完成 ✅")
    except Exception:
        print("阶段 A 训练失败 ⚠️")
        traceback.print_exc()


def train_phase_b(model):
    """阶段 B: 解冻 backbone，全网微调"""
    try:
        kwargs = dict(
            data=DATA_YAML,
            epochs=EPOCHS_PHASE_B,
            imgsz=IMGSZ_PHASE_B,
            batch=BATCH_PHASE_B,
            workers=WORKERS,
            name=RUN_NAME_PHASE_B,
            val=True,
            device=0,
            lr0=1e-3,
            patience=20,
        )
        print(f"开始训练阶段 B: {RUN_NAME_PHASE_B}")
        model.train(**kwargs)
        print("阶段 B 训练完成 ✅")
    except Exception:
        print("阶段 B 训练失败 ⚠️")
        traceback.print_exc()


def train_phase_c(model):
    """阶段 C: 大分辨率精细化，可关闭强增广"""
    try:
        kwargs = dict(
            data=DATA_YAML,
            epochs=EPOCHS_PHASE_C,
            imgsz=IMGSZ_PHASE_C,
            batch=BATCH_PHASE_C,
            workers=WORKERS,
            name=RUN_NAME_PHASE_C,
            val=True,
            lr0=5e-4,
            device=0,
            patience=20,
            # accumulate=2, # 如显存不足再启用梯度累积
        )
        print(f"开始训练阶段 C: {RUN_NAME_PHASE_C}")
        model.train(**kwargs)
        print("阶段 C 训练完成 ✅")
    except Exception:
        print("阶段 C 训练失败 ⚠️")
        traceback.print_exc()


def main():
    try:
        # 1) 构造模型
        model = YOLO(MODEL_YAML)
        print(f"模型架构加载成功: {MODEL_YAML}")

        # 2) 加载预训练权重
        try:
            model.load(WEIGHTS_PTH)
            print(f"已加载预训练权重: {WEIGHTS_PTH}")
        except Exception:
            print("⚠️ 加载预训练权重失败，继续使用随机初始化权重")
            traceback.print_exc()

        # 2) 三个阶段训练
        #train_phase_a(model)
        train_phase_b(model)
        train_phase_c(model)

        print("训练完成 🎉")
    except Exception:
        print("训练初始化失败：")
        traceback.print_exc()


if __name__ == '__main__':
    main()
