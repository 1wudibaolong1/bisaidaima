import os
import traceback
from ultralytics import YOLO

# ---------------- config ----------------
MODEL_YAML = r'/hy-tmp/ultralytics-8.3.184/ultralytics-8.3.184/ultralytics/cfg/models/11/yolo11s.yaml'
WEIGHTS_PTH = r'/hy-tmp/ultralytics-8.3.184/ultralytics-8.3.184/yolo11s.pt'
DATA_YAML = r'/hy-tmp/ultralytics-8.3.184/ultralytics-8.3.184/data/data.yaml'

WORKERS = 16

# ---------- 阶段 A ----------
IMGSZ_PHASE_A = 512
EPOCHS_PHASE_A = 60
BATCH_PHASE_A = 8
RUN_NAME_PHASE_A = 'yolov11_drone_phaseA'

# ---------- 阶段 B ----------
IMGSZ_PHASE_B = 640
EPOCHS_PHASE_B = 120
BATCH_PHASE_B = 6
RUN_NAME_PHASE_B = 'yolov11_drone_phaseB'

# ---------- 阶段 C ----------
IMGSZ_PHASE_C = 896
EPOCHS_PHASE_C = 80
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
            lr0=1e-3,
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
            # accumulate=2, # 如显存不足再启用梯度累积
        )
        print(f"开始训练阶段 C: {RUN_NAME_PHASE_C}")
        model.train(**kwargs)
        print("阶段 C 训练完成 ✅")
    except Exception:
        print("阶段 C 训练失败 ⚠️")
        traceback.print_exc()


def main():
    """运行阶段 C 或从用户指定的断点权重恢复训练。

    用法示例：
      # 从指定的 last.pt 恢复训练
      python train_yolov11_only_phaseC_resume.py --last_path /path/to/runs/detect/yolov11_drone_phaseC/weights/last.pt

      # 不提供 last_path 时，按原逻辑构建模型并只运行阶段 C（可加载预训练权重）
      python train_yolov11_only_phaseC_resume.py
    """
    import argparse

    parser = argparse.ArgumentParser(description='只运行或恢复 YOLOv11 阶段 C 训练')
    parser.add_argument('--last_path', type=str, default=r'/hy-tmp/runs/detect/yolov11_drone_phaseC/weights/last.pt',
                        help='用户指定的断点权重路径（可选）。示例: runs/detect/.../weights/last.pt')
    args = parser.parse_args()

    last_path = args.last_path

    try:
        if last_path:
            # 用户提供 path，直接使用并 resume
            if os.path.exists(last_path):
                try:
                    print(f"使用用户提供的断点权重：{last_path}，将从断点继续训练（resume=True）")
                    model = YOLO(last_path)
                    model.train(resume=True)
                    print("已从断点继续训练 ✅")
                    return
                except Exception:
                    print("从指定断点恢复训练失败，尝试备用流程...")
                    traceback.print_exc()
            else:
                print(f"指定的断点权重不存在：{last_path}，将按原逻辑继续。")

        # 如果没有提供 last_path 或恢复失败，按原逻辑构建模型并只运行阶段 C
        model = YOLO(MODEL_YAML)
        print(f"模型架构加载成功: {MODEL_YAML}")

        # 尝试加载预训练权重（可选）
        try:
            model.load(WEIGHTS_PTH)
            print(f"已加载预训练权重: {WEIGHTS_PTH}")
        except Exception:
            print("⚠️ 加载预训练权重失败，使用随机初始化权重或已加载的权重")
            traceback.print_exc()

        # 只运行阶段 C
        train_phase_b(model)
        train_phase_c(model)

        print("训练完成 🎉")
    except Exception:
        print("训练初始化失败：")
        traceback.print_exc()


if __name__ == '__main__':
    main()
