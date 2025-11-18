import traceback
from ultralytics import YOLO
import os

# ---------------- config ----------------
# 注意：此处应改为你阶段C中断时的检查点路径（通常在runs/detect/[你的阶段C目录]/weights/last.pt）
RESUME_WEIGHTS = r'/hy-tmp/ultralytics-8.3.184/runs/detect/yolov11_drone_phaseC5/weights/last.pt'  # 关键：指向阶段C中断的检查点
DATA_YAML = r'/hy-tmp/ultralytics-8.3.184/data/data/data.yaml'

WORKERS = 16

# ---------- 阶段 C (断点续训) ----------
IMGSZ_PHASE_C = 1024
# 总训练轮次保持不变，YOLO会自动计算剩余轮次（例如之前训练了30轮，这里100的话会再训练70轮）
EPOCHS_PHASE_C = 100
BATCH_PHASE_C = -1  # 自动计算批次大小，若显存不足可改为具体数值（如2、4）
RUN_NAME_PHASE_C = 'yolov11_drone_phaseC'  # 保持与之前相同的名称，结果会保存在同一目录
# ----------------------------------------


def train_phase_c_resume():
    """从阶段C断点继续训练：加载完整训练状态（权重+优化器+学习率调度器）"""
    try:
        # 直接从检查点加载模型（包含完整训练状态）
        model = YOLO(RESUME_WEIGHTS)
        print(f"已从检查点加载模型和训练状态: {RESUME_WEIGHTS}")

        kwargs = dict(
            data=DATA_YAML,
            epochs=EPOCHS_PHASE_C,
            imgsz=IMGSZ_PHASE_C,
            batch=BATCH_PHASE_C,
            workers=WORKERS,
            name=RUN_NAME_PHASE_C,  # 保持名称一致，确保结果保存到同一目录
            val=True,
            lr0=0.00025,  # 保持与之前相同的学习率
            device=0,
            patience=10,
            resume=RESUME_WEIGHTS,  # 明确指定要恢复的检查点路径
            
            # accumulate=2,  # 若显存不足，取消注释启用梯度累积
        )

        print(f"开始从断点继续训练阶段 C: {RUN_NAME_PHASE_C}")
        model.train(** kwargs)
        print("阶段 C 训练完成 ✅")

    except Exception:
        print("阶段 C 断点续训失败 ⚠️")
        traceback.print_exc()


def main():
    # 验证检查点文件是否存在
    if not os.path.exists(RESUME_WEIGHTS):
        print(f"错误：检查点文件不存在 - {RESUME_WEIGHTS}")
        print("请确认RESUME_WEIGHTS路径是否正确")
        return

    # 仅运行阶段C的断点续训
    train_phase_c_resume()
    print("训练完成 🎉")


if __name__ == '__main__':
    main()
