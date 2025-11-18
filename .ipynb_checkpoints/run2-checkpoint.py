import traceback
from ultralytics import YOLO
import os

# ---------------- config ----------------
WEIGHTS_PTH = r'/hy-tmp/ultralytics-8.3.184/runs/detect/yolov11_drone_phaseC_optimized_v4/weights/best.pt'
DATA_YAML = r'/hy-tmp/ultralytics-8.3.184/data/data/data.yaml'

WORKERS = 16

# ---------- 阶段 C (优化改进版) ----------
IMGSZ_PHASE_C = 1024
EPOCHS_PHASE_C = 120  # 增加训练轮数
BATCH_PHASE_C = -1
RUN_NAME_PHASE_C = 'yolov11_drone_phaseC_optimized_v51'
# ----------------------------------------

def train_phase_c_optimized_v2(model):
    """阶段 C 优化改进版: 针对mAP50提升的专项优化"""
    try:
        kwargs = dict(
            data=DATA_YAML,
            epochs=EPOCHS_PHASE_C,
            imgsz=IMGSZ_PHASE_C,
            batch=BATCH_PHASE_C,
            workers=WORKERS,
            name=RUN_NAME_PHASE_C,
            val=True,
            
            # 🔥 学习率策略重大调整
            lr0=0.001,          # 大幅提高初始学习率 (从0.00035→0.001)
            lrf=0.01,           # 更平缓的衰减 (从0.06→0.01)
            
            # 优化器设置
            optimizer='SGD',
            momentum=0.937,
            weight_decay=0.0005,
            
            # 📈 学习率调度优化
            cos_lr=True,
            warmup_epochs=8,    # 延长warmup (从5→8)
            warmup_momentum=0.9,
            warmup_bias_lr=0.15,
            
            # 🎯 数据增强大幅增强 - 针对无人机小目标
            hsv_h=0.02,         # 增强色调变化
            hsv_s=0.8,          # 增强饱和度变化  
            hsv_v=0.5,          # 增强亮度变化
            degrees=8.0,        # 增强旋转增强 (从5→8)
            translate=0.15,     # 增强平移
            scale=0.8,          # 增强尺度变化 (从0.5→0.8)
            shear=3.0,          # 增强剪切变换 (从2→3)
            perspective=0.001,  # 轻微透视变换
            flipud=0.2,         # 启用上下翻转 (无人机视角常见)
            fliplr=0.5,
            
            # 🚀 高级数据增强策略
            mosaic=0.5,         # 增强马赛克 (从0.3→0.5)
            mixup=0.2,          # 增强mixup (从0.1→0.2)
            copy_paste=0.1,     # 增强copy-paste (从0.05→0.1)
            auto_augment='randaugment',  # 启用自动增强
            erasing=0.4,        # 随机擦除
            
            # ⚖️ 损失函数权重重新平衡 - 针对类别不平衡
            box=6.0,            # 降低定位权重 (从7.0→6.0)
            cls=1.2,            # 大幅提高分类权重 (从0.8→1.2)
            dfl=1.5,            # 保持DFL权重
            
            # 🎯 验证和早停优化
            conf=0.2,           # 降低验证置信度阈值 (从0.25→0.2)
            iou=0.6,            # 调整IoU阈值 (从0.55→0.6)
            patience=15,        # 延长早停耐心 (从10→15)
            
            # 📊 模型保存策略
            save=True,
            save_period=10,     # 每10轮保存一次
            exist_ok=True,
            pretrained=True,
            resume=False,
            
            # 🛠️ 新增优化参数
            device=0,
            amp=True,           # 启用混合精度训练
            single_cls=False,
            # 尝试启用多尺度训练 (如果显存允许)
            # multi_scale=True,
            
            # 🎯 针对小目标的特殊设置
            # 通过自定义模型配置文件实现
            # cfg='yolov11_drone_optimized.yaml'  # 如果有自定义配置文件
        )
        
        print(f"🚀 开始优化改进训练阶段 C: {RUN_NAME_PHASE_C}")
        print("🎯 主要改进策略:")
        print("  1. 学习率大幅调整 (lr0: 0.001, lrf: 0.01)")
        print("  2. 数据增强全面增强 - 针对无人机小目标")
        print("  3. 损失权重重新平衡 (box:6.0, cls:1.2) - 重点提升分类精度")  
        print("  4. 延长训练周期和早停耐心")
        print("  5. 新增自动增强和随机擦除")
        print("  6. 验证参数优化 (conf:0.2, iou:0.6)")
        
        # 训练前检查
        print(f"📁 工作目录: {os.getcwd()}")
        print(f"📊 数据配置: {DATA_YAML}")
        print(f"⚖️ 预训练权重: {WEIGHTS_PTH}")
        
        # 开始训练
        results = model.train(**kwargs)
        
        # 训练后分析
        if hasattr(results, 'results_dict'):
            print("\n📈 训练结果分析:")
            final_metrics = results.results_dict
            if 'metrics/mAP50(B)' in final_metrics:
                mAP50 = final_metrics['metrics/mAP50(B)']
                print(f"✅ 最终 mAP50: {mAP50:.4f}")
                if mAP50 > 0.78:
                    print("🎉 mAP50 达到优秀水平!")
                elif mAP50 > 0.75:
                    print("👍 mAP50 有明显提升!")
        
        print("阶段 C 优化改进训练完成 ✅")
        return results
        
    except Exception as e:
        print("❌ 阶段 C 优化改进训练失败")
        print(f"错误详情: {str(e)}")
        traceback.print_exc()
        return None

def create_optimized_config():
    """创建优化配置文件 (如果需要自定义模型结构)"""
    config_content = """
# 优化的YOLOv11无人机检测配置
# 针对小目标和类别不平衡优化

# 可以在这里添加注意力机制、改进的neck结构等
# 需要根据实际的模型结构来调整
"""
    # 这里可以保存自定义配置文件
    # with open('yolov11_drone_optimized.yaml', 'w') as f:
    #     f.write(config_content)
    pass

def main():
    try:
        print("=" * 50)
        print("🚀 YOLOv11 无人机检测模型优化改进版")
        print("=" * 50)
        
        # 创建优化配置 (可选)
        create_optimized_config()
        
        # 直接加载预训练模型
        print(f"📥 加载预训练权重: {WEIGHTS_PTH}")
        model = YOLO(WEIGHTS_PTH)
        print("✅ 模型加载成功")
        
        # 显示模型信息
        print(f"📊 模型类别数: {model.model.nc}")
        print(f"🏷️ 类别名称: {model.names}")
        
        # 运行优化改进版训练
        results = train_phase_c_optimized_v2(model)
        
        if results:
            print("\n🎉 训练完成! 改进策略已应用")
            print("📊 建议下一步:")
            print("  1. 分析训练曲线，确认mAP50提升效果")
            print("  2. 检查混淆矩阵，观察motor和people类改进")
            print("  3. 如仍有提升空间，可进一步调整学习率策略")
        else:
            print("\n⚠️ 训练过程中出现错误，请检查上述日志")
            
    except Exception as e:
        print("❌ 训练初始化失败")
        print(f"错误详情: {str(e)}")
        traceback.print_exc()

if __name__ == '__main__':
    main()