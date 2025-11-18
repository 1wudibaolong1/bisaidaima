import traceback
from ultralytics import YOLO
import os

# ---------------- config ----------------
WEIGHTS_PTH = r'/hy-tmp/ultralytics-8.3.184/runs/detect/yolov11_drone_phaseC_optimized_v4/weights/best.pt'
DATA_YAML = r'/hy-tmp/ultralytics-8.3.184/data/data/data.yaml'
WORKERS = 16

# ---------- 阶段 C (修正版) ----------
IMGSZ_PHASE_C = 1024
EPOCHS_PHASE_C = 150  # 进一步增加训练轮数
BATCH_PHASE_C = -1
RUN_NAME_PHASE_C = 'yolov11_drone_phaseC_optimized_v6_fixed'
# ----------------------------------------

def train_phase_c_fixed(model):
    """阶段 C 修正版: 针对高分类损失的专项优化"""
    try:
        kwargs = dict(
            data=DATA_YAML,
            epochs=EPOCHS_PHASE_C,
            imgsz=IMGSZ_PHASE_C,
            batch=BATCH_PHASE_C,
            workers=WORKERS,
            name=RUN_NAME_PHASE_C,
            val=True,
            
            # 🔧 学习率策略修正 - 降低学习率避免震荡
            lr0=0.0005,          # 从0.001降低到0.0005
            lrf=0.005,           # 进一步降低最终学习率
            
            # 优化器设置保持不变
            optimizer='SGD',
            momentum=0.937,
            weight_decay=0.0005,
            
            # 📈 学习率调度优化
            cos_lr=True,
            warmup_epochs=10,    # 进一步延长warmup
            warmup_momentum=0.95,
            warmup_bias_lr=0.05, # 降低bias学习率
            
            # 🎯 数据增强策略修正 - 降低增强强度
            hsv_h=0.015,         # 降低色调变化
            hsv_s=0.6,           # 降低饱和度变化  
            hsv_v=0.3,           # 降低亮度变化
            degrees=5.0,         # 降低旋转增强
            translate=0.1,       # 降低平移
            scale=0.5,           # 降低尺度变化
            shear=2.0,           # 降低剪切变换
            perspective=0.0,     # 移除透视变换
            flipud=0.1,          # 降低上下翻转概率
            fliplr=0.5,
            
            # 🚀 数据增强策略调整 - 重点优化小目标
            mosaic=0.4,          # 适度马赛克增强
            mixup=0.15,          # 适度mixup增强
            copy_paste=0.08,     # 适度copy-paste增强
            auto_augment='randaugment',
            erasing=0.3,         # 降低随机擦除概率
            
            # ⚖️ 损失函数权重重新调整 - 重点解决分类损失过高
            box=7.5,             # 提高定位权重 (从6.0→7.5)
            cls=0.6,             # 大幅降低分类权重 (从1.2→0.6)
            dfl=1.5,             # 保持DFL权重
            
            # 🎯 新增关键参数 - 处理类别不平衡
            label_smoothing=0.1, # 标签平滑，缓解过拟合
            overlap_mask=True,
            
            # 📊 验证和早停优化
            conf=0.001,          # 大幅降低验证置信度阈值，检测更多目标
            iou=0.5,             # 回归标准IoU阈值
            patience=20,         # 进一步延长早停耐心
            
            # 🛠️ 模型保存和配置
            save=True,
            save_period=5,       # 更频繁保存检查点
            exist_ok=True,
            pretrained=True,
            resume=False,
            
            # 💡 新增训练技巧
            device=0,
            amp=True,
            single_cls=False,
            deterministic=True,
            plots=True,          # 确保绘制训练曲线
            
            # 🎯 针对高分类损失的专项设置
            close_mosaic=15,     # 延迟关闭马赛克增强
        )
        
        print(f"🔧 开始修正训练阶段 C: {RUN_NAME_PHASE_C}")
        print("🎯 主要修正策略:")
        print("  1. 学习率降低 (lr0: 0.0005, lrf: 0.005) - 解决训练不稳定")
        print("  2. 分类损失权重大幅降低 (cls: 0.6) - 解决高分类损失")
        print("  3. 定位损失权重提高 (box: 7.5) - 加强定位能力")
        print("  4. 数据增强强度适度降低 - 避免过度增强")
        print("  5. 新增标签平滑 - 缓解过拟合")
        print("  6. 验证置信度大幅降低 (conf: 0.001) - 检测更多目标")
        
        # 训练前分析
        print(f"\n📊 问题诊断:")
        print("  - 分类损失过高 (3.0+)，需要降低学习率和分类权重")
        print("  - motor和people类性能下降，需要更稳定的训练")
        print("  - 训练曲线波动大，需要降低学习率")
        
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
                    print("🎉 mAP50 显著提升!")
                elif mAP50 > 0.77:
                    print("👍 mAP50 恢复到之前水平!")
                else:
                    print("⚠️ mAP50 仍需进一步优化")
        
        print("阶段 C 修正训练完成 ✅")
        return results
        
    except Exception as e:
        print("❌ 阶段 C 修正训练失败")
        print(f"错误详情: {str(e)}")
        traceback.print_exc()
        return None

def analyze_previous_results():
    """分析之前训练结果，提供针对性建议"""
    print("\n📋 基于之前训练的分析:")
    print("  1. 分类损失从 ~1.4 上升到 ~3.0 → 学习率过大或分类权重过高")
    print("  2. mAP50从 0.772 下降到 0.760 → 训练不稳定")
    print("  3. motor类从 0.598 下降到 0.581 → 需要更稳定的训练策略")
    print("  4. people类从 0.688 下降到 0.657 → 分类任务过于困难")
    print("\n🎯 修正方向:")
    print("  - 降低学习率，提高训练稳定性")
    print("  - 降低分类损失权重，避免梯度爆炸")
    print("  - 适度降低数据增强强度")
    print("  - 延长训练周期，让模型充分收敛")

def main():
    try:
        print("=" * 60)
        print("🔧 YOLOv11 无人机检测模型修正版 - 针对高分类损失优化")
        print("=" * 60)
        
        # 分析之前结果
        analyze_previous_results()
        
        # 加载预训练模型
        print(f"\n📥 加载预训练权重: {WEIGHTS_PTH}")
        model = YOLO(WEIGHTS_PTH)
        print("✅ 模型加载成功")
        
        # 显示模型信息
        print(f"📊 模型类别数: {model.model.nc}")
        print(f"🏷️ 类别名称: {model.names}")
        
        # 运行修正版训练
        results = train_phase_c_fixed(model)
        
        if results:
            print("\n🎉 修正训练完成!")
            print("📊 下一步建议:")
            print("  1. 监控分类损失是否降低到合理范围(<2.0)")
            print("  2. 观察训练曲线是否更加平滑")
            print("  3. 检查motor和people类的AP是否恢复")
            print("  4. 如果仍有问题，可尝试Adam优化器")
        else:
            print("\n⚠️ 训练过程中出现错误，请检查上述日志")
            
    except Exception as e:
        print("❌ 训练初始化失败")
        print(f"错误详情: {str(e)}")
        traceback.print_exc()

if __name__ == '__main__':
    main()