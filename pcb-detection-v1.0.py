# PCB 缺陷检测 - 使用 ModelScope 预训练模型
# 支持的标签: ['Dry_joint', 'Incorrect_installation', 'PCB_damage', 'Short_circuit']
# 支持的缺陷类别：
# Dry_joint (干焊)
# Incorrect_installation (安装错误)
# PCB_damage (PCB 损坏)
# Short_circuit (短路)
# 
# 安装依赖:
# pip install ultralyticsplus==0.0.23 ultralytics==8.0.21 modelscope
#
# 下载模型（使用 ModelScope）:
# modelscope download --model keremberke/yolov8s-pcb-defect-segmentation
#
# 使用示例:
# 1. 检测 data/train 目录（默认）:
#    python pcb-detection.py
#
# 2. 检测单张图像:
#    python pcb-detection.py --image path/to/image.jpg
#
# 3. 检测指定目录:
#    python pcb-detection.py --dir path/to/images/
#
# 4. 检测 data 目录下的特定子集:
#    python pcb-detection.py --subset train    # 检测 data/train 目录（默认）
#    python pcb-detection.py --subset test     # 检测 data/test 目录
#    python pcb-detection.py --subset valid    # 检测 data/valid 目录
#    python pcb-detection.py --subset all      # 检测所有子目录
#
# 5. 指定数据目录:
#    python pcb-detection.py --data-dir custom_data --subset train
#
# 6. 保存结果到指定目录:
#    python pcb-detection.py --subset train --save-dir results/
#
# 7. 指定本地模型路径:
#    python pcb-detection.py --model-path path/to/model.pt

import os
import json
import time
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import numpy as np
from ultralyticsplus import YOLO, render_result
import cv2
import matplotlib.pyplot as plt

# 支持的缺陷类别
DEFECT_CLASSES = ['Dry_joint', 'Incorrect_installation', 'PCB_damage', 'Short_circuit']


def find_modelscope_model_path(model_name='keremberke/yolov8s-pcb-defect-segmentation'):
    """
    查找本地 ModelScope 下载的模型目录
    
    Args:
        model_name: 模型名称，格式为 'user/model-name'
    
    Returns:
        模型目录路径（如果找到），如果找不到则返回 None
        注意：返回目录路径而不是文件路径，这样 YOLO 可以自动处理模型加载
    """
    # ModelScope 默认缓存路径
    home = Path.home()
    
    # Windows 和 Linux/Mac 的常见路径
    # 注意：ModelScope 可能使用 hub/ 或 hub/models/ 作为基础路径
    possible_base_paths = [
        home / '.cache' / 'modelscope' / 'hub',
        home / '.cache' / 'modelscope' / 'hub' / 'models',  # 某些版本使用此路径
        home / '.modelscope' / 'hub',
        home / '.modelscope' / 'hub' / 'models',
    ]
    
    # 检查环境变量
    if os.environ.get('MODELSCOPE_CACHE'):
        cache_base = Path(os.environ.get('MODELSCOPE_CACHE'))
        possible_base_paths.insert(0, cache_base / 'hub' / 'models')
        possible_base_paths.insert(0, cache_base / 'hub')
    
    # 构建完整的模型路径
    possible_paths = []
    for base_path in possible_base_paths:
        if base_path.exists():
            model_path = base_path / model_name
            possible_paths.append(model_path)
    
    # 查找模型目录（如果目录存在，说明模型已下载）
    for base_path in possible_paths:
        if base_path.exists():
            # 检查目录中是否有模型文件（.pt 或 .pth）
            pt_files = list(base_path.glob('*.pt'))
            pth_files = list(base_path.glob('*.pth'))
            if pt_files or pth_files:
                # 找到模型目录，返回 None 表示使用 Hub 路径（让 YOLO 自动处理）
                # 这样可以利用本地缓存，同时避免 PyTorch 2.6 的 weights_only 限制
                return None  # 返回 None 表示找到本地目录，但使用 Hub 路径加载
    
    return False  # 返回 False 表示未找到本地模型


def load_model(model_path=None, conf_threshold=0.25, iou_threshold=0.45, max_detections=1000):
    """
    加载本地 ModelScope 下载的 YOLO 模型
    
    Args:
        model_path: 模型文件路径（可选，如果不提供则自动查找）
        conf_threshold: NMS 置信度阈值
        iou_threshold: NMS IoU 阈值
        max_detections: 每张图像的最大检测数量
    
    Returns:
        加载的 YOLO 模型
    """
    model_name = 'keremberke/yolov8s-pcb-defect-segmentation'
    
    # 如果没有提供模型路径，尝试自动查找本地模型
    if model_path is None:
        print(f"正在查找本地 ModelScope 模型: {model_name}")
        local_model_found = find_modelscope_model_path(model_name)
        
        if local_model_found is None:
            # 找到本地模型目录，使用 Hub 路径（YOLO 会自动使用本地缓存）
            print(f"找到本地 ModelScope 模型，使用 Hub 路径加载（将自动使用本地缓存）")
            model_path = model_name
        elif local_model_found is False:
            # 未找到本地模型
            print(f"警告: 未找到本地模型，将从 ModelScope Hub 加载")
            print(f"提示: 请先运行: modelscope download --model {model_name}")
            model_path = model_name
        else:
            # 如果返回了路径，使用该路径
            model_path = local_model_found
    
    print(f"正在加载模型: {model_path}")
    
    # 加载模型
    model = YOLO(model_path)
    
    # 设置模型参数
    model.overrides['conf'] = conf_threshold  # NMS confidence threshold
    model.overrides['iou'] = iou_threshold  # NMS IoU threshold
    model.overrides['agnostic_nms'] = False  # NMS class-agnostic
    model.overrides['max_det'] = max_detections  # maximum number of detections per image
    
    print("模型加载完成!")
    return model


def predict_single_image(model, image_path, save_dir=None, show_result=True):
    """
    对单张图像进行缺陷检测
    
    Args:
        model: YOLO 模型
        image_path: 图像路径（可以是本地文件路径或 URL）
        save_dir: 结果保存目录（可选）
        show_result: 是否显示结果
    
    Returns:
        检测结果
    """
    print(f"\n正在检测图像: {image_path}")
    
    # 执行推理
    results = model.predict(image_path)
    
    # 获取第一个结果
    result = results[0]
    
    # 打印检测结果
    print("\n检测结果:")
    print(f"边界框数量: {len(result.boxes)}")
    if len(result.boxes) > 0:
        print("\n详细信息:")
        for i, box in enumerate(result.boxes):
            class_id = int(box.cls[0].cpu().numpy())
            confidence = float(box.conf[0].cpu().numpy())
            class_name = model.names[class_id] if class_id < len(model.names) else f"Class_{class_id}"
            print(f"  检测 {i+1}: {class_name} (置信度: {confidence:.4f})")
    
    # 打印分割掩码信息
    if result.masks is not None:
        print(f"分割掩码数量: {len(result.masks)}")
    
    # 渲染结果
    render = render_result(model=model, image=image_path, result=result)
    
    # 保存结果
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        image_name = Path(image_path).stem
        save_path = os.path.join(save_dir, f"{image_name}_detection.jpg")
        render.save(save_path)
        print(f"\n结果已保存到: {save_path}")
    
    # 显示结果
    if show_result:
        render.show()
    
    return result


def predict_batch(model, image_dir, save_dir=None, show_result=False, recursive=False):
    """
    批量检测图像
    
    Args:
        model: YOLO 模型
        image_dir: 图像目录（如果 recursive=True，会递归搜索子目录）
        save_dir: 结果保存目录
        show_result: 是否显示每张图像的结果
        recursive: 是否递归搜索子目录
    
    Returns:
        所有检测结果列表
    """
    # 支持的图像格式
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    
    # 获取所有图像文件
    image_files = []
    dir_path = Path(image_dir)
    
    if recursive:
        # 递归搜索所有子目录
        for ext in image_extensions:
            image_files.extend(dir_path.rglob(f'*{ext}'))
            image_files.extend(dir_path.rglob(f'*{ext.upper()}'))
    else:
        # 只在当前目录搜索
        for ext in image_extensions:
            image_files.extend(dir_path.glob(f'*{ext}'))
            image_files.extend(dir_path.glob(f'*{ext.upper()}'))
    
    # 过滤掉 JSON 文件（如果有的话）
    image_files = [f for f in image_files if f.suffix.lower() in [ext.lower() for ext in image_extensions]]
    
    if not image_files:
        print(f"在目录 {image_dir} 中未找到图像文件")
        return []
    
    print(f"\n找到 {len(image_files)} 张图像")
    
    all_results = []
    for i, image_path in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] 处理: {image_path.name}")
        try:
            result = predict_single_image(model, str(image_path), save_dir, show_result)
            all_results.append({
                'image_path': str(image_path),
                'result': result
            })
        except Exception as e:
            print(f"处理图像 {image_path.name} 时出错: {e}")
            continue
    
    return all_results


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='PCB 缺陷检测')
    parser.add_argument('--image', type=str, help='单张图像路径或 URL')
    parser.add_argument('--dir', type=str, help='图像目录路径（批量检测）')
    parser.add_argument('--data-dir', type=str, default='data', 
                       help='数据目录路径（默认: data）')
    parser.add_argument('--subset', type=str, choices=['train', 'test', 'valid', 'all'], 
                       default='train', help='选择数据子集: train/test/valid/all（默认: train）')
    parser.add_argument('--save-dir', type=str, default='detection_results', 
                       help='结果保存目录（默认: detection_results）')
    parser.add_argument('--model-path', type=str, default=None,
                       help='本地模型路径（可选，如果不提供则自动查找 ModelScope 缓存）')
    parser.add_argument('--conf', type=float, default=0.25, 
                       help='置信度阈值（默认: 0.25）')
    parser.add_argument('--iou', type=float, default=0.45, 
                       help='IoU 阈值（默认: 0.45）')
    parser.add_argument('--max-det', type=int, default=1000, 
                       help='最大检测数量（默认: 1000）')
    # 默认显示检测窗口，如需禁用可加 --no-show
    parser.add_argument('--no-show', action='store_true', 
                       help='不显示检测结果窗口（默认会显示）')
    
    args = parser.parse_args()
    
    # 加载模型（优先使用本地 ModelScope 下载的模型）
    model = load_model(model_path=args.model_path, conf_threshold=args.conf, 
                      iou_threshold=args.iou, max_detections=args.max_det)
    
    # 执行检测
    show_flag = not args.no_show
    if args.image:
        # 单张图像检测
        predict_single_image(model, args.image, args.save_dir, show_flag)
    elif args.dir:
        # 批量检测指定目录
        predict_batch(model, args.dir, args.save_dir, show_flag, recursive=True)
    else:
        # 使用 data 目录结构
        data_dir = Path(args.data_dir)
        
        if not data_dir.exists():
            print(f"\n错误: 数据目录 {data_dir} 不存在")
            print("\n使用示例:")
            print("  python pcb-detection.py --image path/to/image.jpg")
            print("  python pcb-detection.py --dir path/to/images/")
            print("  python pcb-detection.py --subset train    # 检测 data/train 目录")
            print("  python pcb-detection.py --subset test     # 检测 data/test 目录")
            print("  python pcb-detection.py --subset valid    # 检测 data/valid 目录")
            print("  python pcb-detection.py --subset all      # 检测所有子目录")
            return
        
        # 根据 subset 参数选择目录
        if args.subset == 'all':
            # 检测所有子目录
            print(f"\n检测所有子目录: {data_dir}")
            subsets = ['train', 'test', 'valid']
            for subset in subsets:
                subset_dir = data_dir / subset
                if subset_dir.exists() and subset_dir.is_dir():
                    print(f"\n{'='*60}")
                    print(f"处理子集: {subset}")
                    print(f"{'='*60}")
                    subset_save_dir = Path(args.save_dir) / subset
                    predict_batch(model, str(subset_dir), str(subset_save_dir), show_flag, recursive=False)
        else:
            # 检测指定子目录
            subset_dir = data_dir / args.subset
            if not subset_dir.exists():
                print(f"\n错误: 子目录 {subset_dir} 不存在")
                return
            
            print(f"\n检测子集: {args.subset}")
            subset_save_dir = Path(args.save_dir) / args.subset
            predict_batch(model, str(subset_dir), str(subset_save_dir), show_flag, recursive=False)


if __name__ == '__main__':
    main()

