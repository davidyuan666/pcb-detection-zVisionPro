"""
Z-Vision Pro - 工业视觉检测软件平台 (Demo Version)
基于申报书三大核心创新点的实现

核心创新：
1. 多模态深度特征嵌入与融合机制（Vision + CAD + MES）
2. PLM 驱动的跨领域知识迁移与自适应学习算法
3. 面向高可靠性的工业视觉检测软件平台

使用示例:
    python Z-VisionPro-pcb-detection.py --image path/to/image.jpg
    python Z-VisionPro-pcb-detection.py --dir data/valid --save-dir results/
"""

import os
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import numpy as np
import cv2
from ultralyticsplus import YOLO, render_result
import os as _os  # 用于环境变量清理（避免 SSL 证书环境变量导致下载失败）

# ==================== 创新一：多模态深度特征嵌入与融合机制 ====================

class MultiModalEmbedding:
    """
    多模态嵌入模块
    - 视觉模态：Swin Transformer 骨干网络
    - 结构化模态：CAD 数据（Graph 结构）
    - 工艺模态：MES 参数
    """
    
    def __init__(self):
        self.cad_cache = {}
        self.mes_cache = {}
        
    def visual_embedding(self, image: np.ndarray) -> np.ndarray:
        """
        视觉模态嵌入 - 基于 Swin Transformer
        捕捉像素级纹理和板级全局布局特征
        """
        # Demo: 简化实现，实际使用 Swin Transformer
        # 提取基础视觉特征
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        features = cv2.calcHist([gray], [0], None, [256], [0, 256])
        return features.flatten() / np.sum(features)  # 归一化
    
    def cad_embedding(self, cad_path: Optional[str] = None) -> Optional[np.ndarray]:
        """
        结构化模态嵌入 - CAD 数据解析为 Graph
        使用 GAT 编码电路设计逻辑
        """
        if cad_path and Path(cad_path).exists():
            # Demo: 简化实现，实际使用 GAT 网络
            # 返回模拟的 CAD 特征向量
            return np.random.rand(128)  # 128维特征向量
        return None
    
    def mes_embedding(self, process_params: Optional[Dict] = None) -> Optional[np.ndarray]:
        """
        工艺模态嵌入 - MES 工艺参数
        """
        if process_params:
            # Demo: 将工艺参数转换为特征向量
            params_array = np.array([
                process_params.get('layer_temp', 0),
                process_params.get('etch_time', 0),
                process_params.get('pressure', 0)
            ])
            return params_array
        return None
    
    def cross_modal_attention_fusion(
        self, 
        visual_feat: np.ndarray,
        cad_feat: Optional[np.ndarray] = None,
        mes_feat: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        多头跨模态注意力融合网络
        Visual 作为 Query，CAD/MES 作为 Key/Value
        实现动态信息互补
        """
        # Demo: 简化实现，实际使用多头注意力机制
        fused = visual_feat.copy()
        
        if cad_feat is not None:
            # 简单的特征融合（实际使用注意力机制）
            cad_weight = 0.3
            visual_weight = 0.7
            # 对齐维度
            if len(cad_feat) <= len(fused):
                pad_size = len(fused) - len(cad_feat)
                cad_feat = np.pad(cad_feat, (0, pad_size), 'constant')
            else:
                cad_feat = cad_feat[:len(fused)]
            fused = visual_weight * fused + cad_weight * cad_feat[:len(fused)]
        
        if mes_feat is not None:
            # MES 特征融合
            mes_weight = 0.2
            if len(mes_feat) <= len(fused):
                pad_size = len(fused) - len(mes_feat)
                mes_feat = np.pad(mes_feat, (0, pad_size), 'constant')
            else:
                mes_feat = mes_feat[:len(fused)]
            fused = 0.8 * fused + mes_weight * mes_feat[:len(fused)]
        
        return fused


# ==================== 创新二：PLM 驱动的自适应学习算法 ====================

class PCBLMAdaptiveLearning:
    """
    PCB 领域预训练模型 (PCB-PLM) 驱动的自适应学习
    - 两阶段训练：自监督预训练 + 监督微调
    - LoRA 轻量化增量学习
    """
    
    def __init__(self, base_model_path: Optional[str] = None):
        """
        初始化 PCB-PLM 模型
        """
        self.base_model = None
        self.lora_adapters = {}  # 存储 LoRA 适配器
        self.model_path = base_model_path
        
    def _sanitize_ssl_env(self) -> None:
        """
        清理无效的 SSL 证书环境变量，避免 httpx 在 Hugging Face 下载时抛出 FileNotFoundError。
        若 SSL_CERT_FILE/REQUESTS_CA_BUNDLE 指向的文件不存在，则移除该环境变量，回退到 certifi 默认证书。
        """
        for var in ["SSL_CERT_FILE", "REQUESTS_CA_BUNDLE"]:
            val = _os.environ.get(var)
            if val and not Path(val).exists():
                try:
                    del _os.environ[var]
                except Exception:
                    pass

    def _find_local_hf_weights(self, repo_id: str) -> Optional[Path]:
        """
        在本机 Hugging Face 缓存目录中查找已下载的权重文件 (.pt/.pth)。
        常见缓存路径：~/.cache/huggingface/hub/models--<user>--<repo>/snapshots/**
        """
        home = Path.home()
        hf_home = Path(_os.environ.get("HF_HOME", home / ".cache" / "huggingface"))
        # models--keremberke--yolov8s-pcb-defect-segmentation
        cache_root = hf_home / "hub" / f"models--{repo_id.replace('/', '--')}"
        if cache_root.exists():
            # 遍历 snapshots 下的所有版本，查找权重文件
            for pt in cache_root.glob("snapshots/*/*.pt"):
                return pt
            for pth in cache_root.glob("snapshots/*/*.pth"):
                return pth
        # 兼容旧路径结构
        legacy_root = hf_home / f"models--{repo_id.replace('/', '--')}"
        if legacy_root.exists():
            for pt in legacy_root.rglob("*.pt"):
                return pt
            for pth in legacy_root.rglob("*.pth"):
                return pth
        return None

    def load_pretrained_model(self) -> Any:
        """
        加载预训练模型（两阶段：自监督预训练 + 监督微调）
        Demo: 使用 YOLO 作为基础模型
        """
        # 先清理可能导致下载失败的 SSL 环境变量
        self._sanitize_ssl_env()

        # 优先使用用户提供的本地模型路径
        if self.model_path:
            mp = Path(self.model_path)
            if not mp.exists():
                raise FileNotFoundError(f"指定的模型路径不存在: {mp}")
            model = YOLO(str(mp))
        else:
            repo_id = 'keremberke/yolov8s-pcb-defect-segmentation'
            # 尝试从本地 HF 缓存查找已下载的权重，避免联网
            local_weights = self._find_local_hf_weights(repo_id)
            if local_weights and local_weights.exists():
                print(f"[PLM] 使用本地缓存权重: {local_weights}")
                model = YOLO(str(local_weights))
            else:
                # 尝试直接使用仓库 ID（可能需要联网）
                print("[PLM] 未找到本地缓存权重，如需离线运行请使用 --model-path 指定本地 .pt 文件。")
                print("[PLM] 正在尝试从 Hugging Face Hub 加载（可能需要可用的网络与证书配置）...")
                model = YOLO(repo_id)
        
        self.base_model = model
        print("[PLM] PCB 领域预训练模型加载完成")
        return model
    
    def lora_fine_tune(
        self, 
        new_defect_samples: List[str],
        new_defect_type: str,
        epochs: int = 10
    ) -> Dict:
        """
        LoRA 轻量化增量学习
        - 冻结 99%+ 参数，仅训练 <1% 的低秩矩阵
        - 快速适应新缺陷类型，避免灾难性遗忘
        
        Args:
            new_defect_samples: 新缺陷样本路径列表
            new_defect_type: 新缺陷类型名称
            epochs: 微调轮数
        
        Returns:
            训练结果统计
        """
        if self.base_model is None:
            self.load_pretrained_model()
        
        # Demo: 简化实现，实际使用 LoRA 技术
        print(f"[PLM] 开始 LoRA 增量学习，新缺陷类型: {new_defect_type}")
        print(f"[PLM] 样本数量: {len(new_defect_samples)}, 训练轮数: {epochs}")
        
        # 模拟训练过程
        start_time = time.time()
        # 实际实现中会：
        # 1. 冻结基础模型参数
        # 2. 注入低秩适配器矩阵
        # 3. 仅训练适配器参数
        
        training_time = time.time() - start_time
        
        # 保存 LoRA 适配器
        self.lora_adapters[new_defect_type] = {
            'samples': len(new_defect_samples),
            'training_time': training_time,
            'epochs': epochs
        }
        
        return {
            'defect_type': new_defect_type,
            'samples_used': len(new_defect_samples),
            'training_time_hours': training_time / 3600,
            'status': 'completed'
        }
    
    def predict_with_adapters(self, image_path: str) -> Dict:
        """
        使用基础模型 + LoRA 适配器进行预测
        """
        if self.base_model is None:
            self.load_pretrained_model()
        
        results = self.base_model(image_path)
        return {
            'predictions': results,
            'active_adapters': list(self.lora_adapters.keys())
        }


# ==================== 创新三：高可靠性工业视觉检测平台 ====================

class ReliabilityMonitor:
    """
    实时鲁棒性校验与故障预警
    - 数字孪生基线模型
    - 熵嵌入机制
    - 异常检测与预警
    """
    
    def __init__(self, confidence_threshold: float = 0.95, anomaly_threshold: float = 3.0):
        """
        初始化可靠性监控器
        
        Args:
            confidence_threshold: 置信度阈值（95%）
            anomaly_threshold: 异常阈值（3个标准差）
        """
        self.confidence_threshold = confidence_threshold
        self.anomaly_threshold = anomaly_threshold
        
        # 数字孪生基线统计
        self.baseline_stats = {
            'mean_confidence': 0.98,
            'std_confidence': 0.02,
            'mean_detections_per_image': 2.5,
            'std_detections_per_image': 1.0
        }
        
        # 运行时统计
        self.total_images = 0
        self.anomaly_count = 0
        self.start_time = time.time()
        self.anomaly_history = []
        
    def validate_detection_result(self, detections: List[Dict]) -> Dict:
        """
        实时鲁棒性校验
        基于熵嵌入机制评估检测结果置信度
        
        Args:
            detections: 检测结果列表，每个元素包含 {'confidence': float, ...}
        
        Returns:
            校验结果，包含异常状态和警告信息
        """
        if not detections:
            return {
                'is_anomaly': True,
                'status': 'no_detection',
                'warnings': ['未检测到任何目标']
            }
        
        confidences = [d.get('confidence', 0) for d in detections]
        mean_conf = np.mean(confidences)
        entropy = self._calculate_entropy_embedding(confidences)
        
        # 计算与基线的偏离度
        conf_deviation = abs(mean_conf - self.baseline_stats['mean_confidence'])
        num_detections = len(detections)
        det_deviation = abs(num_detections - self.baseline_stats['mean_detections_per_image'])
        
        warnings = []
        is_anomaly = False
        
        # 置信度校验
        if mean_conf < self.confidence_threshold:
            warnings.append(f"置信度低于阈值: {mean_conf:.4f} < {self.confidence_threshold}")
            is_anomaly = True
        
        # 偏离基线校验（3个标准差）
        if conf_deviation > self.anomaly_threshold * self.baseline_stats['std_confidence']:
            warnings.append(
                f"置信度偏离基线: {conf_deviation:.4f} > "
                f"{self.anomaly_threshold * self.baseline_stats['std_confidence']:.4f}"
            )
            is_anomaly = True
        
        # 检测数量异常
        if det_deviation > self.anomaly_threshold * self.baseline_stats['std_detections_per_image']:
            warnings.append(
                f"检测数量异常: {num_detections} "
                f"(均值: {self.baseline_stats['mean_detections_per_image']:.1f})"
            )
            is_anomaly = True
        
        self.total_images += 1
        if is_anomaly:
            self.anomaly_count += 1
            self.anomaly_history.append({
                'timestamp': datetime.now().isoformat(),
                'warnings': warnings,
                'confidence': mean_conf,
                'entropy': entropy
            })
        
        return {
            'is_anomaly': is_anomaly,
            'confidence_score': mean_conf,
            'entropy_embedding': entropy,
            'num_detections': num_detections,
            'warnings': warnings,
            'status': 'anomaly' if is_anomaly else 'normal',
            'deviation_from_baseline': conf_deviation
        }
    
    def _calculate_entropy_embedding(self, confidences: List[float]) -> float:
        """
        计算熵嵌入值 - 量化检测结果的不确定性
        """
        if not confidences:
            return 0.0
        
        conf_array = np.array(confidences)
        probs = conf_array / np.sum(conf_array) if np.sum(conf_array) > 0 else conf_array
        
        # 香农熵
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        max_entropy = np.log(len(confidences) + 1e-10)
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        
        return float(normalized_entropy)
    
    def get_reliability_stats(self) -> Dict:
        """
        获取可靠性统计信息
        """
        elapsed_time = time.time() - self.start_time
        mtbf = elapsed_time / (self.anomaly_count + 1) if self.anomaly_count > 0 else elapsed_time
        
        return {
            'total_images': self.total_images,
            'anomaly_count': self.anomaly_count,
            'anomaly_rate': self.anomaly_count / self.total_images if self.total_images > 0 else 0.0,
            'mtbf_hours': mtbf / 3600,
            'uptime_hours': elapsed_time / 3600,
            'reliability_score': 1.0 - (self.anomaly_count / self.total_images) if self.total_images > 0 else 1.0,
            'anomaly_history_count': len(self.anomaly_history)
        }


# ==================== Z-Vision Pro 主平台类 ====================

class ZVisionPro:
    """
    Z-Vision Pro 工业视觉检测软件平台
    集成三大核心创新点
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        初始化 Z-Vision Pro 平台
        
        Args:
            model_path: 预训练模型路径（可选）
        """
        print("=" * 60)
        print("Z-Vision Pro 工业视觉检测软件平台")
        print("=" * 60)
        
        # 初始化三大核心模块
        self.multimodal = MultiModalEmbedding()
        self.plm = PCBLMAdaptiveLearning(model_path)
        self.reliability = ReliabilityMonitor()
        
        # 加载预训练模型
        self.model = self.plm.load_pretrained_model()
        
        # 统计信息
        self.detection_stats = defaultdict(int)
        
    def detect(
        self, 
        image_path: str,
        cad_path: Optional[str] = None,
        mes_params: Optional[Dict] = None,
        save_result: bool = False,
        output_dir: Optional[str] = None,
        show_result: bool = False
    ) -> Dict:
        """
        执行检测任务（集成多模态融合和可靠性校验）
        
        Args:
            image_path: 图像路径
            cad_path: CAD 文件路径（可选）
            mes_params: MES 工艺参数（可选）
            save_result: 是否保存结果
            output_dir: 结果保存目录
        
        Returns:
            检测结果字典
        """
        # 1. 多模态特征提取
        image = cv2.imread(image_path)
        if image is None:
            return {'error': f'无法读取图像: {image_path}'}
        
        visual_feat = self.multimodal.visual_embedding(image)
        cad_feat = self.multimodal.cad_embedding(cad_path)
        mes_feat = self.multimodal.mes_embedding(mes_params)
        
        # 2. 跨模态注意力融合
        fused_features = self.multimodal.cross_modal_attention_fusion(
            visual_feat, cad_feat, mes_feat
        )
        
        # 3. PLM 模型推理
        results = self.model(image_path)
        
        # 4. 解析检测结果
        detections = []
        if results and len(results) > 0:
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for i in range(len(boxes)):
                        detections.append({
                            'class': int(boxes.cls[i]),
                            'confidence': float(boxes.conf[i]),
                            'bbox': boxes.xyxy[i].cpu().numpy().tolist()
                        })
                        self.detection_stats['total_detections'] += 1
        
        # 5. 可靠性校验
        reliability_result = self.reliability.validate_detection_result(detections)
        
        # 6. 可视化渲染与保存/显示
        rendered_image = None
        rendered_path_str = None
        if results and len(results) > 0:
            try:
                rendered_image = render_result(model=self.model, image=image, result=results[0])
            except Exception:
                rendered_image = image
        else:
            rendered_image = image

        if save_result and output_dir:
            self._save_result(image_path, detections, reliability_result, output_dir)
            try:
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                rendered_path = output_path / f"{Path(image_path).stem}_result.jpg"
                # 兼容 ultralyticsplus.render_result 返回的对象（通常为可 .save 的图像对象）
                # 以及 numpy 数组（OpenCV 图像）。
                if hasattr(rendered_image, 'save'):
                    rendered_image.save(str(rendered_path))
                else:
                    try:
                        if isinstance(rendered_image, np.ndarray):
                            cv2.imwrite(str(rendered_path), rendered_image)
                        else:
                            # 尝试将可能的 PIL.Image 转为 numpy 后保存
                            arr = np.array(rendered_image)
                            if arr.ndim == 3 and arr.shape[2] == 3:
                                arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                            cv2.imwrite(str(rendered_path), arr)
                    except Exception:
                        pass
                rendered_path_str = str(rendered_path)
            except Exception:
                pass

        if show_result:
            try:
                window_title = f"Z-Vision Pro - {Path(image_path).name}"
                cv2.imshow(window_title, rendered_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            except Exception:
                pass
        
        return {
            'image_path': image_path,
            'detections': detections,
            'num_detections': len(detections),
            'reliability': reliability_result,
            'multimodal_fusion': {
                'visual_feat_shape': visual_feat.shape,
                'cad_feat_used': cad_feat is not None,
                'mes_feat_used': mes_feat is not None,
                'fused_feat_shape': fused_features.shape
            },
            'rendered_image_path': rendered_path_str
        }
    
    def _save_result(self, image_path: str, detections: List[Dict], 
                     reliability: Dict, output_dir: str):
        """保存检测结果"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 保存 JSON 结果
        result_file = output_path / f"{Path(image_path).stem}_result.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump({
                'image_path': image_path,
                'detections': detections,
                'reliability': reliability,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2, ensure_ascii=False)
    
    def batch_detect(
        self,
        image_dir: str,
        cad_dir: Optional[str] = None,
        output_dir: Optional[str] = None
    ) -> Dict:
        """
        批量检测
        
        Args:
            image_dir: 图像目录
            cad_dir: CAD 文件目录（可选）
            output_dir: 结果保存目录
        
        Returns:
            批量检测结果统计
        """
        image_dir_path = Path(image_dir)
        image_files = list(image_dir_path.glob('*.jpg')) + list(image_dir_path.glob('*.png'))
        
        results = []
        for img_file in image_files:
            cad_path = None
            if cad_dir:
                cad_path = Path(cad_dir) / f"{img_file.stem}.gerber"
            
            result = self.detect(
                str(img_file),
                cad_path=str(cad_path) if cad_path and cad_path.exists() else None,
                save_result=True,
                output_dir=output_dir
            )
            results.append(result)
        
        # 获取可靠性统计
        reliability_stats = self.reliability.get_reliability_stats()
        
        return {
            'total_images': len(results),
            'results': results,
            'reliability_stats': reliability_stats,
            'detection_stats': dict(self.detection_stats)
        }
    
    def adaptive_learning(self, new_samples: List[str], defect_type: str) -> Dict:
        """
        自适应学习新缺陷类型（LoRA 增量学习）
        """
        return self.plm.lora_fine_tune(new_samples, defect_type)
    
    def get_platform_stats(self) -> Dict:
        """获取平台运行统计"""
        return {
            'reliability': self.reliability.get_reliability_stats(),
            'detection': dict(self.detection_stats),
            'active_adapters': list(self.plm.lora_adapters.keys())
        }


# ==================== 命令行接口 ====================

def main():
    """主函数 - 命令行接口"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Z-Vision Pro - 工业视觉检测软件平台 (Demo)'
    )
    parser.add_argument('--image', type=str, help='单张图像路径')
    parser.add_argument('--dir', type=str, help='图像目录路径')
    parser.add_argument('--model-path', type=str, default=None, help='模型路径')
    parser.add_argument('--cad', type=str, default=None, help='CAD 文件路径')
    parser.add_argument('--save-dir', type=str, default='results', help='结果保存目录')
    parser.add_argument('--stats', action='store_true', help='显示平台统计信息')
    
    args = parser.parse_args()
    
    # 初始化平台
    platform = ZVisionPro(model_path=args.model_path)
    
    # 执行检测
    if args.image:
        # 单张图像检测
        result = platform.detect(
            args.image,
            cad_path=args.cad,
            save_result=True,
            output_dir=args.save_dir,
            show_result=True
        )
        print("\n" + "=" * 60)
        print("检测结果")
        print("=" * 60)
        print(f"图像: {result['image_path']}")
        print(f"检测数量: {result['num_detections']}")
        print(f"可靠性状态: {result['reliability']['status']}")
        if result['reliability']['warnings']:
            print(f"警告: {result['reliability']['warnings']}")
        print(f"置信度: {result['reliability']['confidence_score']:.4f}")
        print(f"多模态融合: CAD={result['multimodal_fusion']['cad_feat_used']}, "
              f"MES={result['multimodal_fusion']['mes_feat_used']}")
        if result.get('rendered_image_path'):
            print(f"已保存渲染结果: {result['rendered_image_path']}")
        
    elif args.dir:
        # 批量检测
        print(f"\n开始批量检测: {args.dir}")
        batch_result = platform.batch_detect(
            args.dir,
            output_dir=args.save_dir
        )
        print("\n" + "=" * 60)
        print("批量检测完成")
        print("=" * 60)
        print(f"总图像数: {batch_result['total_images']}")
        print(f"\n可靠性统计:")
        rel_stats = batch_result['reliability_stats']
        print(f"  - 总处理数: {rel_stats['total_images']}")
        print(f"  - 异常数: {rel_stats['anomaly_count']}")
        print(f"  - 异常率: {rel_stats['anomaly_rate']:.4f}")
        print(f"  - 可靠性得分: {rel_stats['reliability_score']:.4f}")
        print(f"  - 运行时间: {rel_stats['uptime_hours']:.2f} 小时")
    else:
        parser.print_help()
    
    # 显示统计信息
    if args.stats:
        stats = platform.get_platform_stats()
        print("\n" + "=" * 60)
        print("平台统计信息")
        print("=" * 60)
        print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()

