## Note: 
- 数据集名称：DsPCBSD+。
- 数据来源：数据集中的图像来自于实际生产的 PCB 板，通过 AOI（自动光学检查）设备拍摄。
- 图像数量：10,259 张。
- 标记数量：20,276 个手动标记 (缺陷边界框)。
- 缺陷类别：9 种（短路、尖刺、杂铜、断路、鼠咬、孔崩、导体刮痕、导体异物、基材异物）。
- 训练/验证集比例：8:2。
- 验证模型：Co-DETR 和 YOLOv6-L6，mAP 均超过 0.84
  - mAP（mean Average Precision）
  - 召回率（Recall）

----

## 1. 背景介绍

- 印刷电路板（PCB）在制造过程中经常会出现多种表面缺陷，这些缺陷不仅影响外观，还可能对电路板的性能造成损害。
因此，检测 PCB 表面缺陷对于品质管控至关重要。
- 传统的缺陷检测方式主要依赖人工视觉检查，存在主观性强、效率低下等问题。
- 深度学习技术的快速发展提供了更高效且精准的缺陷检测方案，然而，这需要大量且多样的数据集来进行模型训练。
- 该研究建立了一个包含 9 种 PCB 表面缺陷类别的数据集（DsPCBSD+），这些缺陷根据其成因、位置和形态进行分类，总计收集了 10,259 张图像，并手动注解了 20,276 个缺陷 ( 一张图像可能包含多种缺陷 )。
该数据集旨在推动基于深度学习的 PCB 表面缺陷检测研究。

## 2. 方法

- 缺陷图像收集：来自实际生产的内层和外层 PCB 的缺陷图像，通过自动光学检测（AOI）设备收集，图像经过预处理，去除了噪音，并增强了对比度和亮度，最终生成 32,259 张图片，图像大小为 226 x 226 像素。

- 缺陷分类与数据预处理：基于缺陷的成因（如 铜残留 Copper residue、铜不足 Copper deficiency、导体刮痕 Conductor scratch 和 异物 Foreign object）进行分类，并进一步细分为 9 个类别：

  - 短路 (Short) - SH
  - 尖刺 (Spur) - SP
  - 杂铜 (Spurious copper) - SC
  - 断路 (Open) - OP
  - 鼠咬 (Mouse bite) - MB
  - 孔崩 (Hole breakout) - HB
  - 导体刮痕 (Conductor scratch) - CS
  - 导体异物 (Conductor foreign object) - CFO
  - 基材异物 (Base material foreign object) - BMFO

  ![image](doc/PCB_surface_defect_classification.png)

- 缺陷标记与数据集划分：利用 LabelImg 工具对每个缺陷进行手动边界框注解，过滤掉无缺陷的图像、重复缺陷图像和不完整的缺陷图像。最终生成包含 20,276 个缺陷标记的 10,259 张图像。数据集被划分为训练集和验证集，比例为 8:2。

## 3. 数据集的技术验证

研究选用了两种 SOTA 模型（Co-DETR 和 YOLOv6-L6）进行验证。训练过程中对超参数进行了调整，如批量大小和学习率等。验证结果显示，这些模型在检测 PCB 表面缺陷时表现出色，平均精度（mAP）均高于 0.84，显示出该数据集对于深度学习模型的实用性和可靠性。
结果显示，这些模型在多数缺陷类别上都有较高的检测准确性，特别是在处理大范围缺陷时表现良好，但对于小型缺陷的检测仍有挑战。


## 4. 数据集的使用说明

数据集提供了 YOLO 和 COCO 格式，存储了训练和验证数据的图像和标签文件。
标签文件包含了缺陷的类别、边界框的坐标及其尺寸，这些数据可直接用于训练深度学习模型。

  ![image](doc/Count_of_three_size_labels.png)
  ![image](doc/distribution_of_defect_labels_in_dataset_across_different_categories.png)

以下使用 Python 搭配 Seaborn 套件进行数据可视化绘图
  ![image](tools/Figure_v5.png)
> tools > visualization.py


## 5. 数据集的优势与局限性

- 该数据集克服了以往数据集的一些不足，例如缺陷样本不足、标签不均衡等问题，为基于深度学习的 PCB 缺陷检测研究提供了有力支持。
- 该数据集的局限性在于，它仅包含 2D 图像，无法检测如凸起或凹陷等三维缺陷。
- 影像来自于 PCB 的内外层，未包含到焊接遮蔽后的缺陷图像
- 所有图像皆为局部区域的裁剪，未整合成整板图像，这在实际应用中可能需要额外处理来定位缺陷

### Ref : A Dataset for Deep Learning-Based Detection of Printed Circuit Board Surface Defect


支持的缺陷类别：
Dry_joint (干焊)
Incorrect_installation (安装错误)
PCB_damage (PCB 损坏)
Short_circuit (短路)