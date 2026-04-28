如果我是导师，我不会让学生“刷平台”，而是把四个平台分成四种能力训练场：

**Kaggle = 比赛闭环与误差分析**；**Hugging Face = 预训练模型、数据集、部署与模型卡**；**GitHub = 工程化、复现、协作和开源贡献**；**Papers with Code = 论文—代码—指标—榜单的科研入口**。Kaggle 官方文档覆盖 competitions、notebooks、datasets、benchmarks 等实践模块；Hugging Face CV 课程覆盖 CNN、Vision Transformer、多模态、生成模型、基础CV任务、视频、3D视觉、模型优化等内容；GitHub Skills 入门课会训练仓库、分支、提交、Pull Request 和合并流程。([Kaggle](https://www.kaggle.com/docs?utm_source=chatgpt.com "Getting Started on Kaggle | Kaggle"))

## 一、入门阶段：先让学生完成“视觉项目最小闭环”

### 1. Kaggle：从提交一次结果开始

我会让学生做三个入门项目：

| 项目                           | 训练目标                                                        | 学生必须交付                                            |
| ---------------------------- | ----------------------------------------------------------- | ------------------------------------------------- |
| **Digit Recognizer / MNIST** | 理解图像分类、CNN、训练/验证/提交流程；Kaggle 也把它描述为 CV 的 “hello world” 入门任务 | baseline notebook、混淆矩阵、错误样本分析、一次有效提交              |
| **Dogs vs Cats**             | 学会真实图像读取、数据增强、迁移学习、过拟合控制                                    | ResNet/EfficientNet baseline、augmentation 对比、错误案例 |
| **CIFAR-10 图像分类**            | 比较 CNN、ResNet、ViT；理解小图像分类基准                                 | 3个模型对比、参数量/速度/准确率表格、README                        |

Digit Recognizer 适合“第一次跑通”，Kaggle 明确把它定位给有 Python/ML 基础但新接触 CV 的学习者；CIFAR-10 是 10类、60,000张 32×32 彩色图像的经典数据集，适合学生做模型对比。([Kaggle](https://www.kaggle.com/competitions/digit-recognizer?utm_source=chatgpt.com "Digit Recognizer - Kaggle"))

我会规定：**每个 Kaggle 项目不许只追分数，必须写 error analysis**。例如：模型错在哪些类别？是姿态、光照、遮挡、背景偏差，还是标签噪声？

### 2. Hugging Face：从“会用模型”到“会发布模型”

入门任务我会这样布置：

|项目|训练目标|学生必须交付|
|---|---|---|
|**用 ViT 微调图像分类模型**|理解 Transformer 如何处理图像 patch；掌握 processor、trainer、metrics|HF model repo、model card、推理 demo|
|**用 pipeline 做零样本图像分类/分割 demo**|学会快速验证预训练模型能力|Gradio/Spaces 小应用|
|**上传一个小型自定义数据集**|学会 dataset card、数据许可、标签说明|dataset card + 数据清洗脚本|

Hugging Face 的官方文档中，ViT 被定义为“适配计算机视觉任务的 Transformer”，会把图像切成固定大小 patch，类似 NLP 中的 token；Hugging Face Hub 也提供模型、数据集和 Spaces，适合学生把项目从 notebook 变成可展示作品。([Hugging Face](https://huggingface.co/docs/transformers/main/en/model_doc/vit?utm_source=chatgpt.com "Vision Transformer (ViT) - Hugging Face"))

### 3. GitHub：从 notebook 学生变成工程学生

我会先要求每个学生完成 GitHub Skills 的入门练习，因为它会训练仓库、分支、commit、PR 和 merge 这些最基础的协作动作。([GitHub](https://github.com/skills/introduction-to-github "GitHub - skills/introduction-to-github: Get started using GitHub in less than an hour. · GitHub"))

然后规定所有项目都必须有这个结构：

```text
project-name/
  README.md
  requirements.txt 或 environment.yml
  data/README.md
  src/
    train.py
    evaluate.py
    infer.py
  notebooks/
  configs/
  outputs/
  reports/
```

入门 GitHub 项目案例：

|项目|目标|
|---|---|
|**把 Kaggle notebook 重构成可运行仓库**|从“比赛代码”变成“可复现代码”|
|**实现一个最小 CNN 分类框架**|学会 dataset、model、loss、optimizer、scheduler 的模块化|
|**给开源 CV repo 提一个文档 PR**|学会阅读 issue、复现 bug、写清楚 PR 描述|

### 4. Papers with Code：从“看论文标题”到“复现核心表格”

入门阶段我不会让学生一上来追最新 SOTA，而是让他们做经典论文阅读卡：

|论文/方向|学生要回答的问题|
|---|---|
|LeNet / AlexNet / VGG / ResNet|为什么深层网络能训练？残差连接解决了什么？|
|U-Net|encoder-decoder 和 skip connection 为什么适合分割？|
|ViT|图像 patch token 化之后，CNN 的归纳偏置少了什么、多了什么？|
|DETR|为什么目标检测可以被建模成 set prediction？|
|CLIP|为什么图文对比学习能带来 zero-shot 能力？|

Papers with Code / HF Papers 适合让学生从论文页面追踪 arXiv、GitHub 实现和社区讨论；页面中也会展示论文、GitHub 链接和 arXiv 链接，方便做“论文—代码—实验”闭环。([paperswithcode.co](https://paperswithcode.co/?utm_source=chatgpt.com "AI Papers - Discover Latest Research"))

---

## 二、进阶阶段：按任务线训练，而不是按平台训练

### 任务线 A：图像分类 → 细粒度识别 → 长尾识别

**Kaggle 项目：**

1. CIFAR-10 / Dogs vs Cats：基础分类。
    
2. Plant Pathology / iWildCam：细粒度识别、类别不平衡、domain shift。
    
3. 自己构建一个小型业务数据集：例如商品图分类、工业缺陷分类、农作物病害分类。
    

**Hugging Face 项目：**

1. Fine-tune ViT / ConvNeXt / Swin。
    
2. 上传模型到 Hub。
    
3. 做一个 Spaces demo，让别人上传图片得到分类结果。
    

**GitHub 项目：**

1. 写一个完整训练框架。
    
2. 支持配置文件。
    
3. 支持 TensorBoard/W&B 日志。
    
4. 支持 checkpoint、resume、early stopping。
    
5. 写清楚复现实验命令。
    

**Papers with Code 项目：**

1. 复现 ResNet 或 ViT 在 CIFAR-10 上的主要趋势。
    
2. 做 ablation：数据增强、学习率、权重衰减、batch size、预训练权重。
    
3. 写一份 4 页小论文式报告。
    

---

### 任务线 B：目标检测

目标检测我会从 **“检测什么 + 框在哪里 + 置信度多少”** 教起。Hugging Face 文档把目标检测定义为在图像中检测人、建筑、汽车等实例，并输出 bounding box 坐标和标签；这正适合让学生从分类过渡到空间定位。([Hugging Face](https://huggingface.co/docs/transformers/main/en/tasks/object_detection?utm_source=chatgpt.com "Object detection - Hugging Face"))

**入门项目：**

|项目|模型|重点|
|---|---|---|
|玩具数据：水果/交通标志检测|YOLO / Faster R-CNN|bbox 标注、mAP、NMS|
|RSNA Pneumonia Detection|YOLO / Faster R-CNN / DETR|医学影像检测、假阳性/假阴性分析|
|自定义商品检测|YOLO / RT-DETR|数据标注、部署速度、边缘设备推理|

RSNA Pneumonia Detection Challenge 要求算法在胸部 X 光片中自动定位肺部阴影，是很好的医学影像检测案例。([Kaggle](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/?utm_source=chatgpt.com "RSNA Pneumonia Detection Challenge - Kaggle"))

**进阶要求：**

学生不能只跑 YOLO。必须回答：

“为什么 YOLO 快？”  
“为什么 DETR 不需要传统 anchor/NMS 思维？”  
“mAP@0.5 和 mAP@0.5:0.95 有什么区别？”  
“医疗检测中，召回率和精确率哪个更重要？”

---

### 任务线 C：语义分割 / 实例分割 / 全景分割

分割任务适合培养学生对像素级预测的理解。Hugging Face CV 课程把图像分割解释为把图像划分为有意义的区域，并区分 semantic segmentation、instance segmentation 和 panoptic segmentation；同一课程还介绍 IoU、pixel accuracy、Dice 等常用评价指标。([Hugging Face](https://huggingface.co/learn/computer-vision-course/en/unit6/basic-cv-tasks/segmentation "Image Segmentation · Hugging Face"))

**入门项目：**

|项目|模型|重点|
|---|---|---|
|Oxford-IIIT Pet segmentation|U-Net / DeepLabV3|mask、IoU、Dice|
|2018 Data Science Bowl nuclei segmentation|U-Net / Mask R-CNN|生物医学分割、实例级目标|
|道路/建筑物遥感分割|SegFormer / U-Net|类别不平衡、小目标、边界质量|

2018 Data Science Bowl 要求参赛者建立模型识别不同条件下的细胞核，并用不同 IoU 阈值下的 mean average precision 评估，适合做进阶分割训练。([Kaggle](https://www.kaggle.com/competitions/data-science-bowl-2018?utm_source=chatgpt.com "2018 Data Science Bowl - Kaggle"))

**进阶项目：**

1. 用 SAM 做 zero-shot mask 生成。
    
2. 用 SegFormer 微调遥感或医学数据。
    
3. 比较 U-Net、DeepLabV3、SegFormer、SAM 的效果。
    
4. 专门分析边界错误、小目标漏检、mask 破碎问题。
    

---

### 任务线 D：多模态与视觉语言模型

这个阶段我会让学生从 CLIP 开始，而不是直接上复杂 VLM。

**项目案例：**

|项目|目标|
|---|---|
|CLIP zero-shot 图像分类|不训练模型，用 prompt 分类图片|
|图文检索系统|输入一句话，检索最相关图片|
|商品图搜索|用文本描述找商品图|
|视觉问答小 demo|了解 VQA、captioning、grounding|

进阶要求是：学生要比较 **closed-set 分类** 和 **open-vocabulary recognition** 的差异，理解为什么多模态模型改变了传统 CV 的范式。

---

### 任务线 E：生成式视觉与扩散模型

我不会让学生一开始就训练 Stable Diffusion，而是从小规模实验开始。

**项目案例：**

|层级|项目|
|---|---|
|入门|用预训练 diffusion model 做图像生成和风格迁移实验|
|中级|LoRA 微调一个小主题数据集|
|进阶|ControlNet / Inpainting / 图像条件生成|
|科研|比较生成图像的质量、多样性、可控性和偏差|

这个阶段的重点不是“生成好看的图”，而是理解 latent space、denoising、conditioning、guidance scale、LoRA、数据偏差和版权伦理。

---

## 三、我会设计一条 12 周训练路线

|周数|主题|平台重点|项目产出|
|---|---|---|---|
|1|Git、PyTorch、OpenCV、数据集格式|GitHub|一个可运行分类仓库|
|2|CNN + 图像分类|Kaggle|Digit Recognizer 提交|
|3|迁移学习 + 数据增强|Kaggle/HF|Dogs vs Cats 或 CIFAR-10|
|4|ViT 与模型发布|Hugging Face|ViT model card + Spaces demo|
|5|目标检测基础|Kaggle/GitHub|YOLO/DETR 检测项目|
|6|医学/工业检测|Kaggle|RSNA 或自定义检测数据|
|7|语义分割|HF/GitHub|U-Net/SegFormer 分割项目|
|8|实例分割与 SAM|HF/Papers|SAM/Mask R-CNN 实验报告|
|9|多模态 CLIP|HF|图文检索 demo|
|10|论文复现|Papers with Code/GitHub|复现一篇经典论文|
|11|消融实验与鲁棒性|GitHub/Kaggle|ablation 表格 + error analysis|
|12|最终项目答辩|四个平台整合|repo + report + demo + slides|

---

## 四、每个学生最终必须完成的“作品集标准”

我会要求他们毕业时至少有 4 个项目：

1. **Kaggle 项目**：一个有排行榜提交、有误差分析的视觉比赛项目。
    
2. **Hugging Face 项目**：一个上传到 Hub 的模型，包含 model card 和可交互 demo。
    
3. **GitHub 项目**：一个结构清晰、可复现、带 README 和训练脚本的开源仓库。
    
4. **Paper Reproduction 项目**：一篇 CV 论文的复现或简化复现，包含实验表格、消融和失败分析。
    

我最看重的不是分数，而是这五件事：

**能不能复现？能不能解释错误？能不能比较方法？能不能写清楚实验？能不能把模型交付给别人使用？**

---

## 五、给学生的进阶深造项目案例

### 项目 1：低资源医学图像分割

**题目**：只有少量标注时，U-Net、SegFormer、SAM 哪个更可靠？  
**平台**：Kaggle + Hugging Face + GitHub + Papers with Code  
**研究点**：few-shot segmentation、数据增强、伪标签、domain shift。  
**产出**：一份类似 workshop paper 的报告。

### 项目 2：开放词汇商品识别

**题目**：用 CLIP/ViT 做电商商品图检索和分类。  
**平台**：Hugging Face + GitHub  
**研究点**：prompt engineering、图文 embedding、hard negative mining。  
**产出**：一个商品图搜索 demo。

### 项目 3：遥感图像建筑物分割

**题目**：复杂背景下建筑物边界如何分割得更准？  
**平台**：Kaggle + Papers with Code  
**研究点**：边界损失、Dice loss、IoU loss、多尺度特征。  
**产出**：模型对比 + 可视化错误分析。

### 项目 4：野外动物识别中的 domain shift

**题目**：模型在不同摄像头、不同地区上为什么失效？  
**平台**：Kaggle iWildCam + GitHub  
**研究点**：长尾分类、类别不平衡、领域泛化。iWildCam 相关资料强调训练和测试来自全球不同摄像头位置，适合研究分布偏移。([arXiv](https://arxiv.org/abs/2105.03494?utm_source=chatgpt.com "[2105.03494] The iWildCam 2021 Competition Dataset - arXiv.org"))

### 项目 5：目标检测模型部署比较

**题目**：YOLO、RT-DETR、DETR 在速度、精度、部署复杂度上如何取舍？  
**平台**：GitHub + Hugging Face  
**研究点**：mAP、FPS、模型大小、ONNX/TensorRT、边缘部署。  
**产出**：benchmark 表格 + demo。

---

## 六、我会给学生的一句话建议

**Kaggle 训练你把模型跑起来，Hugging Face 训练你把模型用起来，GitHub 训练你把项目做扎实，Papers with Code 训练你把研究看明白。**

真正进阶的标志不是“我跑了很多模型”，而是：

> 我知道这个任务的评价指标是什么，知道 baseline 在哪里，知道模型为什么错，知道论文方法解决了什么问题，也知道如何把自己的改进复现、对比、发布和解释清楚。