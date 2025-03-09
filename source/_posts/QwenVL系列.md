---
title: QwenVL系列阅读
date: 2025-03-01
tags: QwenVL, MLLM
---
QwenVL系列是阿里推出的一系列多模态大语言模型，在多个领域表现优秀
# QwenVL
Paper: [Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond](https://arxiv.org/abs/2308.12966)，发表于23.08

Code: <https://github.com/QwenLM/Qwen-VL>

Blog: <https://qwenlm.github.io/blog/qwen-vl/>

共开源2个模型：QwenVL-7B 和 QwenVL-Chat-7B

闭源的QwenVL-PLUS和QwenVL-MAX：

- QwenVL-PLUS： 增强了细节识别能力和文本识别能力，支持更高的分辨率
- QwenVL-MAX：相比于PLUS版本，进一步增强了视觉推理和指令跟随能力，提供更高层次的视觉感知和认知理解能力

**QwenVL-PLUS和QwenVL-MAX**都是闭源的，提供API调用

![QwenVL-arch](QwenVL-arch.png)

- Vision Encoder: openAI-ViT/bigG(14px)
- Position-aware Vision Language Adapter: 类似BLIP2的Q-Former，使用256个learnable Query Embeddings作为CrossAttn的query，ViT的图像特征作为key，2D-绝对位置编码被整合进query-key中。
- LLM: Qwen-7B

## Train
分为**预训练、多任务预训练和SFT**共3个阶段进行训练
### 1. Pre-training
使用了1.4B的数据（77.3的英文文本，22.7%的中文文本）

图像分辨率 256 x 256，特征长度256

冻结LLM，只训练ViT和Adapter，使用Cross Entropy Loss
### 2. Multi-task Pre-training
高质量细粒度的更大分辨率（448*448）和interleaved image-txt data 视觉语言标注数据

图像分辨率从 256 x 256 增加到 448 x 448，特征长度变成1024

全参训练，也使用Cross Entropy Loss

### 3. SFT
目的：增强指令跟随和对话能力，产出Qwen-VL-Chat

使用混合的多模态数据和纯文本数据来确保模型的通用对话能力，大约用了350K的数据

只训练Adapter和LLM，

## 输入数据格式
使用\<img\>和\</img\>作为图像特征序列的开始和结尾

bbox：bbox被缩放到[0, 100]，使用这样"($X_{topleft}, Y{topleft}$),($X_{bottomright}, Y_{bottomright}$)"格式的字符串表达bbox，使用\<box\>和\</box\>表示box的开始和结束，使用\<ref\>和\</ref\>标记box所引用的内容, 具体样例如下

Vision Question Answering的数据格式：  \<img\>VG_100K_2/1.jpg\</img\> Does the bandage have a different color than the wrist band?
Answer: <font color=blue>No, both the bandage and the wrist band are white.\<eos\> </font>

Caption with Grounding的数据格式：\<img\>coyo700m/1.jpg\</img\>Generate the caption in English with grounding: <font color=blue>Beautiful shot of
\<ref\>bees\</ref\>\<box\>(661,612),(833,812)\</box\>\<box\>(120,555),(265,770) \</box\> gathering
nectars from \<ref\>an apricot flower\</ref\>\<box\>(224,13),(399,313) \</box\>\<eos\></font>

# Qwen2-VL

Paper: [Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution](https://arxiv.org/abs/2409.12191), 发表于24.09

Blog: <https://qwenlm.github.io/zh/blog/qwen2-vl>

开源2B, 7B 和 72B，共3个模型

超越和相当GPT-4o，Claude-3.5-Sonnet

- **支持不同分辨率和不同长宽比的图**
- **理解20分钟以上的长视频**，基于视频的问答、对话和内容创作等应用中。
- **能够操作手机和机器人的视觉智能体**
- **多语言支持**，除中文和英文之外，还支持大多数欧洲语言、日语、韩语、阿拉伯语、越南语等。

## 模型结构
![qwen2vl-arch](qwen2vl-arch.png)

- Vision Encoder: 相比QwenVL更小了，在2B、7B和72B上都使用600M规模的ViT，使用[DFN](https://arxiv.org/abs/2309.17425)的ViT进行初始化，但是将DFN中的绝对位置编码替换成了RoPE。
- LLM: Qwen2系列
- 原生动态分辨率 Naive Dynamic Resolution

    对ViT进行修改，删掉了原始的绝对位置编码，引入了2D-RoPE，

    为了降低图像的token数目，使用一个MLP，将2x2的特征转换成1个

- 多模态旋转位置嵌入（M-RoPE: MultiModal Rotory Position Embedding）
    
    ![qwen2vl-mrope](qwen2vl-mrope.png)

    传统的旋转位置嵌入只能捕捉一维序列的位置信息，而 M-ROPE 通过将原始旋转嵌入分解为代表时间、高度和宽度的三个部分，使得大规模语言模型能够同时捕捉和整合一维文本序列、二维视觉图像以及三维视频的位置信息。

    对于文本，M-RoPE和1D-RoPE等价；对于图像，时域ID保持不变，只对图像宽高进行编码；对于视频，时域、宽、高3个维度都进行编码；

- 统一的图像和视频理解

    使用深度为2的3D卷积处理视频，使之能够不增加特征序列长度的同时能够处理更多帧的视频。为了和视频统一，图像被视作2帧相同的视频进行处理；最长可以处理16384个token

## 训练

和qwenvl 相同的3阶段的训练
### 1 预训练
ViT + Adapter，使用image-text pairs, optical character recognition
(OCR) data, interleaved image-text articles, visual question answering datasets, video dialogues, and image
knowledge datasets；模型共计看到了600B的token
### 2 多任务预训练
全部参数，使用额外的800B token的图像相关的数据
### 3 SFT
Adapter + LLM，不仅使用纯文本对话，也使用了多模态对话数据；

# QwenVL-2.5

Paper: [Qwen2.5-VL Technical Report](https://arxiv.org/abs/2502.13923) ， 发表于2025.02

Blog: <https://qwenlm.github.io/zh/blog/qwen2.5-vl>

开源了Base和Instruct模型，共计3B，7B和72B的模型大小。


Qwen2.5-VL 的主要特点如下所示：

- 感知更丰富的世界：Qwen2.5-VL 不仅擅长识别常见物体，如花、鸟、鱼和昆虫，还能够分析图像中的文本、图表、图标、图形和布局。

- Agent：Qwen2.5-VL 直接作为一个视觉 Agent，可以推理并动态地使用工具，初步具备了使用电脑和使用手机的能力。

- 理解长视频和捕捉事件：Qwen2.5-VL 能够理解超过 1 小时的视频，并且这次它具备了通过精准定位相关视频片段来捕捉事件的新能力。

- 视觉定位：Qwen2.5-VL 可以通过生成 bounding boxes 或者 points 来准确定位图像中的物体，并能够为坐标和属性提供稳定的 JSON 输出。

- 结构化输出：对于发票、表单、表格等数据，Qwen2.5-VL 支持其内容的结构化输出，惠及金融、商业等领域的应用。

![qwenvl2.5-vl-arch](qwenvl2.5-vl-arch.png)

## 模型结构
- LLM：使用Qwen2.5系列，为了更多满足多模态理解的要求，将1D RoPE（Rotary Position Embedding）修改为与绝对时间对齐的多模态RoPE。
- Vision Encoder：一个重新设计过的ViT，整合了2D-RoPE和Window Attention，支持原生分辨率的同时加速整个视觉encoder的计算。在训练和计算的时候，图像的宽高都被resize到28的倍数，ViT使用的patch大小是14.
- MLP-based Vision-Language Merger：首先将ViT输出的特征，在空间上相邻的4个patches的特征进行分组，然后将分组的特征concatenate起来，送到一个2层MLP，将其投影到和LLM对齐的维度。这种方式不仅减少了计算量，而且还提供了一种灵活的方式动态的压缩不同长度的图像特征序列。

### 1. Fast and Efficient Vision Encoder
重新设计了ViT，引入**Window Attention**，只有4个层使用Full Self-Attention，其余的层都使用最大window size是112 x 112（对应到8 x 8的patches）的**Window Attention**，小于112 x 112的区域不使用padding，保持其原有分辨率。这样的设计可以使模型在输入分辨率上更自然的处理，避免不必要的缩放和失真。

对于Position Encoding，采用2D RoPE(Rotary Position Embedding)在2D空间上捕获空间关系。而且为了更好的处理视频，将该方法拓展至3D patch 划分。具体地说，我们使用14 x 14的patch大小作为基本单元，对于视频数据，2个连续的视频帧会组合在一起，显著减少了送入LLM的token数目。这样的设计不仅和现有的架构保持一致，而且增强了处理序列视频数据的效率。

为了简化整个网络结构，将ViT架构与LLMs的设计原则更紧密的对齐，使用**RMSNorm**进行归一化，并使用**SwiGLU**作为激活函数。

训练ViT：从头开始训练。训练过程包括3个阶段：CLIP预训练，Vision-Language 对齐和end-to-end fine tuning。

### 2. Native Dynamic Resolution and Frame Rate
空间领域上，直接使用原始的绝对坐标来表示box，点的位置；不再使用缩放的坐标系了。

对于视频输入：使用动态帧率（Dynamic Frame Rate）和绝对时间编码（absolute time encoding），对齐MRoPE IDs和时间戳，这种方法使模型能够通过时间维度 ID 之间的间隔理解时间的节奏，而无需任何额外的计算开销。

### 3. Multimodal Rotary Position Embedding Aligned to Absolute Time
在Qwen2-VL中引入的MRoPE，但是对于视频输入，MRoPE在时域上与视频帧数目相关，无法表达视频内容变化的速度或视频中时间的绝对时间。因此在Qwen2.5-VL中，将MRoPE中的时域分量和绝对时间对齐。通过利用时间ID之间的间隔，模型能够学习在不同FPS采样率的视频之间保持一致的时间对齐。

## 训练
总体分为预训练和后训练2大阶段，其中Pre-Training又分为3个阶段：Visual Pre-Training， Multimodal Pre-Training和Long-Context Pre-Training；Post-Traning分为2个阶段：SFT和DPO

### 1. Pre-Traning
![qwenvl2.5-arch-pretraining](qwenvl2.5-arch-pretraining.png)
根据输入到LLM的序列的长度，动态打包数据样本；在第1和第2阶段，数据被均匀的packed到8192的长度，但是在第3阶段，序列长度增加到32768。
### 2. Post-Training
在SFT和DPO过程中，ViT都是frozen的。

SFT使用多样化的多模态数据，包括image-text pairs, video, and
pure text, sourced from general VQA, Rejection Sampling, and specialized datasets such as Document
and OCR, Grounding, Video, and Agent-related tasks. 

DPO只使用纯文本数据和图文数据。
