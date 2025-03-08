---
title: InternVL系列解读
date: 2025-02-03
tags:
  - InternVL
  - MLLM
---
InternVL系列教程：<https://internvl.readthedocs.io/en/latest/index.html>
# InternVL
发表于2023.12

Paper: [InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks](https://arxiv.org/pdf/2312.14238)

Tutorial: <https://internvl.readthedocs.io/en/latest/internvl1.0/classification.html>
![internvl-arch](internvl-arch.png)



## 训练策略：
### Stage-1: Vision-Language Contrastive Training
Visual Encoder: InterViT-6B，随机初始化权重

LLM: 多语言的LLaMA-7B（直接使用训练好的权重）

损失：对称的图文对比损失

数据：web scale, noisy image-text pairs，将6.03B图文对清洗之后还剩下4.98B

Visual Encoder和LLM的权重都更新
### Stage-2: Vision-Language Generative Training
QLLaMA：使用Stage-1中的LLaMA-7B的权重

冻结Visual Encoder和LLM的权重，只更新**新增加的权重：learnable queries and cross attention layers**

数据：将上述4.98B数据进一步清洗，得到1.03B数据

损失：类似BLIP2的image-text contrastive loss(ITC)，image-text match loss(ITM)和image-grounded text generation loss(ITG)
### Stage-3: SFT
数据：大约4M的高质量的指令数据

连接LLM： 2种连接方式：

    1. 可以将另外一个LLM（如Vicuna-13B或者InternLLM）通过一个MLP连接到Visual Encoder上
    2. 将LLM通过MLP连接到QLLaMA上，QLLaMA的权重不需要更新
训练：可以只更新MLP的权重，也可以更新MLP + LLM的权重

# InternVL-1.1
发表于2024.01.24 Blog: [InternVL 1.1: Enhance Chinese and OCR Capabilities](https://internvl.github.io/blog/2024-01-24-InternVL-1.1)

Tutorial: <https://internvl.readthedocs.io/en/latest/internvl1.1/introduction.html>
![internvl-1-1](internvl-1-1-arch.png)

发布了InternVL-Chat-V1-1 和 InternViT-6B-448px-V1-0 2个模型

模型架构类似LLaVA，使用MLP连接InternViT-6B和LLaMA2-13B，形成1个19B的模型

分辨率变化：图像输入变成448 x 448，会生成1024个token，使用Pixel Shuffle降低到256个token；

增强OCR能力和中文能力

## 训练
### Stage-1: Pretraining 
只更新ViT和MLP的权重，ViT使用InternVL-1.0的中的InternViT-6B（224 x 224)的权重初始化
### Stage-2: SFT
只更新MLP和LLM的权重
# InternVL-1.2
发表于2024.02.12，比LLaVA-NeXT稍晚一点，受到LLaVA-NeXT-34B的启发，使用了更大的LLM: [Nous-Hermes-2-Yi-34B](https://huggingface.co/NousResearch/Nous-Hermes-2-Yi-34B)

Blog: [InternVL 1.2: Scaling up LLM to 34B](https://internvl.github.io/blog/2024-02-21-InternVL-1.2/)

Tutorial: <https://internvl.readthedocs.io/en/latest/internvl1.2/introduction.html>

![internVL-1-2-arch](internVL-1-2-arch.png)

## 训练
### Stage-1: Pretraining
和InternVL-1.1相同，只更新ViT和MLP的权重
### Stage-2: SFT
此处和InterVL-1.1不同，全模型训练（40B）

Model Card
InternVL-Chat-V1-2使用的是1.2M的SFT数据，InternVL-Chat-V1-2-Plus使用的是12M的SFT数据
<table>
    <thead>
        <tr>
            <th>Type</th>
            <th>Model</th>
            <th>Date</th>
            <th>Download</th>
            <th>Note</th>
        </tr>
    </thead>
<tbody>                                         
<tr>
    <td rowspan="2">Vision Large Language Model</td>
    <td>InternVL-Chat-V1-2-Plus</td>
    <td>2024.02.21</td>
    <td>🤗 <a href="https://huggingface.co/OpenGVLab/InternVL-Chat-V1-2-Plus" rel="nofollow">HF link</a></td>
    <td>more SFT data and stronger</td>
</tr>
<tr>
    <td>InternVL-Chat-V1-2</td>
    <td>2024.02.11</td>
    <td>🤗 <a href="https://huggingface.co/OpenGVLab/InternVL-Chat-V1-2" rel="nofollow">HF link</a></td>
    <td>scaling up LLM to 34B</td>
</tr>
<tr>
    <td>Vision Foundation Model</td>
    <td>InternViT-6B-448px-V1-2</td>
    <td>2024.01.30</td>
    <td>🤗 <a href="https://huggingface.co/OpenGVLab/InternViT-6B-448px-V1-2" rel="nofollow">HF link</a></td>
    <td>vision foundation model, 448 resolution</td>
</tr>
</tbody>
</table>

# InternVL-1.5
发表于 2024.04.25

Paper: [How Far Are We to GPT-4V? Closing the Gap to Commercial Multimodal Models with Open-Source Suites](https://arxiv.org/abs/2404.16821)

Blog: [InternVL 1.5: How Far Are We to GPT-4V? Closing the Gap to Commercial Multimodal Models with Open-Source Suites](https://internvl.github.io/blog/2024-04-30-InternVL-1.5/)
## Model Card
<table>
    <thead>
    <tr>
    <th>Type</th>
    <th>Model</th>
    <th>Date</th>
    <th>Download</th>
    <th>Note</th>
    </tr>
    </thead>
    <tbody>
<tr>
    <td rowspan="2">Vision Large Language Model</td>
    <td>InternVL-Chat-V1-5-Int8</td>
    <td>2024.04.28</td>
    <td>🤗 <a href="https://huggingface.co/OpenGVLab/InternVL-Chat-V1-5-Int8" rel="nofollow">HF link</a></td>
    <td>The INT8 version of InternVL-Chat-V1-5</td>
</tr>
<tr>
    <td><nobr>InternVL-Chat-V1-5</nobr></td>
    <td>2024.04.18</td>
    <td>🤗 <a href="https://huggingface.co/OpenGVLab/InternVL-Chat-V1-5" rel="nofollow">HF link</a></td>
    <td>support 4K image; super strong OCR; Approaching the performance of GPT-4V and Gemini Pro on various benchmarks like MMMU, DocVQA, ChartQA, MathVista, etc. (🔥new)</td>
</tr>
<tr>
        <td>Vision Foundation Model</td>
        <td><nobr>InternViT-6B-448px-V1-5</nobr></td>
    <td>2024.04.20</td>
    <td>🤗 <a href="https://huggingface.co/OpenGVLab/InternViT-6B-448px-V1-5" rel="nofollow">HF link</a></td>
    <td>support dynamic resolution, super strong OCR (🔥new)</td>
</tr>
</tbody>
</table>

![internVL-1.5-arch](internVL-1.5-arch.png)

## 动态高分辨率
按照aspect ratio将原始图分成多个448 x 448的patches + 原始图的缩放到448 x 448

训练的时候：使用最多12个；
推理的时候：最多可以使用40个（支持4K分辨率）

使用Pixel Shuffle降低将单个patch产生的token数从1024降低到256
，即448 x 448的patch/image使用256个token表示

## 训练
### InternVL-Chat-V1-5 (26B)
LLM: InternLM2-20B

Stage-1: 预训练 ViT + MLP

Stage-2: 5M高质量的双语数据，和InterVL-1.2相同，ViT + MLP + LLM 都进行微调

### InternVL-Chat-V1-5-Plus（40B）
LLM: [Nous-Hermes-2-Yi-34B](https://huggingface.co/NousResearch/Nous-Hermes-2-Yi-34B)

Stage-1: 预训练阶段只微调 MLP

Stage-2: 同InternVL-Chat-V1-5，使用5M高质量的双语数据，对ViT + MLP + LLM都微调

# Mini-InterVL 1.5
发表于2024.05.25的Blog: [Mini-InternVL 1.5: A Powerful Pocket Multimodal Model with 8% Parameters for 80% Performance](https://internvl.github.io/blog/2024-05-25-Mini-InternVL-1.5)

## Model Card
<table>
    <thead>
    <tr>
    <th>Type</th>
    <th>Model</th>
    <th>Date</th>
    <th>Download</th>
    <th>Note</th>
    </tr>
    </thead>
<tbody>
<tr>
    <td rowspan="2">Vision Large Language Model</td>
    <td>Mini-InternVL-Chat-2B-V1-5</td>
    <td>2024.05.19</td>
    <td>🤗 <a href="https://huggingface.co/OpenGVLab/Mini-InternVL-Chat-2B-V1-5" rel="nofollow">HF link</a></td>
    <td>🚀🚀 Only 2B parameters, anyone can deploy it locally.</td>
</tr>
<tr>
    <td>Mini-InternVL-Chat-4B-V1-5</td>
    <td>2024.05.28</td>
    <td>🤗 <a href="https://huggingface.co/OpenGVLab/Mini-InternVL-Chat-4B-V1-5" rel="nofollow">HF link</a></td>
    <td>🚀 Only 4B parameters, anyone can deploy it locally.</td>
</tr>
<tr>
    <td>Vision Foundation Model</td>
    <td>InternViT-300M-448px</td>
    <td>2024.05.25</td>
    <td>🤗 <a href="https://huggingface.co/OpenGVLab/InternViT-300M-448px" rel="nofollow">HF link</a></td>
    <td>Distilled small vision foundation model with 300M parameters.</td>
</tr>
</tbody>
</table>

## 模型结构
结构和InternVL-1.5相同
### Visual Encoder
将InternViT-6B-448px-V1-5蒸馏到300M，即InternViT-300M-448px
### LLM
InternLM2-Chat-1.8B 或者 Phi-3-mini-128k-instruct (3.8B)
### 训练方法
和InternVL-1.5类似，使用同样的数据，2B的模型和26B的模型训练方式一样，4B的模型和40B的训练方法一样

# InternVL-2.0
发表于2024.07.02的Blog：[InternVL2: Better than the Best—Expanding Performance Boundaries of Open-Source Multimodal Models with the Progressive Scaling Strategy](https://internvl.github.io/blog/2024-07-02-InternVL-2.0/)

模型大小有：1B, 2B, 4B, 8B, 26B, 40B, 76B, 108B，其中8B及以下的模型使用InternViT-300M-448px，26B及以上的模型使用InternViT-6B-448px-V1-5

支持多模态输入（图像，文本，视频，医疗数据），多任务输出（图，bbox，mask）

## 训练方法
### Stage-1
在InternVL-1.5的数据上做了扩充，只微调MLP
### Stage-2
InternVL-1.5的5M高质量的双语数据
 ViT + MLP + LLM

Progressive with larger language models？

# InternOmini
发表于2024.07.27的Blog: [InternOmni: Extending InternVL with Audio Modality](https://internvl.github.io/blog/2024-07-27-InternOmni/)，增加了对Audio的处理

# Mono-InternVL
发表于2024.10.10 

Paper: [Mono-InternVL: Pushing the Boundaries of Monolithic Multimodal Large Language Models with Endogenous Visual Pre-training](https://arxiv.org/abs/2410.08202)

Blog: [Mono-InternVL: Pushing the Boundaries of Monolithic Multimodal Large Language Models with Endogenous Visual Pre-training](https://internvl.github.io/blog/2024-10-10-Mono-InternVL/)

TODO，详细内容以后再说

# Mini-InternVL-2.0
发表于2024.10.21

Paper: [Mini-InternVL: A Flexible-Transfer Pocket Multimodal Model with 5% Parameters and 90% Performance](https://arxiv.org/abs/2410.16261)

Blog: [Mini-InternVL 2.0: A Flexible-Transfer Pocket Multimodal Model with 5% Parameters and 90% Performance
](https://internvl.github.io/blog/2024-10-21-Mini-InternVL-2.0)

发布了1B, 2B和4B的模型，其中<font color=red>4B的模型使用5%的参数实现了InternVL2-Llama3-76B 90%的性能</font>

![mini-internvl-2.0-arch](mini-internvl-2.0-arch.png)
使用CLIP-ViT-L/336px(300M)初始化InternViT-300M，然后使用InternViT-6B将知识蒸馏到InternViT-300M

InternViT-300M输入是448px，采用动态高分辨率，每个448 x 448的patch产生1024个token，经过Pixel Unshuffle降低到256个Token

## 训练
### Stage-1: 
在InternVL-1.5的扩展数据上进行训练，对于1B和2B模型训练ViT + MLP，对于4B的模型，只训练MLP
### Stage-2:
使用InterVL-1.5的5M高质量的双语数据，对整个模型的参数都做更新（即ViT + MLP + LLM）

# InternVL-2.5
发表于2024.12.6

Paper: [Expanding Performance Boundaries of Open-Source Multimodal Models with Model, Data, and Test-Time Scaling](https://arxiv.org/abs/2412.05271)

Blog: [InternVL2.5: Expanding Performance Boundaries of Open-Source Multimodal Models with Model, Data, and Test-Time Scaling](https://internvl.github.io/blog/2024-12-05-InternVL-2.5)

模型结构和InternVL-1.5，InternVL-2.0都一样

不同类型的数据，使用不同的训练数据格式
![internvl-2.5-data-format](internvl-2.5-data-format.png)

## 训练方法
![internvl-2.5-train](internvl-2.5-train.png)
每个模型都分为3个阶段的训练，3个阶段都是使用的**NTP Loss**
### stage-1: MLP WarmUp
只更新MLP的参数
### Stage-2: ViT Incremental Learning (Optional)
更新ViT + MLP的参数
### Stage-3: Full Model Instruct Learning
更新整个模型（ViT + MLP + LLM）的参数

### 渐近缩放策略（Progressive Scaling Strategy）
在1.5和2.0都提到了，该方法来自于 <u>“even when the ViT and LLM are jointly trained using NTP loss, the resulting visual features are generalizable representations that can be easily understood by other LLMs.”</u>的观察

使用Stage-1训练好1个参数量小的MLLM的的InternViT，可以直接将该InternViT接入到1个更大参数量的MLLM的Stage-1.5阶段，从而使得更大参数量的MLLM省去Stage-1的训练，达到提升训练的目的；

然后进行正常的Stage-2的全参微调

## 训练增强
### 随机JPEG压缩
对图像进行75～100质量的JPEG压缩
### Loss Reweighting
常用的NTP LOSS重采样有**Token平均**和**样本平均**
![internvl-2.5-loss-reweighting](internvl-2.5-loss-reweighting.png)

token平均： $\frac{1}{x^0}$恒等于1，即$w_i$恒等于1，那么每个token对NTP Loss的贡献都是一样的，会导致梯度偏差到更长的token生成上

样本平均：$\frac{1}{x^1}$，会确保每个样本对与NTP Loss的贡献是一样的，但是会导致模型更偏爱产生更短的response

为了减轻在训练中产生更长或者更短的response的偏差，本文采用square averaging的reweighting strategy：即 $w_i = \frac{1}{x^{0.5}}$

## 多模态数据Packing技术
在2.0和2.5的训练中使用，用于增加训练效率；

通过Select、Search、Pack、Maintain这4个阶段实现多模态数据的高效训练；

如果1个样本的token比较短，无法占用LLM一次能够处理的最大token数目，那么就会在数据集里面再搜索1个token数较少的样本，将它们拼接在一起，在计算注意力的时候，每个样本只会对自己的token计算注意力

---
论文还发现数据质量相比于数量更重要；对于数据进行清洗，能够带来可观的改善；

本文观察到LLM比Vision Encoder对噪声更敏感，即使一小部分的异常样本就会在Stage-2的训练中导致模型和用户体验退化。在所有类型的异常中，重复生成是最严重的问题。

在比较难的任务如MMMU上，在推理的时候COT很有用；而且论文还验证了COT和多数投票一起使用效果更好。

# InterVL-2.5-MPO
发表于2024.11.05

Paper: [Enhancing the Reasoning Ability of Multimodal Large Language Models via Mixed Preference Optimization](https://arxiv.org/abs/2411.10442)

Blog: [InternVL2.5-MPO: Enhancing the Reasoning Ability of Multimodal Large Language Models via Mixed Preference Optimization](https://internvl.github.io/blog/2024-12-20-InternVL-2.5-MPO/)

使用MPO之后比不使用MPO的模型在OpenCompass Leaderboard上平均高2个点左右。

## MMPR
MMPR即Multi-Modal Preference Dataset，本文提出了一个高效的便好数据构建pipeline，基于该pipeline构建了一个高质量、大规模的、3M的多模态推理便好数据集.

- 对于有明确标准答案的样本：模型首先被prompt输出推理过程和生成格式为"Final Answer: xx"的最终答案；和gt match的response被认为是positive set $\mathcal{Y}_p$ , 不匹配的被认为是negative set $\mathcal{Y}_n$ ,此外那些not clear response也被合并进 $\mathcal{Y}_n$ . 假设response label被认为是positve和negative，那么可以通过从$\mathcal{Y}_p$和$\mathcal{Y}_n$各选择1个response作为preference pair.
- 对于没有明确标准答案的样本：提出一个简单高效的方法：Dropout NTP(Dropout Next Token Prediction), 具体的讲：使用InternVL2-8B生成的response作为chosen answer。对于该chosen answer, 在一半的地方截断它，然后promptInternVL2-8B对截断的答案补全（不使用图像输入），生成的answer就作为paired sample中的reject answer。值得指出的是：InternVL2-8B生成的答案可能是不完美的，而且InternVL2-8B在不使用图像输入的时候补全的答案会包含更多的幻觉，因此才能购在choosen answer和reject answer之间的偏序关系保持为true。

## MPO
MPO: Mixed Preference Optimization

MPO的关键在于：一个有效的 PO 流程应当使模型能够学习成对响应之间的相对偏好、单个响应的绝对质量以及生成更优响应的过程。

[//]: todo,补充损失的形式
