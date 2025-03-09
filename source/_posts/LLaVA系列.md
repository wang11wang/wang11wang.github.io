---
title: LLaVA Series解读
date: 2025-02-22
tags: LLaVA MLLM
---
![LLaVA发展图](Llava系列.png)

# LLAVA
Paper: [Visual Instruction Tuning](https://arxiv.org/abs/2304.08485)

Code: <https://llava-vl.github.io/>

2023.04.17发布，使用1个Linear Matrix连接Vision Encoder和LLM
## 整体结构
![LLaVA架构](LLaVA_arch.png)

- Vision Encoder: VIT-L/14
- LLM: Vicuna

假设VIT-L/14提供的视觉特征使用 $Z_v = g(X_v)$ 表示，使用1个可训练的Projector Matrix $W$ 将投影到可以输入到LLM的矩阵 $H_v$，$H_v$和LLM接收的text token $H_q$ 具有相同的维度。
$$H_v = W \cdot Z_v , 其中Z_v = g(X_v)$$

### 训练损失
使用LLM的自回归损失，即
![LLaVA-formula-3](LLaVa-formula-3.png)
> 注意，损失并不是在所有token上计算的，只在下图中的绿色部分计算
![LLaVA-table-2](LLaVA-table-2.png)

### two stage training
#### Stage 1: Feature Alignment
只训练Linear Matrix
#### Stage 2: 
冻结Vision Encoder，训练Limear Matrix和LLM

### 数据
![LLaVA-数据生成](LLaVA-数据生成.png)
使用Context Type1和Context Type2对GPT-4进行prompt，然后让GPT-4输出Response的3种回答，作为训练数据；
> 图片并没有作为prompt输入到GPT-4中，放在这里只是方便距离而已；

# LLaVA-Med
Paper: [LLaVA-Med: Training a Large Language-and-Vision Assistant for Biomedicine in One Day](https://arxiv.org/abs/2306.00890) 

Code: <https://github.com/microsoft/LLaVA-Med?tab=readme-ov-file#archive>

2023.06.01发布，达到GPT-4级别的生物医学领域的MLLM
![LLaVA-Med-arch](LLaVA-Med-arch.png)
使用原始的LLaVA初始化模型，然后首先做**医学概念对齐**，然后做**医学指令微调**

- 医学概念对齐：冻结Vision Encoder和LLM，只微调Linear Matrix
- 医学指令微调：冻结Vision Encoder，微调Linear Matrix和LLM
# LLaVA-RLHF
Paper: [Aligning Large Multimodal Models with Factually Augmented RLHF](https://arxiv.org/abs/2309.14525)

Code: <https://llava-rlhf.github.io/>

发表于2023.09，是第一个开源的RLHF-trained的通用的MLLM。

RLHF: 即 Reinforcement Learning from Human Feedback

PPO: 即Proximal Policy Optimization，近端策略优化

![LLaVA-RLHF](LLaVA-RLHF.png)

## 3个训练阶段
### Stage-1: SFT
- 在CC3M的子集上，使用Linear Matrix做特征对齐
- 在Attention和FFN上做Lora(r=64), 使用的数据是 Visual Chat and HQ Multimodal Instruction:， 具体是 90k LLaVA-Instruct task, 83k VQA-v2 and 16k A-OKVQA multi-round QA task, and 23k Flickr30k Spotting Caption task，得到 LLaVA-SFT<sup>+</sup>
- 得到的模型：

    - LLaVA-SFT<sup>+</sup>-7b: Vicuna-V1.5-7b LLM and ViT-L/14 (224 x 224)
    - LLaVA-SFT<sup>+</sup>-13b: Vicuna-V1.5-13b LLM and ViT-L/14 (336 x 336)
### Stage-2:  Human Preference Collection & Preference Modeling
We collect 10k human preferences where human annotators are asked to compare two responses and pinpoint the more hallucinated one.
### Stage-3: 事实增强的RLHF

#### RLHF中的4个模型
- **Reward Model**: 结构和LLaVA相同，但是使用embedding输出的last token被投影到1个用来指示整个response好坏的分数；在PPO训练中**<font color=red >不更新梯度</font>**

- **Policy Model**：也称为Actor Model, 也是RLHF最终生成的模型，在PPO训练过程中更新梯度

- **Value Model**: 也称为Critic Model, 用来预测respose的好坏，在训练过程中实时调整模型，选择对未来累积收益最大的行为。该Paper根据 **AlpacaFarm**，Value Model使用Reward Model初始化。当使用LLaVA-13B-based的Reward Model训练LLaVA-7B-based的Policy Model的时候，使用的Value Model也是13B的。

- **Reference Model**：在本文中叫做：Old Policy Model，提供一个SFT模型的备份，帮助模型不会出现极端的变化，在PPO训练中**<font color=red >不更新梯度</font>**


本文中的Reward Model，也称为Preference Model, 被训练成一个对于“better response”给予higher score。使用的是成对的比较的数据，可以表示为：$\mathcal{D}_{RM} = \{(\mathcal{I}, x, y_0, y_1, i)\}$, 其中$\mathcal{I}$表示图像，$x$表示prompt，$y_0$和$y_1$表示2个相关的response，$i$表示偏好response的index。 Reward Model 使用交叉熵损失：
$$\mathcal{L}(r_\mathbf{\theta}) = -\mathbf{E}_{(\mathcal{I}, x, y_0, y_1, i)~\mathcal\sim{RM}}[log\sigma(r_{\mathbf{\theta}}(\mathcal{I},x,y_{i})-r_{\mathbf{\theta}}(\mathcal{I},x,y_{i-1})]$$

在RLHF（PPO）过程中，对4个模型（Policy, Reward, Value, Old Policy Model)在10k人类偏好数据上使用LoRA（*r*=64）进行fine-tuning；PPO的实现和**Alpacafarm**相同；

# LLaVA-1.5
Paper: [Improved Baselines with Visual Instruction Tuning](https://arxiv.org/abs/2310.03744)

发表于2023.10, 模型结构和LLaVA相同，细节上：
- Vision-Encoder使用CLIP VIT-L/336px，LLM使用Vicuna v1.5（13B）
- Linear Matrix换成了2层MLP，相比于LLaVA，可以增强多模态能力
- 增加了学术任务导向的VQA数据

当需要简介的回答的时候，可以在prompt的末尾添加：Answer the question using a single word or phrase

## LLaVA-HD
![LLaVA-1.5-HD](LLaVA-1.5-HD.png)
AnyRes策略：使用CLIP VIT-L/14(224px)作为Visual Encoder，将图像split成$224^2$的grids，支持6个grids（1x1, 1x2, 1x3, 1x4, 1x5, 1x6,
2x2, 2x3, 以及他们的转置)；对这些patches分别独立的提取特征，然后merge成1个大的feature map，再**后处理**成flatten features；此外还和原始图像缩放到224x224讲过Visual Encoder得到的特征concat起来作为LLM的输入；

### 后处理
- Padding Reoval: 对应到图像padding的位置的特征被丢弃掉，用来降低特征维度，提升效率；
- Row-end Token：在每一行特征的后面增加一个特殊Token，用来显式的提供图像shape的indication。
- Flatten：最后flatten image feature，和文本token一起送进LLM

### 训练
不需要额外的pretraining，直接使用instruct tuning，即先对齐，然后sft。

# LLaVA-Interactive
Paper: [LLaVA-Interactive: An All-in-One Demo for Image Chat, Segmentation, Generation and Editing](https://arxiv.org/abs/2311.00571)

Code: <https://llava-vl.github.io/llava-interactive>

发表于2023.11，具有多模态人机交互能力：支持视觉输入，视觉输出，视觉交互。结合了三个模型的优势：LLaVA的视觉对话、[SEEM](https://github.com/UX-Decoder/Segment-Everything-Everywhere-All-At-Once)的视觉prompt分割和[GLIGEN](https://gligen.github.io/)的视觉生成/编辑。

# LLaVA-Plus
Paper: [LLaVA-Plus: Learning to Use Tools for Creating Multimodal Agents](https://arxiv.org/abs/2311.05437)

Code: <https://llava-vl.github.io/llava-plus/>

发表于2023.11

LLaVA-Plus Capabilities enabled by **P**lug and **L**earning to **U**se **S**kills, Plus是这4份单词的缩写；

# LLaVA‐NeXT
发表于2024.01.30的Blog: *LLaVA‐NeXT: Improved reasoning, OCR, and world knowledge* <https://llava-vl.github.io/blog/2024-01-30-llava-next/>，相比于LLaVA-1.5，有以下提升：

- 增加image分辨率，支持3个aspect ratio：，672x672, 336x1344, 1344x336
- 更强的视觉推理和OCR能力
- 在更多场景上更强的视觉对话能力
- 使用SGLang的高效的部署和推理

Zero-Shot Chinese Capaticy：**涌现**能力，只在英文多模态数据上训练，在中文多模态场景表现也不错

## 详细的技术提升
### 1. 动态高分辨率
详见LLaVA-1.5-HD
### 2. 数据混合
- 高质量的用户交互数据：首先，task instructions的丰富度，其次：数据的准确性；使用了2个数据源：（1）Existing GPT-V data. LAION-GPT-V and ShareGPT-4V，（2）15k的来自LLaVA demo的覆盖各个场景的visual instruction tuning数据
- 多模态文档/表格数据：（1）去掉了TextCaps数据因为TextCaps使用了TextVQA的训练图像；将TextCaps替换成DocVQA和SynDog-EN （2）受到Qwen-VL-7B-Chat的启发，为了更多的图标理解，增加了ChartQA、 DVQA和AI2D
### 3. Scaling LLM Backbone
除了Vicuna 1.5(7B和13B)之外，也使用 [Mistral-7B](https://mistral.ai/news/announcing-mistral-7b) 和 [Nous-Hermes-2-Yi-34B](https://huggingface.co/NousResearch/Nous-Hermes-2-Yi-34B)

Model Card:

**注意：第2阶段对整个模型都进行了训练**
<table>
    <tr>
        <th colspan="2">Name</th>
        <th>LLaVA-1.6-7B</th>
        <th>LLaVA-1.6-13B</th>
        <th>LLaVA-1.6-34B</th>
    </tr>
    <tr>
        <th rowspan="4">Model Size</th>
        <td>Total</td>
        <td><b>7.06B</b></td>
        <td><b>13.35B</b></td>
        <td><b>34.75B</b></td>
    </tr>
    <tr>
        <td>Vision Encoder</td>
        <td>303.5M</td>
        <td>303.5M</td>
        <td>303.5M</td>
    </tr>
    <tr>
        <td>Connector</td>
        <td>21M</td>
        <td>31.5M</td>
        <td>58.7M</td>
    </tr>
    <tr>
        <td>LLM</td>
        <td>6.74B</td>
        <td>13B</td>
        <td>34.39B</td>
    </tr>
    <tr>
        <th colspan="2">Resolution</th>
        <td colspan="3">336 x [(2,2), (1,2), (2,1), (1,3), (3,1), (1,4), (4,1)]</td>
    </tr>
    <tr>
        <th>Stage-1</th>
        <th>Training Data</th>
        <td colspan="3">558K</td>
    </tr>
    <tr>
        <th></th>
        <th>Trainable Module</th>
        <td colspan="3">Connector</td>
    </tr>
    <tr>
        <th>Stage-2</th>
        <th>Training Data</th>
        <td colspan="3">760K</td>
    </tr>
    <tr>
        <th></th>
        <th>Trainable Module</th>
        <td colspan="3">Full model</td>
    </tr>
    <tr>
        <th colspan="2">Compute (#GPU x #Hours)</th>
        <td>8x20</td>
        <td>16x24</td>
        <td>32x30</td>
    </tr>
    <tr>
        <th colspan="2">Training Data (#Samples)</th>
        <td colspan="3">1318K</td>
    </tr>
</table>

# LLaVA-NeXT Video
发表于2024.04.30的Blog [LLaVA-NeXT: A Strong Zero-shot Video Understanding Model](https://llava-vl.github.io/blog/2024-04-30-llava-next-video/)

几点提升：

1. **Zero-shot video representation capabilities with AnyRes**：只在图像数据上训练就可以在video任务上表现表现很好，第一个**strong zero-shot modality transfer ability**的MLLM
2. **Inference with length generalization improves on longer videos**：The linear scaling technique enables length generalization, allowing LLaVA-NeXT to effectively handle long-video beyond the limitation of the "max_token_length" of the LLM.
3. **Strong video understanding ability**：
(1) LLaVA-Next-Image, which combines the above two techniques, yields superior zero-shot performance than open-source LMMs tuned on videos. (2) LLaVA-Next-Video, further supervised fine-tuning (SFT) LLaVA-Next-Image on video data, achieves better video understanding capabilities compared to LLaVA-Next-Image. (3) LLaVA-Next-Video-DPO, which aligns the model response with AI feedback using direct preference optimization (DPO), showing significant performance boost.
4. 结合SGLang的高效部署和推理

## 技术亮点
### 1. AnyRes: From Multi-patch to Multi-frame
![LLaVA-NeXT-AnyRes](LLaVA-NeXT-AnyRes.png)
多个patch可以很容易的改成多帧，假设每帧可以转换成24x24个token，对于max_token_length限制是4096的LLM，那么就需要确保24x24xN + the number of text tokens < 4096。也可以使用2x2的pooling将24x24的feature转变成12x12的，经过实验发现12x12的token表现更好；最多使用16帧。
### 2. Length generalization: From multi-frame to long-video.
在旋转位置编码（RoPE: Rotary Position Embeddings)中使用**linear scaling**，可以在推理的时使用比训练时更长的token。

通过使用该技术，使用scaling factor = 2,就可以将模型的**max_token_length**提升到8096，对于每帧视频表示成12x12个token,能够处理的视频帧数达到56帧，显著的提升了对于长视频的处理能力。

### 3. DPO from AI Feedback
直接使用 [LLaVA-Hound](https://arxiv.org/pdf/2404.01258)中的方法，进行DPO，可以得到更强的模型

## 经验
### 1. 如何表示视频
使用12x12个token表示视频帧最好
### 2. how to fine-tune on videos?
使用image和video混合的数据训练，每个batch都有image和video

# LLaVA-NeXT Stronger LLMs
发表于2024.05.10的Blog: [LLaVA-NeXT: Stronger LLMs Supercharge Multimodal Capabilities in the Wild](https://llava-vl.github.io/blog/2024-05-10-llava-next-stronger-llms/)

亮点：
1. 更大的LLM会使模型更强，本文使用LLaMA3(8B)，Qwen-1.5(72B和110B)
2. 提出了1个LLaVA-Bench (Wilder): 用作Daily-life Visual Chat Benchmarks

# LLaVA-NeXT-Ablations
发表于2024.05.25的Blog：[LLaVA-NeXT: What Else Influences Visual Instruction Tuning Beyond Data?
](https://llava-vl.github.io/blog/2024-05-25-llava-next-ablations/)

## Insights on Architectures
### Language Models
1. 更大的LLM，会使MLLM能力更强
2. 更大的LLM，收敛更快且loss更小
3. 学习率调整
    
    训练过程中出现尖刺，往往预示着更糟糕的模型性能，即使损失和没有尖刺的时候一样小

    使用更小的学习率可以缓解该问题，如(LLM, Vision)的学习率设置为：(2e-5, 2e-6), (2e-5, 1e-6), (1e-5, 2e-6), (1e-5, 1e-6), (5e-6, 1e-6), 和 (5e-6, 5e-7)

    Vision Encoder的lr应当比LLM的小5x或者10x
### Vision Encoders
分辨率、token数目和预训练的数据比模型大小更重要，如基于代价和性能的平衡，使用了WEBLI-10B数据（384x384)，输出729个token的SO400M表现出了最好的性能
## Visual Representations
Higher-AnyRes
![llava-next-ablations-higher-any-res](llava-next-ablations-higher-any-res.png)
阈值化的双线性插值【Thresholded Bilinear Interpolation】，对于一个grids（宽为a，高为b）的AnyRes，假设每个grid的token数时T，那么最多能够表示的Token个数是 $L = (a \times b + 1) \times T$；设置一个阈值$\tau$，使用双线性插值来减少每个grid的token数目
$$ T_{new}=\left\{
\begin{array}{rcl}
\tau/(a\times b+1)       &      & if \quad{L>\tau}\\
T     &      & if \quad {L\leq \tau}\\
\end{array} \right. $$
## Traning Strategies
在原本的2阶段训练的图文对齐和视觉指令微调中加入**Stage-1.5: 进行高质量的知识学习**：训练方法和Stage-2类似，数据质量要高（一般可以使用ReCaped Detailed Description Data，文档/OCR Data）
![llava-next-ablations-training-strategies](llava-next-ablations-training-strategies.png)

# LLaVA-NeXT-Interleave
发表于2024.06.16的Blog: [LLaVA-NeXT: Tackling Multi-image, Video, and 3D in Large Multimodal Models](https://llava-vl.github.io/blog/2024-06-16-llava-next-interleave/)

图文交错格式（image-text interleave format）适合作为一般的数据模版统一不同的场景，如单图或者多图作为特殊case，视频（视作多帧），3D数据（视作multi view),因此开发出了LLaVA-NeXT-Interleave：将模型能力拓展到真实世界的场景：多图，多帧，多视图（3D），并保持多Patches场景的性能，称之为M4。


## 涌现能力
1. 单图和多图之间的Task Transfer
2. 单图和视频之间的Task Transfer
3. 实际应用场景：

    对多张图进行总结和信息检索

    识别多个艺术家的绘画风格，讲述他们的不同点

    为图像生成任务提供图像编辑提prompt

    总结多个文档的信息，并提供比较
## Interleave Visual Instruction Tuning
构建了M4-Instruct作为训练数据，构建了LLaVA-Interleave Bench
## 训练技术
模型结构：LLaVA-NeXT-Interleave (0.5/7/14B) adopts **Qwen 1.5-0.5B, 7B and -14B** as base LLMs, **SigLIP-400M** with **384x384** resolutions as the vision encoder, and a **two-layer MLP** as the projection layer.

1. 以LLaVA-NeXT-Image（经过stage-1和stage-2训练）作为base model， 在M4-Intruct数据上继续训练更好，能够继承单图指令跟随能力，并将其拓展到多图，视频和3D数据上。
2. image 放在哪里更好？
    - 所有的images token放在text前面，在用到的时候，使用1个special token进行引用
    - 将image放在原始的交错数据中的位置
    
    实验发现将2种方法混合在一起训练，效果更好
3. 使用pooling降低image feature map的大小，使得能够处理更多的视频帧，会降低性能；

# LLaVA-OneVision

发表于2024.08.06 Paper: [LLaVA-OneVision: Easy Visual Task Transfer](https://arxiv.org/abs/2408.03326)

Blog: [LLaVA-OneVision Easy Visual Task Transfer](https://llava-vl.github.io/blog/2024-08-05-llava-onevision)

![llava-onevision-roadmap](llava-onevision-roadmap.png)

综合了上述4个blog的内容，Vision Encoder使用SigLIP, LLM使用Qwen-2(0.5B, 7B, 72B)

# LLaVA-Video
发表于2024.10.03 Paper: [Video Instruction Tuning With Synthetic Data](https://arxiv.org/abs/2410.02713)

Blog: [Video Instruction Tuning with Synthetic Data](https://llava-vl.github.io/blog/2024-09-30-llava-video/)

构建了一个高质量合成的Video-Language Instruction-Following Data: LLaVA-Video-178K

视频表示：借鉴Slow-Fast的处理方法，得到LLaVA-Video-7B和LLaVA-Video-72B

# LLaVA-Critic
发表于2024.10.3 Paper: [LLaVA-Critic: Learning to Evaluate Multimodal Models](https://arxiv.org/abs/2410.02712)

Blog: [LLaVA-Critic: Learning to Evaluate Multimodal Models](https://llava-vl.github.io/blog/2024-10-03-llava-critic/)

第1个被设计为用于评估大范围的多模态任务的generalist evaluator的开源LMM

## Critic Instruction-Following Dataset
收集了一个评价指令跟随数据集：**LLaVA-Critic-113k**：既可以对单个response进行打分，也可以比较2个resposes哪个更好。同时，也要给出打分或者比较的原因；

## 用途
1. LLaVA-Critic可以做LMM的Judge：用于评估2个LMMs产生的response哪个更好，并给出原因；
2. 偏好学习：可以集成到任意的偏好学习算法种，如RLHF和DPO中。
    
    通过对LLaVA-OV-7B和LLaVA-OV-72B，使用LLaVA-Critic做DPO学习，得到LLaVA-OV-Chat，比原模型效果有较大的提升；
