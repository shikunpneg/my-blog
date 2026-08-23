---
title: LoRA：低秩自适应大语言模型（论文翻译）
date: 2026-07-09
tags:
  - 人工智能
  - 论文翻译
description: Edward Hu et al. 2021 LoRA 论文全文翻译（保留 LaTeX 公式）
---

> AI 参与声明：本文档基于 arxiv 2106.09685v2 翻译，使用本地 qwen2.5:7b 模型生成，公式保留 LaTeX 源码。



---

## 原文第 1-1 页

**LORA：大规模语言模型的低秩自适应**

*Edward Hu\*  ·  Yelong Shen\*  ·  Phillip Wallis  ·  Zeyuan Allen-Zhu  ·  Yuanzhi Li  ·  Shean Wang  ·  Lu Wang  ·  Weizhu Chen*

*Microsoft Corporation*

{edwardhu, yeshe, phwallis, zeyuana, yuanzhil, swang, luw, wzchen}@microsoft.com  
yuanzhil@andrew.cmu.edu

*（版本 2）*

## 摘要

自然语言处理的一个重要范式是在通用领域数据上进行大规模预训练，然后针对特定任务或领域进行适应。随着我们预训练更大规模的模型，全量微调变得不再可行，因为这需要重新训练所有模型参数。以GPT-3 175B为例——部署独立的微调实例，每个实例包含175B个参数，成本极其高昂。我们提出了一种低秩自适应方法（Low-Rank Adaptation, LoRA），该方法冻结预训练模型的权重，并在Transformer架构中的每一层注入可训练的秩分解矩阵，极大地减少了下游任务中可训练参数的数量。与使用Adam微调GPT-3 175B相比，LoRA可以将可训练参数数量减少10,000倍，同时减少GPU内存需求3倍。尽管LoRA具有更少的可训练参数、更高的训练吞吐量，并且没有额外的推理延迟，但在RoBERTa、DeBERTa、GPT-2和GPT-3上的模型质量表现与微调相当或更好。我们还对语言模型适应中的秩不足问题进行了实证研究，这为LoRA的有效性提供了见解。我们提供了一个便于将LoRA集成到PyTorch模型的软件包，并在 [https://github.com/microsoft/LoRA](https://github.com/microsoft/LoRA) 上发布了我们的实现和模型检查点。

## 1 引言

预训练权重 \( W_0 \in \mathbb{R}^{d \times d} \)，通过低秩分解 \( \Delta W = BA \) 注入可训练更新：

- \( A \in \mathbb{R}^{r \times k} \)：随机高斯初始化 \( \mathcal{N}(0, \sigma^2) \)
- \( B \in \mathbb{R}^{d \times r} \)：初始化为零
- 训练时仅更新 \( A \) 和 \( B \)，\( W_0 \) 保持冻结
- 输出按坐标相加：\( h = W_0 x + BA x \)

> **图 1：我们的重参数化。我们仅训练 A 和 B。**（详见 LoRA 论文原图 1）

许多自然语言处理的应用依赖于将一个大规模的预训练语言模型适应到多个下游应用中。这种适应通常通过微调来实现，即更新预训练模型的所有参数。微调的主要缺点是新模型包含与原始模型相同数量的参数。随着每几个月就训练出更大规模的模型，这从GPT-2（Radford et al., b）或RoBERTa large（Liu et al., 2019）的“不便”变成了对GPT-3（Brown et al., 2020）这一拥有1750亿可训练参数的大规模模型的关键部署挑战。

许多人试图通过仅适应部分参数或为新任务学习外部模块来缓解这个问题。这样，我们只需要在每个任务中存储和加载少量的任务特定参数，而无需额外的预训练模型，从而大大提升了部署时的操作效率。然而，现有的技术
∗ 共同贡献。
0 与V1相比，此草稿包括更好的基线、GLUE上的实验以及更多关于适配器延迟的内容。
1 尽管GPT-3 175B通过少样本学习取得了显著的性能，但微调极大地提升了其性能（见附录A）。

[1] Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). Language models are unsupervised multitask learners. OpenAI blog.
[2] Liu, Y., Ott, M., Goyal, N., Naik, M., Lewis, M. W., Zettlemoyer, L., & Stumpf, R. (2019). RoBERTa: A robustly optimized bert pretraining approach. arXiv preprint arXiv:1907.11692.
[3] Brown, T. B., Mann, V., Ryder, N., Subbiah, M., Kaplan, J., Dhariwal, P., ... & Neelakantan, A. (2020). Language models are few-shot learners. arXiv preprint arXiv:2005.14165.
[4] Appendix A: Fine-tuning GPT-3 175B significantly boosts its performance in few-shot learning scenarios.

---

## 原文第 2-2 页

often 引入推断延迟（Houlsby et al., 2019；Rebuffi et al., 2017）通过增加模型深度或减少模型可使用的序列长度（Li & Liang, 2021；Lester et al., 2021；Hambardzumyan et al., 2020；Liu et al., 2021）（第3节）。更为重要的是，这些方法往往无法与微调基准相匹配，从而在效率和模型质量之间形成权衡。
我们受到Li et al. (2018a) 和 Aghajanyan et al. (2020) 的启发，他们表明学习到的过参数化模型实际上驻留在一个低固有维度上。我们假设模型适应过程中权重的变化也具有低“固有秩”，从而提出了低秩自适应（Low-Rank Adaptation, LoRA）方法。LoRA 允许我们在训练神经网络中的某些密集层时间接优化这些层在适应过程中的变化的秩分解矩阵，同时冻结预训练权重，如图1所示。以GPT-3 175B为例，我们展示了即使全秩（即d）高达12,288时，一个非常低的秩（即r，在图1中为一或二）就足够了，这使得LoRA 在存储和计算效率方面都表现出色。
LoRA 具有几个关键优势：
- 一个预训练模型可以被共享并用于构建多个小型LoRA模块以适应不同的任务。我们可以冻结共享的模型，并通过替换图1中的矩阵A和B来高效地切换任务，从而显著减少存储需求和任务切换开销。
- LoRA 使得训练更加高效，并且在使用自适应优化器时将硬件门槛降低多达3倍，因为我们不需要为大多数参数计算梯度或维护优化状态。相反，我们只需优化注入的、规模更小的低秩矩阵。
- 我们简单的线性设计允许我们在部署时将可训练矩阵与冻结权重合并，从而在构造上不引入推断延迟，这与完全微调模型相比并无不同。
- LoRA 与许多先前的方法正交，并可以与它们结合使用，例如前缀微调。我们将在附录E中提供一个示例。
术语和约定 我们频繁引用Transformer架构并采用其维度的常规术语。我们将Transformer层的输入和输出尺寸称为dmodel。我们用Wq、Wk、Wv和Wo来指代自注意力模块中的查询/键/值/输出投影矩阵。W或W0表示预训练权重矩阵，∆W表示适应过程中累积的梯度更新。我们使用r来表示LoRA模块的秩。我们遵循Vaswani et al. (2017)、Brown et al. (2020) 的约定，并使用Adam（Loshchilov & Hutter, 2019；Kingma & Ba, 2017）进行模型优化，采用Transformer MLP前馈维度dffn = 4×dmodel。
2 问题陈述
尽管我们的提议对训练目标是通用的，但我们专注于语言建模作为激励用例。以下是对语言建模问题的简要描述以及特定于任务提示的最大化条件概率。
假设我们有一个由Φ参数化的预训练自回归语言模型PΦ(y|x)。例如，PΦ(y|x)可以是一个基于Transformer架构（Vaswani et al., 2017）的多任务学习者，如GPT (Radford et al., b；Brown et al., 2020)。考虑将这个预训练模型适应到下游条件文本生成任务，例如摘要、机器阅读理解(MRC)和自然语言到SQL(NL2SQL)。每个下游任务由一组上下文-目标对的训练数据集表示：Z = {(xi, yi)}i=1,...,N ，其中xi和yi都是标记序列。例如，在NL2SQL中，xi是一个自然语言查询，而yi是其对应的SQL命令；对于摘要而言，xi是一篇文章的内容，而yi是其摘要。

---

## 原文第 3-3 页

在全量微调过程中，模型从预训练权重 $\Phi_0$ 初始化，并通过反复沿梯度方向最大化条件语言建模目标进行更新至 $\Phi_0 + \Delta\Phi$：
$$
\max_{\Phi} \sum_{(x,y) \in Z} |y| \sum_{t=1}^{|y|} \log (P_\Phi(y_t|x, y < t)) \quad (1)
$$

全量微调的主要缺点之一是，对于每个下游任务，我们都需要学习一组不同的参数 $\Delta\Phi$，其维度 $|\Delta\Phi|$ 等于 $|\Phi_0|$。因此，如果预训练模型规模较大（例如 GPT-3 有 $|\Phi_0| \approx 175$ 亿），存储和部署许多独立的微调实例可能会非常具有挑战性，甚至不可行。

在本文中，我们采用了一种更为参数高效的策略，在这种策略下，任务特定的参数增量 $\Delta\Phi = \Delta\Phi(\Theta)$ 进一步编码为一个较小规模的参数集 $\Theta$，其维度 $|\Theta| \ll |\Phi_0|$。因此，寻找 $\Delta\Phi$ 的问题转化为优化 $\Theta$：
$$
\max_{\Theta} \sum_{(x,y) \in Z} |y| \sum_{t=1}^{|y|} \log \left( p_{\Phi_0 + \Delta\Phi(\Theta)}(y_t|x, y < t) \right) \quad (2)
$$

在后续章节中，我们将提出使用低秩表示来编码 $\Delta\Phi$，从而实现计算和内存的高效性。当预训练模型为 GPT-3 175B 时，可训练参数 $|\Theta|$ 可以小至 $|\Phi_0|$ 的 0.01%。

### 是否现有的解决方案足够好？
我们所要解决的问题并非新问题。自迁移学习诞生以来，已有众多研究致力于使模型适应更加参数和计算高效。详见第6节对一些知名工作的综述。以语言建模为例，高效的适应策略主要有两种：添加适配器层（Houlsby et al., 2019；Rebuffi et al., 2017；Pfeiffer et al., 2021；Rücklé et al., 2020）或优化输入层激活值的某些形式（Li & Liang, 2021；Lester et al., 2021；Hambardzumyan et al., 2020；Liu et al., 2021）。然而，这两种策略都有其局限性，尤其是在大规模和延迟敏感的生产场景中。

### 适配器层引入推理延迟
适配器层有许多变体。我们主要关注 Houlsby 等（2019）最初的设计，该设计每个 Transformer 块有两个适配器层；以及 Lin 等（2020）的较新版本，每个块只有一个适配器层但附加了一个 LayerNorm（Ba et al., 2016）。虽然可以通过剪枝层或利用多任务设置来减少整体延迟（Rücklé et al., 2020；Pfeiffer et al., 2021），但没有直接方法可以绕过适配器层的额外计算。这似乎不是一个大问题，因为适配器层设计时参数较少（有时 <1% 的原始模型），通过一个小瓶颈维度限制了它们能增加的 FLOPs。然而，在大规模神经网络中，依赖硬件并行来保持低延迟，而适配器层必须按顺序处理。这意味着在在线推理场景中，批大小通常仅为一个时，使用适配器会导致明显的延迟增加（表 1）。

当需要像 Shoeybi 等（2020）、Lepikhin 等（2020）那样拆分模型时，问题会变得更糟。因为额外的深度要求更多的同步 GPU 操作，如 AllReduce 和 Broadcast，除非我们冗余存储适配器参数多次。

### 直接优化提示词困难
另一种方法，例如前缀调优（Li & Liang, 2021），面临不同的挑战。我们观察到前缀调优难以优化，并且其性能在可训练参数中非单调变化，这与原始论文中的类似观察一致。更根本的是，为适应保留部分序列长度必然会减少用于处理下游任务的可用序列长度，我们认为这使得调整提示词的效果不如其他方法。我们将任务性能的研究推迟到第5节。

---

## 原文第 4-4 页

### 4 我们的方法

我们描述了低秩自适应（Low-Rank Adaptation, LoRA）的简单设计及其实际优势。本文中概述的原则适用于任何深度学习模型中的密集层，尽管我们在实验中仅专注于特定权重在转换器语言模型中的应用作为动机案例。

#### 4.1 低秩参数化更新矩阵

神经网络包含许多执行矩阵乘法的密集层。这些层中的权重矩阵通常具有满秩。当适应特定任务时，Aghajanyan等人（2020）表明预训练的语言模型具有较低的“固有维度”，即使在随机投影到较小子空间后仍能高效学习。受此启发，我们假设在适应过程中权重更新也具有较低的“固有秩”。对于一个预先训练好的权重矩阵 \(W_0 \in \mathbb{R}^{d \times k}\)，我们通过低秩分解约束其更新，表示为 \(W_0 + \Delta W = W_0 + BA\)，其中 \(B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times k}\) 且秩 \(r \ll \min(d, k)\)。在训练过程中，\(W_0\) 被冻结且不接收梯度更新，而 \(A\) 和 \(B\) 包含可训练参数。注意，\(W_0\) 和 \(\Delta W = BA\) 都与相同的输入相乘，并且它们各自的输出向量按坐标相加。对于 \(h = W_0 x\)，我们的修改前向传播过程为：
\[ h = W_0 x + \Delta W x = W_0 x + B A x \tag{3} \]
我们用随机高斯初始化表示 \(A\) 并将 \(B\) 初始化为零，因此在训练开始时 \(\Delta W = BA = 0\)。然后我们将 \(\Delta W x\) 按比例缩放为 \(\frac{\alpha}{r}\)，其中 \(\alpha\) 是一个常数。使用 Adam 优化器进行调整时，调节 \(\alpha\) 大致相当于适当缩放初始化后的学习率调整。因此，我们简单地将 \(\alpha\) 设定为我们尝试的第一个值 \(r\) 并不对其进行调优。这种比例有助于减少在改变 \(r\) 时重新调整超参数的需求（Yang & Hu, 2021）。

#### 全量微调的一种推广

更一般的微调形式允许训练预训练参数的一个子集。LoRA 进一步发展，适应过程中不需要累积的梯度更新对权重矩阵具有满秩。这意味着当我们将 LoRA 应用于所有权重矩阵并训练所有偏置时，通过将 LoRA 的秩 \(r\) 设定为预先训练好的权重矩阵的秩，我们大致恢复了全量微调的表达能力。换句话说，随着可训练参数数量增加，LoRA 的训练过程逐渐收敛于原始模型的训练，而基于适配器的方法则收敛于一个 MLP 模型，基于前缀的方法则收敛于无法处理长输入序列的模型。

#### 无额外推理延迟

在生产部署中，我们可以显式计算并存储 \(W = W_0 + BA\) 并进行常规推理。注意，\(W_0\) 和 \(BA\) 均为 \(\mathbb{R}^{d \times k}\) 矩阵。当需要切换到另一个下游任务时，我们可以通过减去 \(BA\) 再加上不同的 \(B'A'\) 来恢复 \(W_0\)，这是一个快速操作且内存开销很小的操作。关键在于，与权重相比，它们代表的参数数量可以忽略不计。[^2]

[^2]: 与预训练权重相比，LoRA 引入的参数量可忽略不计。

---

## 原文第 5-5 页

保证在推理过程中，LoRA 的引入不会增加额外的延迟，与微调模型相比，其设计初衷即为如此。
### 4.2 将 LoRA 应用于 Transformer

原则上，我们可以通过减少神经网络中权重矩阵的数量来应用 LoRA 来降低可训练参数的数量。在 Transformer 架构中，自注意力模块包含四个权重矩阵（Wq, Wk, Wv, Wo），而多层感知机 (MLP) 模块则包含两个。我们将 Wq（或 Wk、Wv）视为一个维度为 dmodel×dmodel 的单一矩阵，尽管输出维度通常被分割成注意力头。我们仅对下游任务进行自注意力权重的适应，并冻结 MLP 模块（因此在下游任务中不会对其进行训练），以简化模型并提高参数效率。我们在第 7.1 节中进一步研究了不同类型的自注意力权重矩阵在 Transformer 中的效果。对于 MLP 层、LayerNorm 层和偏置项的适应，我们留待未来的工作进行实证调查。

### 实用优势与局限性

LoRA 最显著的优势在于减少了内存和存储使用量。对于通过 Adam 训练的大规模 Transformer 模型，如果 r≪ dmodel，则可以将 VRAM 使用量减少至 2/3，因为我们无需为冻结参数存储优化器状态。在 GPT-3 175B 上，训练期间的 VRAM 消耗从 1.2TB 减少到 350GB。当 r = 4 且仅适应查询和值投影矩阵时，检查点大小减少了约 10,000 倍（从 350GB 减少到 35MB）[4]。这使得我们可以使用更少的 GPU 进行训练，并避免 I/O 瓶颈。另一个优势在于，在部署过程中通过仅切换 LoRA 权重而无需更换所有参数，可以以更低的成本在不同任务之间进行切换。这允许创建许多可随时替换的定制模型，这些模型可以在存储预训练权重于 VRAM 的机器上即时加载和卸载。我们还观察到，在 GPT-3 175B 上训练时，与全微调相比，LoRA 可以提高约 25% 的速度[5]，因为我们无需为绝大多数参数计算梯度。

### LoRA 的局限性

例如，如果选择将 A 和 B 吸收进 W 中以消除额外的推理延迟，则在单次前向传递中对不同任务的不同输入进行批量处理并不直观。尽管如此，在延迟不关键的情况下，仍有可能不合并权重并动态选择用于批次样本的 LoRA 模块。

### 5 实验

我们评估了 LoRA 在 RoBERTa (Liu et al., 2019)、DeBERTa (He et al., 2021) 和 GPT-2 (Radford et al., b) 上的下游任务性能，之后扩展到 GPT-3 175B (Brown et al., 2020)。我们的实验涵盖了从自然语言理解（NLU）到生成（NLG）的各种任务。具体而言，在 RoBERTa 和 DeBERTa 上我们使用 GLUE (Wang et al., 2019) 基准进行评估；在 GPT-2 上遵循 Li & Liang (2021) 的设置以直接比较，并添加了 WikiSQL (Zhong et al., 2017)（自然语言到 SQL 查询）和 SAMSum (Gliwa et al., 2019)（对话摘要）进行大规模实验。更多数据集详情见附录 C。我们使用 NVIDIA Tesla V100 进行所有实验。

### 5.1 基线

为了与其它基线进行广泛比较，我们复制了先前工作的设置并在可能的情况下重用了其报告的数字。然而，这意味着某些基线可能仅出现在特定的实验中。
全微调（FT）是一种常见的适应方法。在全微调过程中，模型初始化为预训练权重和偏置，并且所有模型参数都会经历梯度更新。一种简单的变体是仅更新某些层而冻结其他层。我们包括了 Li & Liang (2021) 在 GPT-2 上报告的一个基线，该基线仅适应最后两层（FTTop2）。

[4] 我们在部署时仍需要 350GB 的模型；然而，存储 100 个已适配的模型只需约 350GB + 35MB * 100≈ 354GB，而无需 100 * 350GB≈ 35TB。
[5] 对于 GPT-3 175B，在全微调时每 V100 GPU 的训练吞吐量为 32.5 tokens/s；通过相同数量的权重切片进行模型并行，LoRA 在每 V100 GPU 上的吞吐量提高到 43.1 tokens/s。

---

## 原文第 6-6 页

| Model & Method | # Trainable Parameters | MNLI | SST-2 | MRPC | CoLA | QNLI | QQP | RTE | STS-B | Avg. |
|----------------|------------------------|------|-------|------|------|------|------|------|-------|------|
| RoBbase (FT)*  | 125.0M                 | 87.6 | 94.8  | 90.2 | 63.6 | 92.8 | 91.9 | 78.7 | 91.2  | 86.4 |
| RoBbase (BitFit)* | 0.1M                 | 84.7 | 93.7  | 92.7 | 62.0 | 91.8 | 84.0 | 81.5 | 90.8  | 85.2 |
| RoBbase (AdptD)* | 0.3M                 | 87.1±.0 | 94.2±.1 | 88.5±1.1 | 60.8±.4 | 93.1±.1 | 90.2±.0 | 71.5±2.7 | 89.7±.3 | 84.4 |
| RoBbase (AdptD)* | 0.9M                 | 87.3±.1 | 94.7±.3 | 88.4±.1 | 62.6±.9 | 93.0±.2 | 90.6±.0 | 75.9±2.2 | 90.3±.1 | 85.4 |
| RoBbase (LoRA) | 0.3M                   | 87.5±.3 | 95.1±.2 | 89.7±.7 | 63.4±1.2 | 93.3±.3 | 90.8±.1 | 86.6±.7 | 91.5±.2 | 87.2 |
| RoBlarge (FT)* | 355.0M                 | 90.2 | 96.4  | 90.9 | 68.0 | 94.7 | 92.2 | 86.6 | 92.4  | 88.9 |
| RoBlarge (LoRA) | 0.8M                  | 90.6±.2 | 96.2±.5 | 90.9±1.2 | 68.2±1.9 | 94.9±.3 | 91.6±.1 | 87.4±2.5 | 92.6±.2 | 89.0 |
| RoBlarge (AdptP)† | 3.0M                | 90.2±.3 | 96.3±.3 | 90.2±.7 | 68.3±1.0 | 94.8±.2 | 91.9±.1 | 83.8±2.9 | 92.1±.7 | 88.4 |
| RoBlarge (AdptP)† | 0.8M                | 90.5±.3 | 96.6±.2 | 89.7±1.2 | 67.8±2.5 | 94.8±.3 | 91.7±.2 | 80.1±2.9 | 91.9±.4 | 87.9 |
| RoBlarge (AdptH)† | 6.0M                | 89.9±.5 | 96.2±.3 | 88.7±2.9 | 66.5±4.4 | 94.7±.2 | 92.1±.1 | 83.4±1.1 | 91.0±1.7 | 87.8 |
| RoBlarge (AdptH)† | 0.8M                | 90.3±.3 | 96.3±.5 | 87.7±1.7 | 66.3±2.0 | 94.7±.2 | 91.5±.1 | 72.9±2.9 | 91.5±.5 | 86.4 |
| RoBlarge (LoRA)† | 0.8M                | 90.6±.2 | 96.2±.5 | 90.2±1.0 | 68.2±1.9 | 94.8±.3 | 91.6±.2 | 85.2±1.1 | 92.3±.5 | 88.6 |
| DeBXXL (FT)*  | 1500.0M                | 91.8 | 97.2  | 92.0 | 72.0 | 96.0 | 92.7 | 93.9 | 92.9  | 91.1 |
| DeBXXL (LoRA) | 4.7M                   | 91.9±.2 | 96.9±.2 | 92.6±.6 | 72.4±1.1 | 96.0±.1 | 92.9±.1 | 94.9±.4 | 93.0±.2 | 91.3 |
Table 2: RoBERTabase, RoBERTalarge, and DeBERTaXXL 在不同适应方法下的表现，于 GLUE 基准测试上。我们报告了 MNLI 的总体准确率（匹配和不匹配），CoLA 的 Matthew’s 相关系数，STS-B 的皮尔逊相关系数，以及其他任务的准确率。所有指标中数值越大越好。* 表示数据来自先前研究工作。† 表示配置与 Houlsby 等人 (2019) 类似以进行公平比较。
基线为仅训练偏差向量而冻结其他部分的 Bias-only 或 BitFit 方法。目前，该方法也由 BitFit（Zaken 等人, 2021）进行了研究。
前缀嵌入调谐 (PreEmbed) 在输入标记之间插入特殊标记。这些特殊标记具有可训练的词向量，并且通常不在模型词汇表中。放置这些标记的位置可能会影响性能。我们关注“前缀”（prefixing），即在提示前添加此类标记，以及“内插”（infixing），即将其附加到提示后；这两种方法均见于 Li & Liang (2021) 中。我们用 lp 表示前缀标记的数量，li 表示内插标记的数量。可训练参数数量为 |Θ| = dmodel × (lp + li)。
前缀层调谐 (PreLayer) 是前缀嵌入调谐的扩展。不仅学习一些特殊标记（等价于嵌入层后的激活）的词向量，还学习每个 Transformer 层后的激活。由前一层计算出的激活直接被可训练的激活替换。可训练参数数量为 |Θ| = L × dmodel × (lp + li)，其中 L 是 Transformer 层数。
Houlsby 等人（2019）提出的适配器调谐在自注意力模块（以及 MLP 模块）和后续残差连接之间插入适配器层。每个适配器层包含两个带偏置的全连接层，并且有一个非线性函数在它们之间。我们称之为原始设计 AdapterH。最近，Lin 等人（2020）提出了一种更高效的适配器层应用方法，在 MLP 模块和 LayerNorm 之后应用适配器层。我们称之为 AdapterL。这与 Pfeiffer 等人（2021）提出的另一种设计非常相似，我们称之为 AdapterP。我们还包含一个称为 AdapterDrop 的基线（Rücklé 等人, 2020），通过丢弃一些适配器层来提高效率。我们在可能的情况下引用先前研究中的数字；它们位于第一列带有星号 (*) 的行中。
在所有情况下，|Θ| = ˆLAdpt × (2 × dmodel × r + r + dmodel) + 2 × ˆLLN × dmodel，其中 ˆLAdpt 是适配器层的数量，ˆLLN 是可训练的 LayerNorm 数量（例如，在 AdapterL 中）。
低秩自适应 (LoRA) 在现有权重矩阵旁并行插入可训练的秩分解矩阵。如第 4.2 节所述，在大多数实验中我们仅将 LoRA 应用于 Wq 和 Wv 以简化操作。可训练参数数量由秩 r 和原始权重形状决定：|Θ| = 2 × ˆLLoRA × dmodel × r，其中 ˆLLoRA 是应用 LoRA 的权重矩阵的数量。

---

## 原文第 7-7 页

| 模型与方法 | 参数量 | BLEU | NIST | MET | ROUGE-L | CIDEr |
|------------|--------|------|------|-----|---------|-------|
| GPT-2 M (FT)*         | 354.92M  | 68.2     | 8.62     | 46.2   | 71.0     | 2.47     |
| GPT-2 M (AdapterL)*   | 0.37M    | 66.3     | 8.41     | 45.0   | 69.8     | 2.40     |
| GPT-2 M (AdapterL)*   | 11.09M   | 68.9     | 8.71     | 46.1   | 71.3     | 2.47     |
| GPT-2 M (AdapterH)    | 11.09M   | 67.3±.6  | 8.50±.07 | 46.0±.2 | 70.7±.2 | 2.44±.01 |
| GPT-2 M (FTTop2)*     | 25.19M   | 68.1     | 8.59     | 46.0   | 70.8     | 2.41     |
| GPT-2 M (PreLayer)*   | 0.35M    | 69.7     | 8.81     | 46.1   | 71.4     | 2.49     |
| GPT-2 M (LoRA)        | 0.35M    | 70.4±.1  | 8.85±.02 | 46.8±.2 | 71.8±.1 | 2.53±.02 |
| GPT-2 L (FT)*         | 774.03M  | 68.5     | 8.78     | 46.0   | 69.9     | 2.45     |
| GPT-2 L (AdapterL)    | 0.88M    | 69.1±.1  | 8.68±.03 | 46.3±.0 | 71.4±.2 | 2.49±.0  |
| GPT-2 L (AdapterL)    | 23.00M   | 68.9±.3  | 8.70±.04 | 46.1±.1 | 71.3±.2 | 2.45±.02 |
| GPT-2 L (PreLayer)*   | 0.77M    | 70.3     | 8.85     | 46.2   | 71.7     | 2.47     |
| GPT-2 L (LoRA)        | 0.77M    | 70.4±.1  | 8.89±.02 | 46.8±.2 | 72.0±.2 | 2.47±.02 |

**表 3：不同适应方法在 E2E 自然语言生成挑战中的 GPT-2 中型（M）和大型（L）。所有指标数值越高越好。LoRA 在可训练参数相当或更少的情况下，优于多个基线模型。实验结果的置信区间已显示。\* 表示这些数字来自先前的研究。**

### 5.2 RoBERTa 基础版 / 大型
RoBERTa（Liu et al., 2019）优化了原始在BERT（Devlin et al., 2019a）中提出的预训练配方，并在不引入更多可训练参数的情况下提升了任务性能。尽管近年来，RoBERTa 在自然语言处理排行榜如GLUE基准上已被更大规模的模型超越（Wang et al., 2019），但其仍因体型较小而保持竞争力且广受欢迎。我们从HuggingFace Transformers库中获取预训练的RoBERTA基础版（125M）和大型版（355M），并在GLUE基准任务上评估不同高效适应方法的表现。我们也根据Houlsby et al. (2019) 和Pfeiffer et al. (2021) 的设置进行了复现。为了确保公平比较，我们在与适配器基线对比时对LoRA的评估做出两项关键调整：首先，我们为所有任务使用相同的批量大小，并将序列长度设为128以匹配适配器基线；其次，我们将模型初始化至MRPC、RTE和STS-B的预训练模型，而非已经适应MNLI的微调基准模型。遵循Houlsby et al. (2019) 更加严格的设置进行的实验标记为†。结果见表2（顶部三节）。详情请参阅D.1节以了解所使用的超参数。

5.3 DEBERTA XXL
DeBERTa（He et al., 2021）是BERT的一个较新的变体，训练规模更大，在GLUE（Wang et al., 2019）和SuperGLUE（Wang et al., 2020）等基准测试中表现出色。我们评估LoRA是否能在GLUE上与完全微调的DeBERTa XXL（1.5B）保持同等性能。结果见表2（底部一节）。详情请参阅D.2节以了解所使用的超参数。

5.4 GPT-2 中型 / 大型
在展示了LoRA在自然语言理解模型上的竞争力后，我们希望回答一个问题：即LoRA是否同样适用于自然语言生成模型，如GPT-2中型和大型（Radford et al., b）。我们将设置尽可能接近Li & Liang (2021) 以进行直接比较。由于篇幅限制，本节仅展示我们在E2E自然语言生成挑战中的结果（见表3）。关于WebNLG（Gardent et al., 2017）和DART（Nan et al., 2020）的结果，请参阅F.1节。我们将在D.3节中列出所使用的超参数。


---

## 原文第 8-8 页

**表 4：模型与方法 # 训练可调 | WikiSQL | MNLI-m | SAMSum**

| 参数量（M） | WikiSQL 准确率 (%) | MNLI-m 验证准确率 (%) | R1/R2/RL |
| --- | --- | --- | --- |
| GPT-3 (FT) 175,255.8M | 73.8 | 89.5 | 52.0/28.0/44.5 |
| GPT-3 (BitFit) 14.2M | 71.3 | 91.0 | 51.3/27.4/43.5 |
| GPT-3 (PreEmbed) 3.2M | 63.1 | 88.6 | 48.3/24.2/40.5 |
| GPT-3 (PreLayer) 20.2M | 70.1 | 89.5 | 50.8/27.3/43.5 |
| GPT-3 (AdapterH) 7.1M | 71.9 | 89.8 | 53.0/28.9/44.8 |
| GPT-3 (AdapterH) 40.1M | 73.2 | 91.5 | 53.2/29.0/45.1 |
| GPT-3 (LoRA) 4.7M | 73.4 | 91.7 | 53.8/29.8/45.9 |
| GPT-3 (LoRA) 37.7M | 74.0 | 91.6 | 53.4/29.2/45.1 |

**表 4：不同适应方法在 GPT-3 175B 上的表现。我们报告了 WikiSQL 的逻辑形式验证准确率、MultiNLI-matched 的验证准确率以及 SAMSum 的 Rouge-1/2/L。LoRA 在所有三个数据集上的表现优于先前的方法，包括全微调。WikiSQL 上的验证准确率波动在 ±0.5% 左右，MNLI-m 上为 ±0.1%，SAMSum 为 ±0.2/±0.2/±0.1（三个指标）。**

### 5.5 扩展至 GPT-3 175B
作为 LoRA 的最终压力测试，我们将其扩展到具有 1750 亿参数的 GPT-3。由于训练成本高昂，我们仅报告了每个任务在随机种子上的典型标准差，而非为每项提供单独的标准差。详情见第 D.4 节中的超参数设置。如表 4 所示，LoRA 在所有三个数据集上均匹配或超过了全微调基线。值得注意的是，并非所有方法都能从更多的可训练参数中单调受益，如图 2 所示。我们观察到，在使用超过 256 个前缀嵌入特殊标记或超过 32 个前缀层特殊标记时，性能会显著下降。这与 Li & Liang (2021) 的类似观察结果一致。尽管对这一现象的全面调查超出了本文范围，但我们怀疑更多的特殊标记会导致输入分布进一步偏离预训练数据分布。此外，在低数据域中不同适应方法的表现见第 F.3 节。

**图 2：GPT-3 175B 在 WikiSQL 和 MNLI-matched 上的验证准确率与可训练参数数量的关系。LoRA 展现了更好的可扩展性和任务性能。详情见第 F.2 节中的数据点说明。**

### 6 相关工作
变换器语言模型。变换器 (Vaswani et al., 2017) 是一种基于序列到序列架构的模型，大量使用自注意力机制。Radford 等人 (a) 将其应用于自回归语言建模，通过堆叠 Transformer 解码器实现。自此之后，基于变换器的语言模型在自然语言处理领域占据主导地位，并在许多任务中达到了最先进的水平。随着 BERT (Devlin et al., 2019b) 和 GPT-2 (Radford et al., b) 的出现，一种新的范式出现了——两者都是大规模的变换器语言模型。

---

## 原文第 9-9 页

大规模预训练语言模型在大量文本上进行训练——与仅在特定任务数据上直接进行微调相比，如果在通用领域数据上先进行预训练，然后在特定任务数据上进行微调，则通常可以获得显著的性能提升。一般而言，训练更大的Transformer会带来更好的性能，并且仍然是一个活跃的研究方向。GPT-3（Brown et al., 2020）是迄今为止最大的单一语言模型，拥有175B个参数。
提示工程与微调。尽管GPT-3 175B只需少量额外的训练示例即可适应其行为，但结果高度依赖于输入提示（Brown et al., 2020）。这需要通过实验艺术地组合和格式化提示以最大化模型在特定任务上的性能，这一过程被称为提示工程或提示黑客。微调是指重新训练一个预先在通用领域数据上进行预训练的模型到特定任务（Devlin et al., 2019b；Radford et al. (a)）。其变体包括仅学习参数子集（Devlin et al., 2019b；Collobert & Weston, 2008），但实践者通常会重新训练所有参数以最大化下游性能。然而，由于GPT-3 175B的庞大模型规模使得常规微调变得困难，这主要是因为其产生的大型检查点以及高昂的硬件门槛，因为它具有与预训练相同的内存占用量。
参数高效适应。许多研究提出了在神经网络中插入适配器层（Houlsby et al., 2019；Rebuffi et al., 2017；Lin et al., 2020）。我们的方法使用类似瓶颈结构，对权重更新施加低秩约束。关键的功能差异在于我们学习的权重可以在推理过程中与主权重合并，从而不会引入任何延迟，而适配器层则并非如此（第3节）。适配器的一个当代扩展是COMPACTER（Mahabadi et al., 2021），它本质上是使用Kronecker积和某些预定义的权重共享方案来参数化适配器层。同样地，将LoRA与基于张量积的方法结合可能有助于提高其参数效率，我们将在未来的工作中进行探讨。最近，许多研究提出了优化输入词嵌入以替代微调的方法（Li & Liang, 2021；Lester et al., 2021；Hambardzumyan et al., 2020；Liu et al., 2021），类似于连续和可微的提示工程泛化。我们在实验部分中包括了与Li & Liang (2021) 的比较。然而，这种方法只能通过在提示中使用更多的特殊标记来扩展，而这些特殊标记会占用用于任务标记的学习位置嵌入时可用的序列长度。
深度学习中的低秩结构。低秩结构在机器学习中非常常见。许多机器学习问题具有一定的固有低秩结构（Li et al., 2016；Cai et al., 2010；Li et al., 2018b；Grasedyck et al., 2013）。此外，众所周知，在许多深度学习任务中，尤其是那些具有高度过参数化神经网络的任务中，经过训练的神经网络将享有低秩性质（Oymak et al., 2019）。一些先前的工作甚至在训练原始神经网络时明确施加了低秩约束（Sainath et al., 2013；Povey et al., 2018；Zhang et al., 2014；Jaderberg et al., 2014；Zhao et al., 2016；Kho-dak et al., 2021；Denil et al., 2014），但据我们所知，这些工作没有考虑对冻结模型进行低秩更新以适应下游任务。在理论文献中，已知神经网络在某些低秩结构的概念类下会优于其他经典学习方法，包括相应的有限宽度神经核（Allen-Zhu et al., 2019；Li & Liang, 2018）（Ghorbani et al., 2020；Allen-Zhu & Li, 2019；Allen-Zhu & Li, 2020a）。另一个理论结果表明，低秩适应对于对抗训练是有用的。总之，我们认为我们提出的低秩更新方法得到了文献的良好支持。
7 理解低秩更新
鉴于LoRA的经验优势，我们希望进一步解释从下游任务中学习到的低秩适应特性。需要注意的是，低秩结构不仅降低了硬件门槛，使我们能够并行运行多个实验，还提供了更好的可解释性，即更新权重与预训练权重的相关性。我们将研究重点放在GPT-3 175B上，在此模型中，我们在不损害任务性能的情况下实现了最大的可训练参数减少（最多10,000倍）。
我们进行了一系列实证研究以回答以下问题：1) 在给定的参数预算约束下，我们应该适应预训练Transformer中的哪些权重矩阵子集？

---

## 原文第 10-10 页

### 7.1 应该对Transformer中的哪些权重矩阵应用LoRA？

在有限的参数预算下，我们应该调整哪类权重以获得最佳下游任务性能？如第4.2节所述，我们仅考虑自注意力模块中的权重。我们在GPT-3 175B上设置了18M（约35MB存储于FP16）的参数预算，若调整一种类型的注意权重，则对应r=8；若调整两种类型，则r=4，总共96层。结果见表5。

| 权重类型         | 等级 r | WikiSQL (±0.5%) | MultiNLI (±0.1%) |
|------------------|--------|------------------|------------------|
| Wq               | 8      | 70.4             | 91.0             |
| Wk               | 8      | 70.0             | 90.8             |
| Wv               | 8      | 73.0             | 91.0             |
| Wo               | 8      | 73.2             | 91.3             |
| Wq, Wk           | 4      | 71.4             | 91.3             |
| Wq, Wv           | 4      | 73.7             | 91.3             |
| Wq, Wk, Wv, Wo   | 2      | 73.7             | 91.7             |

**表 5：在 GPT-3 上应用 LoRA 到不同类型的注意力权重后，在相同可训练参数数量下的验证准确率（参数预算 18M）。同时调整 Wq 和 Wv 的整体性能最佳。我们发现对于给定的数据集，随机种子的标准差是一致的，并报告在第一列。**

值得注意的是，将所有参数放入∆Wq或∆Wk会导致显著降低性能，而同时调整Wq和Wv则获得最佳结果。这表明即使r为4也能捕获足够信息，使得调整更多权重矩阵比使用较大等级单种类型的权重更优。

### 7.2 LoRA的最佳等级r是多少？

我们关注不同等级r对模型性能的影响。我们分别调整{Wq, Wv}、{Wq, Wk, Wv, Wo}和仅Wq进行比较。

| 评估指标       | 权重类型         | r = 1 | r = 2 | r = 4 | r = 8 | r = 64 |
|----------------|------------------|-------|-------|-------|-------|--------|
| WikiSQL (±0.5%) | Wq              | 68.8  | 69.6  | 70.5  | 70.4  | 70.0   |
| WikiSQL (±0.5%) | Wq, Wv          | 73.4  | 73.3  | 73.7  | 73.8  | 73.5   |
| WikiSQL (±0.5%) | Wq, Wk, Wv, Wo  | 74.1  | 73.7  | 74.0  | 74.0  | 73.9   |
| MultiNLI (±0.1%) | Wq             | 90.7  | 90.9  | 91.1  | 90.7  | 90.7   |
| MultiNLI (±0.1%) | Wq, Wv         | 91.3  | 91.4  | 91.3  | 91.6  | 91.4   |
| MultiNLI (±0.1%) | Wq, Wk, Wv, Wo | 91.2  | 91.7  | 91.7  | 91.5  | 91.4   |

**表 6：不同等级 r 下的验证准确率。令人惊讶的是，对于这些数据集而言，即使 r 为 1 也能很好地适应 Wq 和 Wv，而单独训练 Wq 需要较大的 r。我们在第 H.2 节中对 GPT-2 进行了类似实验。**

表6显示，LoRA在非常小的r下已经表现出色（尤其是{Wq, Wv}比仅Wq更好）。这表明更新矩阵∆W可能具有非常小的“固有等级”。为进一步支持这一发现，我们检查了不同选择r和不同随机种子所学习子空间之间的重叠。我们认为增加r并不能覆盖更有意义的子空间，这表明低秩适应矩阵已经足够。

然而，我们不期望一个小的r适用于所有任务或数据集。考虑以下思想实验：如果下游任务使用的语言与预训练模型不同的语言，则重新训练整个模型（类似于LoRA中的r=dmodel）可能会比小r的LoRA表现更好。

---

## 原文第 11-11 页

不同秩下的子空间相似性。给定 $A_{r=8}$ 和 $A_{r=64}$，这两个矩阵分别使用相同的预训练模型学习得到自适应矩阵，且它们的秩分别为 8 和 64。我们对其进行奇异值分解并获得右奇异酉矩阵 $U_{A_{r=8}}$ 和 $U_{A_{r=64}}$。希望回答的问题是：在 $U_{A_{r=8}}$ 中前 i 个奇异向量所张成的子空间中有多少部分被包含在 $U_{A_{r=64}}$ 的前 j 个奇异向量所张成的子空间中？我们通过基于格拉斯曼距离的归一化子空间相似度来测量这一数量（见附录 G 更详细的讨论）：
$$
\varphi(A_{r=8}, A_{r=64}, i, j) = \left\| U_{A_{r=8}, i}^{\top} U_{A_{r=64}, j} \right\|_F / \min(i, j) \in [0, 1] \quad (4)
$$
其中 $U_i^{(A_{r=8})}$ 表示与 $A_{r=8}$ 的前 i 个奇异向量对应的列。$\varphi(\cdot)$ 的范围为 [0, 1]，其中 1 表示子空间完全重叠，0 表示完全分离。见图 3 所示 $\varphi$ 随 $i$ 和 $j$ 变化的图形。由于篇幅限制，我们仅分析了第 48 层（共 96 层），但其他层的结果也相同，详见 H.1 节。

> **图 3 描述**：$A_{r=8}$ 和 $A_{r=64}$ 的列向量之间的子空间相似性，对于 $\Delta W_q$ 和 $\Delta W_v$ 均如此。第三和第四幅图分别放大了前两幅图的左下三角区域。第 8 秩中的顶级方向被包含在第 64 秩中，反之亦然。
>
> *（详见 LoRA 论文原图 3——4 个子空间相似度热图，分别对应 $\Delta W_q$ 和 $\Delta W_v$ 在 $A_{r=8}$ vs $A_{r=64}$ 的相似度矩阵，y 轴为 $i$ (1..8)，x 轴为 $j$ (1..58)；最高相似度出现在 $i=1, j=1$ 附近，并随方向阶数增加而迅速衰减至 0）*

从图 3 中我们做出一个重要观察。
与 $A_{r=8}$ 和 $A_{r=64}$ 相关的顶级奇异向量之间显著重叠，而其他向量则不然。具体而言，$A_{r=8}$ 的 $\Delta W_v$（或 $\Delta W_q$）和 $A_{r=64}$ 的 $\Delta W_v$（或 $\Delta W_q$）共享一个维度为 1 的子空间，并且归一化相似度大于 0.5，这解释了为什么秩 r = 1 在我们的下游任务中表现得相当好。

由于 $A_{r=8}$ 和 $A_{r=64}$ 均使用相同的预训练模型学习得到，图 3 表明第 8 秩和第 64 秩中的顶级奇异向量方向是最有用的，而其他方向可能主要包含在训练过程中积累的随机噪声。因此，自适应矩阵确实可以具有非常低的秩。

不同随机种子下的子空间相似性。我们进一步通过绘制两个随机种子运行中 $r = 64$ 的归一化子空间相似度来确认这一点，结果见图 4。
$\Delta W_q$ 比 $\Delta W_v$ 具有更高的“固有秩”，因为对于 $\Delta W_q$ 来说，在两次运行中学习到的共同奇异值方向更多，这与我们在表 6 中的经验观察一致。作为对比，我们还绘制了两个随机高斯矩阵，它们之间没有任何共同的奇异值方向。

7.3 自适应矩阵 $\Delta W$ 与 $W$ 的比较
我们进一步研究自适应矩阵 $\Delta W$ 和原始权重矩阵 $W$ 之间的关系。具体来说，$\Delta W$ 是否高度相关于 $W$？（或从数学上讲，$\Delta W$ 主要包含在 $W$ 的顶级奇异方向中吗？）

---

## 原文第 12-12 页

> **图 4 描述**：左和中：两个不同随机种子生成的 $A_{r=64}$ 在第 48 层 $\Delta W_q$ 和 $\Delta W_v$ 的列向量之间的归一化子空间相似性。右：两个随机高斯矩阵的列向量之间的相同热图。参见附录 H.1 中的其他层。
>
> *（详见 LoRA 论文原图 4——3 个子空间相似度热图，分别对应 $\Delta W_q$、$\Delta W_v$ 与随机高斯矩阵在两个随机种子下 $r=64$ 的相似度矩阵；y 轴为 $i$ (1..64)，x 轴为 $j$ (1..59)；$\Delta W_q$ 顶部方向在两次随机种子间高度重合，$\Delta W_v$ 仅前 1~2 个方向重合，随机高斯矩阵的相似度接近 0）*

---

**如何"大"是 $\Delta W$ 相对于其对应的 $W$ 方向？这可以揭示预训练语言模型适应机制的内在原理。**

为了回答这些问题，我们通过计算 $U^\top W V^\top$ 将 $W$ 投影到 $\Delta W$ 的 $r$ 维子空间中，其中 $U/V$ 为 $\Delta W$ 的左/右奇异向量矩阵。然后，我们将 Frobenius 范数 $\|U^\top W V^\top\|_F$ 与 $\|W\|_F$ 进行比较。作为对比，我们还通过用 $W$ 或随机矩阵的前 $r$ 个奇异向量替换 $U,V$ 来计算 $\|U^\top W V^\top\|_F$。

| 指标 | r = 4 (∆Wq) | r = 4 (Wq) | r = 4 (随机) | r = 64 (∆Wq) | r = 64 (Wq) | r = 64 (随机) |
|------|-------------|------------|--------------|--------------|-------------|----------------|
| $\|U^\top W_q V^\top\|_F$ | 0.32 | 21.67 | 0.02 | 1.90 | 37.71 | 0.33 |
| $\|W_q\|_F$ | 61.95 | — | — | 61.95 | — | — |
| $\|\Delta W_q\|_F$ | 6.91 | — | — | 3.57 | — | — |

**表 7：$U^\top W_q V^\top$ 的 Frobenius 范数，其中 $U$ 和 $V$ 分别是 (1) $\Delta W_q$、(2) $W_q$、或者 (3) 随机矩阵的前 $r$ 个奇异向量方向。权重矩阵取自 GPT-3 第 48 层。**

从表 7 中我们得出几个结论。首先，$\Delta W$ 与 $W$ 的相关性比随机矩阵更强，表明 $\Delta W$ 放大了已经在 $W$ 中存在的某些特征。其次，$\Delta W$ 仅放大了在 $W$ 中未被强调的方向，而不是重复 $W$ 的顶级奇异方向。第三，放大的因子相当大：对于 $r = 4$，$21.5 \approx 6.91/0.32$。参见附录 H.4 了解为什么 $r = 64$ 时放大的因子较小。我们还在附录 H.3 中提供了如何随着从 $W_q$ 引入更多顶级奇异方向而变化的相关性可视化图。这表明低秩适应矩阵可能放大了特定下游任务中学习但未在通用预训练模型中强调的重要特征。

## 8 结论与未来工作
对庞大语言模型进行微调在硬件需求、存储/切换成本方面都是极其昂贵的，且不同任务需要独立实例时会增加额外开销。我们提出了一种高效的适应策略LoRA，在不引入推理延迟或减少输入序列长度的情况下保持高模型质量。重要的是，当作为服务部署时，它允许快速的任务切换，因为可以共享大部分模型参数。尽管我们专注于Transformer语言模型，但所提出的原理适用于任何具有密集层的神经网络。
未来工作有许多方向：1) LoRA 可以与其他高效的适应方法结合使用，可能提供互补改进；2) 微调或LoRA背后的机制尚不明确——预训练期间学习到的特征是如何转换为在下游任务中表现良好的？我们认为LoRA使得回答这个问题更加可行，而不仅仅是进行全微调。

---

## 原文第 13-13 页

调优。3) 我们主要依赖启发式方法来选择应用于 LoRA 的权重矩阵。是否有更严谨的方法可以实现这一点？4) 最后，∆W 的秩亏现象表明 W 也可能存在秩亏问题，这也可以为未来的研究提供灵感。

参考文献
1. Armen Aghajanyan, Luke Zettlemoyer, and Sonal Gupta. 入禀赋解释了语言模型微调的有效性。arXiv:2012.13255 [cs]，2020年12月。URL http://arxiv.org/abs/2012.13255.
2. Zeyuan Allen-Zhu 和 Yuanzhi Li. 什么是 ResNet 能高效学习的超越核？在 NeurIPS, 2019。完整版本参见 http://arxiv.org/abs/1905.10337。
3. Zeyuan Allen-Zhu 和 Yuanzhi Li. 向后特征校正：深度学习如何进行深度学习。arXiv 预印本 arXiv:2001.04413, 2020a。
4. Zeyuan Allen-Zhu 和 Yuanzhi Li. 特征净化：对抗训练如何实现稳健的深度学习。arXiv 预印本 arXiv:2005.10190, 2020b。
5. Zeyuan Allen-Zhu, Yuanzhi Li, and Zhao Song. 过参数化下的深层学习收敛理论。在 ICML, 2019。完整版本参见 http://arxiv.org/abs/1811.03962。
6. Jimmy Lei Ba, Jamie Ryan Kiros, 和 Geoffrey E. Hinton. 层归一化，2016。
7. Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, 和 Dario Amodei. 语言模型是少样本学习者。arXiv:2005.14165 [cs]，2020年7月。URL http://arxiv.org/abs/2005.14165。
8. Jian-Feng Cai, Emmanuel J Candès, 和 Zuowei Shen. 用于矩阵完成的奇异值阈值算法。优化期刊，20(4):1956–1982，2010年。
9. Daniel Cer, Mona Diab, Eneko Agirre, Inigo Lopez-Gazpio, 和 Lucia Specia. Semeval-2017 任务1：语义文本相似性多语言和跨语言集中评估。第十一届国际语义评价研讨会（SemEval-2017）论文集，2017年。doi: 10.18653/v1/s17-2001。URL http://dx.doi.org/10.18653/v1/S17-2001。
10. Ronan Collobert 和 Jason Weston. 自然语言处理的统一架构：深层神经网络与多任务学习。在第25届国际机器学习会议（ICML）上，ICML ’08, pp. 160–167, 纽约, NY , USA, 2008年7月。ACM 计算机协会。ISBN 978-1-60558-205-4。doi: 10.1145/1390156.1390177。URL https://doi.org/10.1145/1390156.1390177。
11. Misha Denil, Babak Shakibi, Laurent Dinh, Marc’Aurelio Ranzato, 和 Nando de Freitas. 深度学习中参数预测，2014年。
12. Jacob Devlin, Ming-Wei Chang, Kenton Lee, 和 Kristina Toutanova. BERT：用于语言理解的深度双向变压器预训练，2019a。
13. Jacob Devlin, Ming-Wei Chang, Kenton Lee, 和 Kristina Toutanova. BERT：用于语言理解的深度双向变压器预训练。arXiv:1810.04805 [cs]，2019b。URL http://arxiv.org/abs/1810.04805。arXiv: 1810.04805。
14. William B. Dolan 和 Chris Brockett. 自动构建短语对齐的语料库。在第三届平行句法研讨会（IWP2005）上，2005年。URL https://aclanthology.org/I05-5002。
15. Claire Gardent, Anastasia Shimorina, Shashi Narayan, 和 Laura Perez-Beltrachini. WebNLG 挑战：从RDF数据生成文本。在第十届自然语言生成国际会议（ICNG）上，pp. 124–133，2017年。

---

## 原文第 14-14 页

Behrooz Ghorbani, Song Mei, Theodor Misiakiewicz, and Andrea Montanari. 何时神经网络超越核方法？arXiv预印本 arXiv:2006.13409，2020。
Bogdan Gliwa, Iwona Mochol, Maciej Biesek, and Aleksander Wawer. Samsum语料库：一个由人工标注的对话数据集用于摘要生成。CoRR，abs/1911.12237，2019。URL http://arxiv.org/abs/1911.12237。
Lars Grasedyck, Daniel Kressner, and Christine Tobler. 低秩张量逼近技术综述。GAMM Mitteilungen，36(1)：53-78，2013。
Jihun Ham和Daniel D. Lee. 格拉斯曼判别分析：基于子空间学习的统一视角。在ICML中，pp. 376-383，2008。URL https://doi.org/10.1145/1390156。
Karen Hambardzumyan, Hrant Khachatrian, and Jonathan May. W ARP：字级对抗重编程。arXiv:2101.00121 [cs]，2020年12月。URL http://arxiv.org/abs/2101.00121。arXiv: 2101.00121。
Pengcheng He, Xiaodong Liu, Jianfeng Gao, and Weizhu Chen. DeBERTa：解码增强的BERT与解耦注意力，2021。
Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morrone, Quentin de Laroussilhe, Andrea Gesmundo, Mona Attariyan, and Sylvain Gelly. 参数高效迁移学习在NLP中的应用。arXiv:1902.00751 [cs, stat]，2019年6月。URL http://arxiv.org/abs/1902.00751。
Max Jaderberg, Andrea Vedaldi, and Andrew Zisserman. 通过低秩扩展加速卷积神经网络。arXiv预印本 arXiv:1405.3866，2014。
Mikhail Khodak, Neil Tenenholtz, Lester Mackey, and Nicol`o Fusi. 因子化神经层的初始化与正则化，2021。
Diederik P. Kingma和Jimmy Ba. Adam：随机优化的一种方法，2017。
Dmitry Lepikhin, HyoukJoong Lee, Yuanzhong Xu, Dehao Chen, Orhan Firat, Yanping Huang, Maxim Krikun, Noam Shazeer, and Zhifeng Chen. GShard：通过条件计算和自动分片扩展大型模型，2020。
Brian Lester, Rami Al-Rfou, 和 Noah Constant. 规模的力量：参数高效提示调优。arXiv:2104.08691 [cs]，2021年4月。URL http://arxiv.org/abs/2104.08691。
Chunyuan Li, Heerad Farkhoor, Rosanne Liu, and Jason Yosinski. 测量目标景观的固有维度。arXiv:1804.08838 [cs, stat]，2018年4月a。URL http://arxiv.org/abs/1804.08838。arXiv: 1804.08838。
Xiang Lisa Li和Percy Liang. 前缀调优：优化连续提示以生成。arXiv:2101.00190 [cs]，2021年1月。URL http://arxiv.org/abs/2101.00190。
Yuanzhi Li和Yingyu Liang. 在过参数化神经网络中通过随机梯度下降学习结构化数据上的目标函数。在神经信息处理系统进展会议上，2018年。
Yuanzhi Li, Yingyu Liang, 和 Andrej Risteski. 交替最小二乘法在加权低秩逼近中的恢复保证。在国际机器学习会议中，pp. 2358-2367。PMLR，2016。
Yuanzhi Li, Tengyu Ma, and Hongyang Zhang. 过参数化矩阵感知和具有二次激活的神经网络的算法正则化。在学习理论会议上，pp. 2-47。PMLR，2018b。
Zhaojiang Lin, Andrea Madotto, 和 Pascale Fung. 通过参数高效迁移学习探索多功能生成语言模型。在计算语言学会协会发现：EMNLP 2020 中，pp. 441-459，在线，2020年11月。计算语言学会。doi: 10.18653/v1/2020.findings-emnlp.41。URL https://aclanthology.org/2020.findings-emnlp.41。

---

## 原文第 15-15 页

Xiao Liu, Yanan Zheng, Zhengxiao Du, Ming Ding, Yujie Qian, Zhilin Yang, and Jie Tang. GPT
Understands, Too. arXiv:2103.10385 [cs], 2021年3月. URL <http://arxiv.org/abs/2103.10385>. arXiv: 2103.10385.
Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike
Lewis, Luke Zettlemoyer, and Veselin Stoyanov. Roberta: A Robustly Optimized BERT Pretraining
Approach, 2019.
Ilya Loshchilov and Frank Hutter. Decoupled Weight Decay Regularization. arXiv预印本 arXiv:1711.05101, 2017.
Ilya Loshchilov and Frank Hutter. Decoupled Weight Decay Regularization, 2019.
Rabeeh Karimi Mahabadi, James Henderson, and Sebastian Ruder. Compacter: Efficient Low-Rank
Hypercomplex Adapter Layers, 2021.
Linyong Nan, Dragomir Radev, Rui Zhang, Amrit Rau, Abhinand Sivaprasad, Chiachun Hsieh,
Xiangru Tang, Aadit Vyas, Neha Verma, Pranav Krishna, et al. Dart: Open-Domain Structured
Data Record to Text Generation. arXiv预印本 arXiv:2007.02871, 2020.
Jekaterina Novikova, Ondˇrej Duˇsek, and Verena Rieser. The E2E Dataset: New Challenges for End-
to-End Generation. arXiv预印本 arXiv:1706.09254, 2017.
Samet Oymak, Zalan Fabian, Mingchen Li, and Mahdi Soltanolkotabi. Generalization Guarantees
for Neural Networks via Harnessing the Low-Rank Structure of the Jacobian. arXiv预印本 arXiv:1906.05392, 2019.
Jonas Pfeiffer, Aishwarya Kamath, Andreas R¨uckl´e, Kyunghyun Cho, and Iryna Gurevych. Adapter-
Fusion: Non-Destructive Task Composition for Transfer Learning, 2021.
Daniel Povey, Gaofeng Cheng, Yiming Wang, Ke Li, Hainan Xu, Mahsa Yarmohammadi, and San-
jeev Khudanpur. Semi-Orthogonal Low-Rank Matrix Factorization for Deep Neural Networks. 在 Interspeech 会议，第3743–3747页，2018年。
Alec Radford, Karthik Narasimhan, Tim Salimans, and Ilya Sutskever. Improving Language Under-
standing by Generative Pre-Training. 第12页.
Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language
Models are Unsupervised Multitask Learners. 第24页。
Pranav Rajpurkar, Robin Jia, and Percy Liang. Know What You Don’t Know: Unanswerable Questions
for Squad. CoRR，abs/1806.03822，2018年。URL <http://arxiv.org/abs/1806.03822>。
Sylvestre-Alvise Rebuffi, Hakan Bilen, and Andrea Vedaldi. Learning Multiple Visual Domains with
Residual Adapters. arXiv:1705.08045 [cs, stat]，2017年11月。URL <http://arxiv.org/abs/1705.08045>. arXiv: 1705.08045.
Andreas R¨uckl´e, Gregor Geigle, Max Glockner, Tilman Beck, Jonas Pfeiffer, Nils Reimers, and
Iryna Gurevych. Adapterdrop: On the Efficiency of Adapters in Transformers, 2020.
Tara N Sainath, Brian Kingsbury, Vikas Sindhwani, Ebru Arisoy, and Bhuvana Ramabhadran. Low-
rank Matrix Factorization for Deep Neural Network Training with High-Dimensional Output Targets.
在2013年IEEE国际声学、说话人和信号处理会议，第6655–6659页。IEEE，2013年。
Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, and Bryan
Catanzaro. Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Par-
allelism, 2020.
Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher D. Manning, Andrew Ng,
and Christopher Potts. Recursive Deep Models for Semantic Compositionality over a Sentiment
Treebank. 在2013年会议Empirical Methods in Natural Language Processing，第1631–1642页，西雅图，美国华盛顿州，2013年10月。计算语言学协会。URL <https://aclanthology.org/D13-1170>.

---

## 原文第 16-16 页

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin. 注意力就是你所需要的全部。在第31届国际神经信息处理系统会议论文集上，第6000-6010页，2017年。
Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel R. Bowman. Glue：一个用于自然语言理解的多任务基准和分析平台，2019年。
Alex Wang, Yada Pruksachatkun, Nikita Nangia, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel R. Bowman. Superglue：一个更粘性的基准用于通用语言理解系统，2020年。
Alex Warstadt, Amanpreet Singh, and Samuel R. Bowman. 神经网络可接受性判断。arXiv预印本 arXiv:1805.12471，2018年。
Adina Williams, Nikita Nangia, and Samuel Bowman. 一个广泛覆盖的挑战语料库用于通过推理理解句子。在第2018年度北美计算语言学协会会议：人类语言技术会议上（长论文集），第1112-1122页，路易斯安那州新奥尔良市，2018年6月。计算语言学学会。doi: 10.18653/v1/N18-1101。URL https://www.aclweb.org/anthology/N18-1101。
Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, R. ´Emi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, and Alexander M. Rush. 变形器：最先进的自然语言处理。在2020年度会议上的实验方法在自然语言处理演示会上，第38-45页，在线发布，2020年10月。计算语言学学会。URL https://www.aclweb.org/anthology/2020.emnlp-demos.6。
Greg Yang and Edward J. Hu. 在无限宽度神经网络中的特征学习。arXiv:2011.14522 [cond-mat]，2021年5月。URL http://arxiv.org/abs/2011.14522。
arXiv: 2011.14522。
Elad Ben Zaken, Shauli Ravfogel, and Yoav Goldberg. BitFit：一种简单的参数高效微调方法用于基于变形器的掩码语言模型，2021年。
Yu Zhang, Ekapol Chuangsuwanich, and James Glass. 使用低秩矩阵分解提取深度神经网络瓶颈特征。在2014年度IEEE声学、语音和信号处理国际会议（ICASSP）上，第185-189页。IEEE，2014年。
Yong Zhao, Jinyu Li, and Yifan Gong. 低秩加对角线适应性用于深度神经网络。在2016年度IEEE声学、语音和信号处理国际会议（ICASSP）上，第5005-5009页。IEEE，2016年。
Victor Zhong, Caiming Xiong, and Richard Socher. Seq2SQL：使用强化学习从自然语言生成结构化查询。CoRR，abs/1709.00103，2017年。URL http://arxiv.org/abs/1709.00103。
大型语言模型仍然需要参数更新
少量样本的少-shot 学习或提示工程在我们只有少量训练样本时非常有利。然而，在实践中，对于性能敏感的应用程序，我们往往可以精心策划数千甚至更多的训练示例。如表8所示，在大规模和小规模数据集上，微调相比少-shot 学习显著提高了模型的性能。我们采用GPT-3论文中的RTE少-shot 结果（Brown等人，2020）。对于MNLI-matched，我们每类使用两个演示示例，并总共使用六个上下文示例。

---

## 原文第 17-17 页

### 表 8：微调 vs 少量样本学习

| 方法 | MNLI-m（验证准确率/%） | RTE（验证准确率/%） |
|------|-----------------------|---------------------|
| GPT-3 少量样本学习 | 40.6 | 69.0 |
| GPT-3 微调        | 89.5 | 85.4 |

**表 8 描述**：微调显著优于 GPT-3 的少量样本学习 (Brown et al., 2020)。

### 适配器层引入的推理延迟
适配器层是按顺序添加到预训练模型中的外部模块，而我们的提案 LoRA 则可以视为并行方式添加的外部模块。因此，适配器层必须在基础模型之外进行计算，不可避免地会增加额外的延迟。正如 Rücksle 等人在 (2020) 中指出的那样，在模型批量大小和/或序列长度足够大以充分利用硬件并行性时，适配器层引入的延迟可以被缓解。我们通过在 GPT-2 中进行类似的延迟研究确认了他们的观察结果，并指出在批量大小较小的情况下（例如在线推理），额外增加的延迟可能是显著的。

我们通过平均 100 次试验来测量单次前向传播的延迟，使用 NVIDIA Quadro RTX8000 进行测试。我们改变输入批量大小、序列长度以及适配器瓶颈维度 r。我们测试了两种适配器设计：Houlsby 等人 (2019) 的原始设计，称为 Adapter H，以及 Lin 等人 (2020) 更高效的变体，称为 Adapter L。更多细节见第 5.1 节。我们将与无适配器基线相比的延迟下降百分比绘制成图 5。

**图 5 描述**：与无适配器（r = 0）基线相比的推理延迟下降百分比。本图由 6 个子图组成：

- 第一行（Adapter H）：(a) 不同瓶颈维度 r 下的延迟变化；(b) 序列长度 128/256/512 下的延迟变化；(c) 批量大小 1~32 下的延迟变化。
- 第二行（Adapter L）：(d) 不同瓶颈维度 r 下的延迟变化；(e) 批量大小 1~32 下的延迟变化。

较大的批量大小和序列长度有助于缓解延迟，但在在线、短序列长度场景中，延迟下降可能高达 30% 以上。我们调整了颜色图以提高可读性。*原始 PDF 子图请参阅 LoRA 论文原图 5（附录 D.2 节）*。

### 数据集详情
GLUE 基准是一个广泛涵盖自然语言理解任务的集合。它包括 MNLI（推理，Williams 等人 (2018)）、SST-2（情感分析，Socher 等人 (2013)）、MRPC（同义句检测，Dolan & Brockett (2005)）、CoLA（语言接受性，Warstadt 等人 (2018)）、QNLI（推理，Rajpurkar 等人 (2018)）、QQP 8（问答）以及 RTE（推理）。  
8 https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs
17

---

## 原文第 18-18 页

并且包括 STS-B（文本相似度，Cer 等人 (2017)）。广泛的覆盖范围使得 GLUE 基准成为评估如 RoBERTa 和 DeBERTa 的 NLU 模型的标准指标。各个数据集在不同的宽松许可下发布。
WikiSQL 由 Zhong 等人 (2017) 引入，包含 56,355/8,421 训练/验证样本。任务是从自然语言问题和表结构生成 SQL 查询。我们将上下文编码为 \(x = \{表结构, 查询\}\)，目标编码为 \(y = \{SQL\}\)。该数据集在 BSD 3 条件许可下发布。
SAMSum 由 Gliwa 等人 (2019) 引入，包含 14,732/819 训练/测试样本。它由两个人之间的分阶段对话和相应的由语言学家撰写的摘要组成。我们将上下文编码为“ \n”连接的发言后跟一个“ \n\n”，目标编码为 \(y = \{摘要\}\)。该数据集在非商业许可：Creative Commons BY-NC-ND 4.0 下发布。
E2E 自然语言生成挑战最初由 Novikova 等人 (2017) 引入，作为训练端到端、数据驱动的自然语言生成系统的数据集，并常用于数据到文本评估。E2E 数据集包含大约 42,000 个训练样本、4,600 个验证样本和 4,600 个测试样本，均来自餐厅领域。每个用作输入的源表可以有多个参考。每个样本输入 (x,y) 包含一系列槽值对，并附带相应的自然语言参考文本。该数据集在 Creative Commons BY-NC-SA 4.0 许可下发布。
DART 是一个开放领域的数据到文本数据集，由 Nan 等人 (2020) 描述。DART 输入结构化为实体—关系—实体三元组序列。总共包含 82,000 个样本，DART 比较而言是一个显著更大且更复杂的开放领域数据到文本任务，相比 E2E。该数据集在 MIT 许可下发布。
WebNLG 是另一个常用于数据到文本评估的数据集（Gardent 等人, 2017）。总共包含 22,000 个样本的 WebNLG 包含 14 个不同的类别，其中九个在训练中可见。由于四个总类别的五个未在训练集中出现但在测试集中代表，评估通常按“可见”类别（S）、“不可见”类别（U）和“全部”（A）进行划分。每个输入样本由一系列主语—属性—宾语三元组表示。该数据集在 Creative Commons BY-NC-SA 4.0 许可下发布。
D 实验中使用的超参数

### D.1 RoBERTa
我们使用 AdamW 进行训练，并采用线性学习率衰减计划。我们在 LoRA 中扫掠学习率、训练周期数和批量大小。根据 Liu 等人 (2019)，在适应 MRPC、RTE 和 STS-B 时，我们将 LoRA 模块初始化为我们的最佳 MNLI 检查点，而不是通常的初始化；预训练模型在所有任务中保持冻结状态。我们报告了五个随机种子的中位数结果；每次运行的结果来自最佳周期。为了与 Houlsby 等人 (2019) 和 Pfeiffer 等人 (2021) 的设置进行公平比较，我们将模型序列长度限制为 128，并为所有任务使用固定批量大小。重要的是，在适应 MRPC、RTE 和 STS-B 时，我们从预训练的 RoBERTa 大型模型开始，而不是一个已经适应 MNLI 的模型。这些受限设置下的运行标记为†。请参阅我们在实验中使用的超参数在表 9 中。
D.2 DeBERTa
我们再次使用 AdamW 进行训练，并采用线性学习率衰减计划。根据 He 等人 (2021)，我们调整了学习率、dropout 概率、预热步数和批量大小。我们使用与 (He 等人, 2021) 相同的模型序列长度以保持比较的一致性。根据 He 等人 (2021)，在适应 MRPC、RTE 和 STS-B 时，我们将 LoRA 模块初始化为我们的最佳 MNLI 检查点，而不是通常的初始化；预训练模型在所有任务中保持冻结状态。我们报告了五个随机种子的中位数结果；每次运行的结果来自最佳周期。请参阅我们在实验中使用的超参数在表 10 中。
18

---

## 原文第 19-19 页

**表 9：我们在 GLUE 基准上使用的 RoBERTa 超参数。**

| 通用参数 | 值 |
|----------|-----|
| 优化器 | AdamW |
| 预热比例 | 0.06 |
| 学习率调度 | 线性 |

| 模型与方法 | MNLI | SST-2 | MRPC | CoLA | QNLI | QQP | RTE | STS-B |
|------------|------|-------|------|------|------|------|------|-------|
| **RoBERTa base + LoRA** |  |  |  |  |  |  |  |  |
| 批量大小  | 16 | 16 | 16 | 32 | 32 | 16 | 32 | 16 |
| 训练轮数  | 30 | 60 | 30 | 80 | 25 | 25 | 80 | 40 |
| 学习率    | $5{\times}10^{-4}$ | $5{\times}10^{-4}$ | $4{\times}10^{-4}$ | $4{\times}10^{-4}$ | $4{\times}10^{-4}$ | $5{\times}10^{-4}$ | $5{\times}10^{-4}$ | $4{\times}10^{-4}$ |
| LoRA 配置 | $r_q = r_v = 8$ (全表) |
| LoRA α    | 8 (全表) |
| 最大序列长度 | 512 (全表) |
| **RoBERTa large + LoRA** |  |  |  |  |  |  |  |  |
| 批量大小  | 4 | 4 | 4 | 4 | 4 | 4 | 8 | 8 |
| 训练轮数  | 10 | 10 | 20 | 20 | 10 | 20 | 20 | 30 |
| 学习率    | $3{\times}10^{-4}$ | $4{\times}10^{-4}$ | $3{\times}10^{-4}$ | $2{\times}10^{-4}$ | $2{\times}10^{-4}$ | $3{\times}10^{-4}$ | $4{\times}10^{-4}$ | $2{\times}10^{-4}$ |
| LoRA 配置 | $r_q = r_v = 8$ (全表) |
| LoRA α    | 16 (全表) |
| 最大序列长度 | 128 / 128 / 512 / 128 / 512 / 512 / 512 / 512 |
| **RoBERTa large + LoRA†** |  |  |  |  |  |  |  |  |
| 批量大小  | 4 (全表) |
| 训练轮数  | 10 | 10 | 20 | 20 | 10 | 20 | 20 | 10 |
| 学习率    | $3{\times}10^{-4}$ | $4{\times}10^{-4}$ | $3{\times}10^{-4}$ | $2{\times}10^{-4}$ | $2{\times}10^{-4}$ | $3{\times}10^{-4}$ | $4{\times}10^{-4}$ | $2{\times}10^{-4}$ |
| LoRA 配置 | $r_q = r_v = 8$ (全表) |
| LoRA α    | 16 (全表) |
| 最大序列长度 | 128 (全表) |
| **RoBERTa large + AdptP (3M)†** |  |  |  |  |  |  |  |  |
| 批量大小  | 32 (全表) |
| 训练轮数  | 10 | 20 | 20 | 20 | 10 | 20 | 20 | 20 |
| 学习率    | $3{\times}10^{-5}$ | $3{\times}10^{-5}$ | $3{\times}10^{-4}$ | $3{\times}10^{-4}$ | $3{\times}10^{-4}$ | $3{\times}10^{-4}$ | $3{\times}10^{-4}$ | $3{\times}10^{-4}$ |
| 瓶颈维度 r | 64 (全表) |
| 最大序列长度 | 128 (全表) |
| **RoBERTa large + AdptP (0.8M)†** |  |  |  |  |  |  |  |  |
| 批量大小  | 32 (全表) |
| 训练轮数  | 5 | 20 | 20 | 20 | 10 | 20 | 20 | 20 |
| 学习率    | $3{\times}10^{-4}$ (全表) |
| 瓶颈维度 r | 16 (全表) |
| 最大序列长度 | 128 (全表) |
| **RoBERTa large + AdptH (6M)†** |  |  |  |  |  |  |  |  |
| 批量大小  | 32 (全表) |
| 训练轮数  | 10 | 5 | 10 | 10 | 5 | 20 | 20 | 10 |
| 学习率    | $3{\times}10^{-5}$ | $3{\times}10^{-4}$ | $3{\times}10^{-4}$ | $3{\times}10^{-4}$ | $3{\times}10^{-4}$ | $3{\times}10^{-4}$ | $3{\times}10^{-4}$ | $3{\times}10^{-4}$ |
| 瓶颈维度 r | 64 (全表) |
| 最大序列长度 | 128 (全表) |
| **RoBERTa large + AdptH (0.8M)†** |  |  |  |  |  |  |  |  |
| 批量大小  | 32 (全表) |
| 训练轮数  | 10 | 5 | 10 | 10 | 5 | 20 | 20 | 10 |
| 学习率    | $3{\times}10^{-4}$ (全表) |
| 瓶颈维度 r | 8 (全表) |
| 最大序列长度 | 128 (全表) |

### D.3 GPT-2
我们使用AdamW (Loshchilov & Hutter, 2017) 和线性学习率调度训练所有GPT-2模型，共5轮。我们采用Li & Liang (2021) 中描述的批量大小、学习率和束搜索束大小。相应地，我们也为LoRA调整了上述超参数。我们在3个随机种子上报告均值；每个运行的结果取自最佳训练轮次。GPT-2中使用的LoRA超参数见表11。其他基线模型所用的超参数请参阅Li & Liang (2021)。

### D.4 GPT-3
对于所有GPT-3实验，我们使用AdamW (Loshchilov & Hutter, 2017)，批量大小为128样本，训练2轮，并设置权重衰减因子为0.1。序列长度设为384。

---

## 原文第 20-20 页

**表 10：DeBERTa XXL 在 GLUE 基准任务中的超参数。**

| 参数 | MNLI | SST-2 | MRPC | CoLA | QNLI | QQP | RTE | STS-B |
|------|------|-------|------|------|------|------|------|-------|
| 优化器 | AdamW (所有任务) |   |   |   |   |   |   |   |
| 预热比例 | 0.1 (所有任务) |   |   |   |   |   |   |   |
| 学习率调度 | 线性 (所有任务) |   |   |   |   |   |   |   |
| DeBERTa XXL | 是 (所有任务) |   |   |   |   |   |   |   |
| LoRA | 是 (所有任务) |   |   |   |   |   |   |   |
| 批量大小 | 8 | 8 | 32 | 4 | 6 | 8 | 4 | 4 |
| 训练轮数 | 5 | 16 | 30 | 10 | 8 | 11 | 11 | 10 |
| 学习率 | $1{\times}10^{-4}$ | $6{\times}10^{-5}$ | $2{\times}10^{-4}$ | $1{\times}10^{-4}$ | $1{\times}10^{-4}$ | $1{\times}10^{-4}$ | $2{\times}10^{-4}$ | $2{\times}10^{-4}$ |
| 权重衰减 | 0 | 0.01 | 0.01 | 0 | 0.01 | 0.01 | 0.01 | 0.1 |
| CLS 丢弃率 | 0.15 | 0 | 0 | 0.1 | 0.1 | 0.2 | 0.2 | 0.2 |
| LoRA 配置 | $r_q = r_v = 8$ (全表) |
| LoRA α   | 8 (全表) |
| 最大序列长度 | 256 | 128 | 128 | 64 | 512 | 320 | 320 | 128 |

**表 11：GPT-2 LoRA 在 E2E、WebNLG 和 DART 中的超参数。**

| 参数 | E2E | WebNLG | DART |
|------|-----|--------|------|
| 训练优化器 | AdamW (所有任务) |   |   |
| 权重衰减 | 0.01 | 0.01 | 0.0 |
| Dropout 概率 | 0.1 | 0.1 | 0.0 |
| 批量大小 | 8 (所有任务) |   |   |
| 训练轮数 | 5 (所有任务) |   |   |
| 预热步数 | 500 (所有任务) |   |   |
| 学习率调度 | 线性 (所有任务) |   |   |
| 标签平滑度 | 0.1 | 0.1 | 0.0 |
| 学习率 | $2{\times}10^{-4}$ (所有任务) |   |   |
| LoRA 配置 | $r_q = r_v = 4$ (所有任务) |   |   |
| LoRA α   | 32 (所有任务) |   |   |
| 解码器束大小 | 10 (所有任务) |   |   |
| 长度惩罚因子 | 0.9 | 0.8 | 0.8 |
| 禁止重复 n-gram 大小 | 4 (所有任务) |   |   |

### WikiSQL (Zhong et al., 2017)，MNLI (Williams et al., 2018) 和 SAMSum (Gliwa et al., 2019) 的超参数设置为768、2048。我们对所有方法-数据集组合进行了学习率调整。更多关于使用的超参数细节见第D.4节。对于前缀嵌入调优，我们发现最优的lp和li分别为256和8，总计3.2M可训练参数。使用lp = 8和li = 8进行前缀层调优，共有20.2M可训练参数以获得最佳性能。我们为LoRA提供了两种参数预算：4.7M（$r_q = r_v = 1$ 或 $r_v = 2$）和37.7M（$r_q = r_v = 8$ 或 $r_q = r_k = r_v = r_o = 2$）。我们从每次运行中报告最佳验证性能。我们的GPT-3实验中的训练超参数见表12。

### 结合LoRA与前缀调优
LoRA可以自然地与现有的基于前缀的方法相结合。在本节中，我们将评估两种LoRA和前缀调优变体的组合在WikiSQL和MNLI上的表现。

LoRA+前缀嵌入（LoRA+PE）结合了LoRA与前缀嵌入调优，其中我们插入lp + li个特殊标记，并将其嵌入视为可训练参数。关于前缀嵌入调优的更多细节见第5.1节。

LoRA+前缀层（LoRA+PL）结合了LoRA与前缀层调优。同样地，我们也插入了lp + li个特殊标记；然而，不同之处在于这些标记的隐藏表示不会自然演化。

---

## 原文第 21-21 页

Hyper参数微调预嵌入层预层BitFit适配器H低秩自适应（Low-Rank Adaptation, LoRA）
优化器AdamW
批量大小128
# 训练周期2
预热令牌250,000
学习率调度线性
学习率$5.00 \times 10^{-6}$ $5.00 \times 10^{-4}$ $1.00 \times 10^{-4}$ $1.6 \times 10^{-3}$ $1.00 \times 10^{-4}$ $2.00 \times 10^{-4}$
表12：不同GPT-3微调方法的训练Hyper参数。我们对所有数据集在调整学习率后使用相同的Hyper参数。
此外，我们在每个Transformer块之后用一个输入无关向量替换它们。因此，嵌入和后续的Transformer激活都被视为可训练参数。关于预层微调的更多内容，请参见第5.1节。
表15展示了LoRA+PE和LoRA+PL在WikiSQL和MultiNLI上的评估结果。首先，LoRA+PE显著优于LoRA和预嵌入微调，在WikiSQL上表现更佳，这表明LoRA与预嵌入微调之间存在一定程度的正交性。在MultiNLI上，LoRA+PE的表现并不比LoRA更好，可能是因为LoRA本身已经达到了与人类基线相当的性能。其次，我们注意到，即使具有更多的可训练参数，LoRA+PL的表现也略逊于LoRA。我们将这一现象归因于预层微调对学习率选择的高度敏感性，这使得在LoRA+PL中优化LoRA权重变得更加困难。
F 额外的实证实验
F.1 GPT-2额外实验
我们还按照 Li & Liang (2021) 的设置重复了 DART（Nan et al., 2020）和 WebNLG（Gardent et al., 2017）上的实验。结果如表 13 所示。类似于我们在 E2E NLG 挑战中报告的结果，第 5 节所述，给定相同的可训练参数数量时，LoRA 的表现优于或至少与基于预层的方法相当。

| 模型 | 方法 | # 可训练参数 | BLEU ↑ | MET ↑ | TER ↓ |
|------|------|--------------|--------|-------|-------|
| GPT-2 Medium | 微调 (Fine-Tune)  | 354M   | 46.2   | 0.39  | 0.46  |
| GPT-2 Medium | AdapterL         | 0.37M  | 42.4   | 0.36  | 0.48  |
| GPT-2 Medium | AdapterL         | 11M    | 45.2   | 0.38  | 0.46  |
| GPT-2 Medium | FTTop2           | 24M    | 41.0   | 0.34  | 0.56  |
| GPT-2 Medium | 预层 (PreLayer)  | 0.35M  | 46.4   | 0.38  | 0.46  |
| GPT-2 Medium | LoRA             | 0.35M  | $47.1 \pm .2$ | 0.39  | 0.46  |
| GPT-2 Large  | 微调 (Fine-Tune)  | 774M   | 47.0   | 0.39  | 0.46  |
| GPT-2 Large  | AdapterL         | 0.88M  | $45.7 \pm .1$ | 0.38  | 0.46  |
| GPT-2 Large  | AdapterL         | 23M    | $47.1 \pm .1$ | 0.39  | 0.45  |
| GPT-2 Large  | 预层 (PreLayer)  | 0.77M  | 46.7   | 0.38  | 0.45  |
| GPT-2 Large  | LoRA             | 0.77M  | $47.5 \pm .1$ | 0.39  | 0.45  |

**表 13：不同微调方法的 GPT-2 在 DART 上的表现。所有微调方法中 MET 和 TER 的标准差均小于 0.01。**
21

---

## 原文第 22-22 页

### 方法 WebNLG

| BLEU | MET | TER |
| --- | --- | --- |
| U S A | U S A | U S A |
| GPT-2 中型模型 | 微调 (354M) | 27.7 | 64.2 | 46.5 | .30 | .45 | .38 | .76 | .33 | .53 |
| AdapterL (0.37M) | 45.1 | 54.5 | 50.2 | .36 | .39 | .38 | .46 | .40 | .43 |
| AdapterL (11M) | 48.3 | 60.4 | 54.9 | .38 | .43 | .41 | .45 | .35 | .39 |
| FTTop2 (24M) | 18.9 | 53.6 | 36.0 | .23 | .38 | .31 | .99 | .49 | .72 |
| 前缀 (0.35M) | 45.6 | 62.9 | 55.1 | .38 | .44 | .41 | .49 | .35 | .40 |
| LoRA (0.35M) | 46.7±.4 | 62.1±.2 | 55.3±.2 | .38 | .44 | .41 | .46 | .33 | .39 |

| GPT-2 大型模型 | 微调 (774M) | 43.1 | 65.3 | 55.5 | .38 | .46 | .42 | .53 | .33 | .42 |
| AdapterL (0.88M) | 49.8±.0 | 61.1±.0 | 56.0±.0 | .38 | .43 | .41 | .44 | .35 | .39 |
| AdapterL (23M) | 49.2±.1 | 64.7±.2 | 57.7±.1 | .39 | .46 | .43 | .46 | .33 | .39 |
| 前缀 (0.77M) | 47.7 | 63.4 | 56.3 | .39 | .45 | .42 | .48 | .34 | .40 |
| LoRA (0.77M) | 48.4±.3 | 64.0±.3 | 57.0±.1 | .39 | .45 | .42 | .45 | .32 | .38 |

**表 14：不同适应方法在 WebNLG 上的 GPT-2。所有实验中 MET 和 TER 的方差均小于 0.01。“U”表示未见过的类别，“S”表示见过的类别，“A”表示 WebNLG 测试集中的所有类别。**

### F.2 GPT-3 不同适应方法的附加实验

我们在表 15 中展示了不同适应方法在 GPT-3 上的额外运行结果，重点在于识别性能与可训练参数数量之间的权衡。

### F.3 低数据域

为了评估不同适应方法在低数据域的表现，在 MNLI 的完整训练集上随机抽取 100、1k 和 10k 训练样本形成低数据 MNLI-n 任务。表 16 显示了不同适应方法在 MNLI-n 上的性能。令人惊讶的是，前缀嵌入和前缀层在 MNLI-100 数据集上的表现非常差，其中前缀嵌入的表现仅略优于随机猜测（37.6% 对 33.3%）。前缀层虽然比前缀嵌入表现稍好，但在 MNLI-100 上仍然显著劣于微调或 LoRA。随着训练样本数量的增加，基于前缀的方法与 LoRA/微调之间的差距逐渐缩小，这可能表明基于前缀的方法不适合 GPT-3 的低数据任务。LoRA 在 MNLI-100 和 MNLI-Full 上均优于微调，并且在考虑随机种子导致的 (±0.3) 方差时，在 MNLI-1k 和 MNLI-10K 上取得了可比的结果。

不同适应方法在 MNLI-n 上的训练超参数见表 17。我们在 MNLI-100 集上为前缀层使用了较小的学习率，因为较大的学习率并没有使训练损失减少。

### G 测量子空间之间的相似性

本文中我们使用 $\varphi(A, B, i, j) = \psi(U_i^A, U_j^B) = \|U_i^{A\top} U_j^B\|_F / \min\{i, j\}$ 来测量两个列正交矩阵 $U_i^A \in \mathbb{R}^{d \times i}$ 和 $U_j^B \in \mathbb{R}^{d \times j}$ 之间的子空间相似性，这两个矩阵分别由 $A$ 和 $B$ 的左奇异值矩阵的列获得。我们指出这种相似性仅仅是标准投影度量的反向版本，该度量用于衡量子空间间的距离 Ham & Lee (2008)。

> **表 14 描述**：不同适应方法在 WebNLG 上的 GPT-2。所有实验中 MET 和 TER 的方差均小于 0.01。"U"表示未见过的类别，"S"表示见过的类别，"A"表示 WebNLG 测试集中的所有类别。
>
> **表 15 描述**：不同适应方法在 GPT-3 上的额外运行结果。重点在于识别性能与可训练参数数量之间的权衡。
>
> **表 16 描述**：不同适应方法在 MNLI-n 上的表现。令人惊讶的是，前缀嵌入和前缀层在 MNLI-100 数据集上的表现非常差，其中前缀嵌入的表现仅略优于随机猜测（37.6% 对 33.3%）。前缀层虽然比前缀嵌入表现稍好，但在 MNLI-100 上仍然显著劣于微调或 LoRA。随着训练样本数量的增加，基于前缀的方法与 LoRA/微调之间的差距逐渐缩小，这可能表明基于前缀的方法不适合 GPT-3 的低数据任务。LoRA 在 MNLI-100 和 MNLI-Full 上均优于微调，并且在考虑随机种子导致的 (±0.3) 方差时，在 MNLI-1k 和 MNLI-10K 上取得了可比的结果。
>
> **表 17 描述**：不同适应方法在 MNLI-n 上的训练超参数。我们在 MNLI-100 集上为前缀层使用了较小的学习率，因为较大的学习率并没有使训练损失减少。

**子空间相似度公式：**
$$
\varphi(A, B, i, j) = \psi(U_i^A, U_j^B) = \|U_i^{A\top} U_j^B\|_F / \min\{i, j\}
$$

这是用于测量两个列正交矩阵之间的子空间相似性的度量，这两个矩阵分别由 $A$ 和 $B$ 的左奇异值矩阵的列获得。这种相似性仅仅是标准投影度量的反向版本，该度量用于衡量子空间间的距离 Ham & Lee (2008)。

---

## 原文第 23-23 页

| 方法分类 | 方法超参数 | # 可训练参数 | WikiSQL | MNLI-m |
|----------|------------|--------------|---------|--------|
| Fine-Tune | - | 175B | 73.8 | 89.5 |
| PrefixEmbed | lp = 32, li = 8 | 0.4M  | 55.9 | 84.9 |
| PrefixEmbed | lp = 64, li = 8 | 0.9M  | 58.7 | 88.1 |
| PrefixEmbed | lp = 128, li = 8 | 1.7M | 60.6 | 88.0 |
| PrefixEmbed | lp = 256, li = 8 | 3.2M | 63.1 | 88.6 |
| PrefixEmbed | lp = 512, li = 8 | 6.4M | 55.9 | 85.8 |
| PrefixLayer | lp = 2, li = 2 | 5.1M   | 68.5 | 89.2 |
| PrefixLayer | lp = 8, li = 0 | 10.1M  | 69.8 | 88.2 |
| PrefixLayer | lp = 8, li = 8 | 20.2M  | 70.1 | 89.5 |
| PrefixLayer | lp = 32, li = 4 | 44.1M | 66.4 | 89.6 |
| PrefixLayer | lp = 64, li = 0 | 76.1M | 64.9 | 87.9 |
| AdapterH | r = 1 | 7.1M   | 71.9 | 89.8 |
| AdapterH | r = 4 | 21.2M  | 73.2 | 91.0 |
| AdapterH | r = 8 | 40.1M  | 73.2 | 91.5 |
| AdapterH | r = 16 | 77.9M | 73.2 | 91.5 |
| AdapterH | r = 64 | 304.4M | 72.6 | 91.5 |
| LoRA | rv = 2 | 4.7M   | 73.4 | 91.7 |
| LoRA | rq = rv = 1 | 4.7M  | 73.4 | 91.3 |
| LoRA | rq = rv = 2 | 9.4M  | 73.3 | 91.4 |
| LoRA | rq = rk = rv = ro = 1 | 9.4M  | 74.1 | 91.2 |
| LoRA | rq = rv = 4 | 18.8M | 73.7 | 91.3 |
| LoRA | rq = rk = rv = ro = 2 | 18.8M | 73.7 | 91.7 |
| LoRA | rq = rv = 8 | 37.7M | 73.8 | 91.6 |
| LoRA | rq = rk = rv = ro = 4 | 37.7M | 74.0 | 91.7 |
| LoRA | rq = rv = 64 | 301.9M | 73.6 | 91.4 |
| LoRA | rq = rk = rv = ro = 64 | 603.8M | 73.9 | 91.4 |
| LoRA+PE | rq = rv = 8, lp = 8, li = 4 | 37.8M  | 75.0 | 91.4 |
| LoRA+PE | rq = rv = 32, lp = 8, li = 4 | 151.1M | 75.9 | 91.1 |
| LoRA+PE | rq = rv = 64, lp = 8, li = 4 | 302.1M | 76.2 | 91.3 |
| LoRA+PL | rq = rv = 8, lp = 8, li = 4 | 52.8M  | 72.9 | 90.2 |

**表 15：不同适应方法在 WikiSQL 和 MNLI 上的超参数分析。随着可训练参数数量的增加，前缀嵌入调优（PrefixEmbed）和前缀层调优（PrefixLayer）的表现逐渐变差，而 LoRA 的表现趋于稳定。性能以验证准确率衡量。**

| 方法 | MNLI(m)-100 | MNLI(m)-1k | MNLI(m)-10k | MNLI(m)-392K |
|------|------------|------------|-------------|--------------|
| GPT-3 (Fine-Tune)   | 60.2 | 85.8 | 88.9 | 89.5 |
| GPT-3 (PrefixEmbed) | 37.6 | 75.2 | 79.5 | 88.6 |
| GPT-3 (PrefixLayer) | 48.3 | 82.5 | 85.9 | 89.6 |
| GPT-3 (LoRA)        | 63.8 | 85.6 | 89.2 | 91.7 |

**表 16：使用 GPT-3 175B 在 MNLI 不同子集上的验证准确率。MNLI-n 描述了一个包含 n 个训练样本的子集。我们用完整验证集进行评估。LoRA 相比其他方法，包括微调，在样本效率方面表现出更优性能。**
具体而言，令Ui⊤A UjB 的奇异值为σ1, σ2,..., σp，其中 p = min{i,j}。我们知道投影度量Ham & Lee (2008) 定义如下：
$$
d(U_i^{\top} A, U_j^{\top} B) = \sqrt{p - \sum_{i=1}^{p} \sigma_i^2} \in [0, \sqrt{p}]
$$

---

## 原文第 24-24 页

| Hyper 参数 | MNLI-100 | MNLI-1k | MNLI-10K | MNLI-392K |
|-------------|----------|---------|----------|-----------|
| 优化器      | AdamW (所有任务) |   |   |   |
| 预热 Token 数 | 250,000 (所有任务) |   |   |   |
| 学习率调度  | 线性 (所有任务) |   |   |   |
| 批次大小    | 20   | 20   | 100  | 128  |
| 训练轮次    | 40   | 40   | 4    | 2    |
| 微调学习率  | 5.00E-06 (所有任务) |   |   |   |
| 前缀嵌入学习率 | 2.00E-04 | 2.00E-04 | 4.00E-04 | 5.00E-04 |
| 前缀层学习率 | 5.00E-05 | 5.00E-05 | 5.00E-05 | 1.00E-04 |
| LoRA 学习率 | 2.00E-4 (所有任务) |   |   |   |
| 前缀嵌入 $l_p$ | 16  | 32  | 64  | 256 |
| 前缀嵌入 $l_i$ | 8 (所有任务) |   |   |   |
| 特定前缀微调 $l_p = l_i = 8$ | — | — | — | — |
| LoRA $r_q = r_v = 8$ | — | — | — | — |

**表 17：不同 GPT-3 适应方法在 MNLI(m)-n 上的 Hyper 参数设置。**
其中，我们定义相似度为：
\[ \varphi(A, B, i, j) = \psi(U_i^A, U_j^B) = \sum_{p=1}^{P} \sigma_p^2 \]
\[ P = \frac{1}{d(U_i^A, U_j^B)^2 + 1} \]
此相似度满足：如果 \(U_i^A\) 和 \(U_j^B\) 共享相同的列空间，则 \(\varphi(A, B, i, j) = 1\)；如果它们完全正交，则 \(\varphi(A, B, i, j) = 0\)，否则 \(\varphi(A, B, i, j) \in (0, 1)\)。

H.1 低秩矩阵模块之间的相关性
请参见图6和图7以了解图3和图4的结果如何推广到其他层。

H.2 r 对 GPT-2 的影响
我们在GPT-2中重复了关于r的影响（第7.2节）的实验。以E2E NLG挑战集为例，我们报告了训练26,000步后不同选择的r所获得的验证损失和测试指标。结果见表18。对于GPT-2 Medium模型，最优秩取决于使用的度量标准，在4到16之间变化，这与GPT-3 175B的结果相似。值得注意的是，关于模型大小与适应性最优秩之间的关系仍然存在开放问题。

H.3 W 和 ∆W 的相关性
请参见图8以了解不同r下W和∆W的归一化子空间相似度。
再次注意，由于∆W中前4个方向与W中前10%方向之间的相似度几乎不超过0.2，因此∆W包含那些“任务特定”的方向，而这些方向在W中并未被强调。
一个有趣的问题是：为了使模型适应性良好，我们需将这些任务特定的方向放大到何种程度？

表18：不同r选择下GPT-2 Medium的验证损失和测试指标

**图 6 / 图 7 / 图 8**：详见 LoRA 论文原图（H.1 / H.2 / H.3 节附录），本翻译版未提取图像。

---

## 原文第 25-25 页

> **图 6**：在 96 层 Transformer 中，第 1、32、64 和 96 层的 $\Delta W_q$ 和 $\Delta W_v$ 的列向量之间的归一化子空间相似性。对于 $A_r = 8$ 和 $A_r = 64$。
>
> *（详见 LoRA 论文原图 6——4 × 4 = 16 个子空间相似度热图，分别对应层 1/32/64/96、$\Delta W_q$/$\Delta W_v$、$A_r=8$/$A_r=64$ 的组合；y 轴为 $i$ (1..8)，x 轴为 $j$ (1..58)；所有层的结果与第 48 层一致，验证了"自适应矩阵低秩"的结论可推广到所有层）*

### H.4 放大因子

自然可以考虑特征放大因子为比值 $\frac{\|\Delta W\|_F}{\|U^\top W V^\top\|_F}$，其中 $U$ 和 $V$ 分别是 $\Delta W$ 的 SVD 分解的左奇异矩阵和右奇异矩阵。（回忆一下 $UU^\top W V^\top V$ 给出了"投影"$W$ 到由 $\Delta W$ 张成的子空间。）

直观上，当 $\Delta W$ 主要包含任务特定的方向时，这个量度测量了这些方向被 $\Delta W$ 放大的程度。如第 7.3 节所示，对于 $r = 4$，此放大因子高达 20。换句话说，在预训练模型 $W$ 的整个特征空间中，每层通常有四个特征方向需要通过一个非常大的因子 20 来放大，以实现我们报告的下游特定任务的准确性。并且，每个不同的下游任务应该会放大一组完全不同的特征方向。

然而，可以注意到对于 $r = 64$，此放大因子仅为约 2，这意味着大多数在 $\Delta W$ 中学习的方向通过 $r = 64$ 并未被显著放大。这并不令人惊讶，并且事实上再次证明了表示"任务特定方向"（因此用于模型适应）所需的固有秩较低。相比之下，在 $\Delta W$ 的秩-4 版本中对应于 $r = 4$ 的那些方向通过一个更大的因子 20 被放大。

---

## 原文第 26-26 页

> **图 7**：两个随机种子运行中从 Transformer 第 1/32/64/96 层提取的 $A_{r=64}$ 列向量之间的归一化子空间相似性，分别针对 $\Delta W_q$ 和 $\Delta W_v$。
>
> *（详见 LoRA 论文原图 7——2 × 4 = 8 个子空间相似度热图，y 轴为 $i$ (1..64)，x 轴为 $j$ (1..59)；说明两个随机种子在第 96 层的结果可推广到所有层）*

### 表 18：不同 $r$ 选择下 GPT-2 Medium 的验证损失和测试指标

| 秩 $r$ | 验证损失 | BLEU | NIST | METEOR | ROUGE-L | CIDEr |
|---------|----------|------|------|--------|---------|-------|
| 1       | 1.23     | 68.72 | 8.7215 | 0.4565 | 0.7052 | 2.4329 |
| 2       | 1.21     | 69.17 | 8.7413 | 0.4590 | 0.7052 | 2.4639 |
| 4       | 1.18     | 70.38 | 8.8439 | 0.4689 | 0.7186 | 2.5349 |
| 8       | 1.17     | 69.57 | 8.7457 | 0.4636 | 0.7196 | 2.5196 |
| 16      | 1.16     | 69.61 | 8.7483 | 0.4629 | 0.7177 | 2.4985 |
| 32      | 1.16     | 69.33 | 8.7736 | 0.4642 | 0.7105 | 2.5255 |
| 64      | 1.16     | 69.24 | 8.7174 | 0.4651 | 0.7180 | 2.5070 |
| 128     | 1.16     | 68.73 | 8.6718 | 0.4628 | 0.7127 | 2.5030 |
| 256     | 1.16     | 68.92 | 8.6982 | 0.4629 | 0.7128 | 2.5012 |
| 512     | 1.16     | 68.78 | 8.6857 | 0.4637 | 0.7128 | 2.5025 |
| 1024    | 1.17     | 69.37 | 8.7495 | 0.4659 | 0.7149 | 2.5090 |

**表 18 描述**：使用不同秩 $r$ 的 LoRA 在 GPT-2 Medium 上实现 E2E NLG 挑战验证集损失和测试集指标。与 GPT-3 相比，当 $r=1$ 时许多任务已经足够，但在这里验证损失的最佳表现出现在 $r=16$，而 BLEU 的最佳表现为 $r=4$，表明 GPT-2 Medium 的固有秩与 GPT-3 175B 相似。值得注意的是，我们的一些超参数是在 $r=4$ 上调整的，这与另一个基线的参数数量相匹配，因此对于其他选择的 $r$ 可能不是最优的。

> **图 8**：不同 $r$ 值下 $W_q$ 的奇异方向与 $\Delta W_q$ 的奇异方向之间的归一化子空间相似性，以及一个随机基线。$\Delta W_q$ 放大了在 $W$ 中未被强调但重要的方向。较大的 $r$ 值倾向于选择更多已经在 $W$ 中被强调的方向。
>
> *（详见 LoRA 论文原图 8——4 个子空间相似度热图，分别对应 $r=4$、$r=8$、$r=64$ 和随机基线；x 轴为 $j$ (1..1176)，y 轴为 $i$ (1..451)；$\Delta W_q$ 中前 4 个方向与 $W_q$ 中前 10% 方向之间的相似度不超过 0.2，说明 $\Delta W$ 包含 $W$ 中未强调的"任务特定"方向）*
