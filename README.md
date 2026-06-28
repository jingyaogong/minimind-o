<div align="center">

![logo](./images/logo.png)

</div>


<div align="center">

![visitors](https://visitor-badge.laobi.icu/badge?page_id=jingyaogong/minimind-o)
[![GitHub Repo stars](https://img.shields.io/github/stars/jingyaogong/minimind-o?style=social)](https://github.com/jingyaogong/minimind-o/stargazers)
[![GitHub Code License](https://img.shields.io/github/license/jingyaogong/minimind-o?v=1)](LICENSE)
[![GitHub last commit](https://img.shields.io/github/last-commit/jingyaogong/minimind-o)](https://github.com/jingyaogong/minimind-o/commits/master)
[![GitHub pull request](https://img.shields.io/badge/PRs-welcome-blue)](https://github.com/jingyaogong/minimind-o/pulls)
[![Collection](https://img.shields.io/badge/🤗-MiniMind--O%20%20Collection-blue)](https://huggingface.co/collections/jingyaogong/minimind-o)
[![Technical Report](https://img.shields.io/badge/Technical%20Report-arXiv-red)](http://arxiv.org/abs/2605.03937)

</div>

<div align="center">
  <h3>"大道至简"</h3>
</div>

<div align="center">

中文 | [English](./README_en.md)

</div>

* 此开源项目旨在从 0 完整实现一个小规模的端到端 Omni 模型，单一权重同时支持文 / 音 / 图三模态输入与文本 / 流式语音输出。
* 其中 `minimind-3o` 仅 ~0.1B，普通个人 GPU 即可完成训练、CPU即可快速推理，是当前公开模型中规模最小的完整 Omni 实现（或之一）。
* 开源 mini 与 full 两套训练数据：mini 单卡 3090 上约 2 小时跑通完整链路，便于入门；full 与发布权重对应。
* 开源 Omni 模型的完整代码与技术报告，覆盖 Thinker–Talker 双路径、流式语音生成、实时打断、近似双工交互、音色克隆与电话模式 WebUI。
* 所有核心算法代码均从 0 使用 PyTorch 原生实现，不依赖三方框架提供的高层抽象。
* MiniMind-O 进一步延续了 [MiniMind](https://github.com/jingyaogong/minimind)（语言）与 [MiniMind-V](https://github.com/jingyaogong/minimind-v)（视觉多模态）的设计范式。

> 注："约 2 小时" 指 mini 数据集在单张 NVIDIA RTX 3090 上跑完 SFT 的实测耗时。

---

<div align="center">

[📄 MiniMind-O Technical Report](http://arxiv.org/abs/2605.03937)

https://github.com/user-attachments/assets/10cbcc5f-4e70-45cf-bdc5-d6361e40bb86

[🔗 在线体验 (Gradio)](https://modelscope.cn/studios/gongjy/MiniMind-O) &nbsp;|&nbsp; [🔗 视频介绍](https://www.bilibili.com/video/BV1V1RsBcEMX)


</div>

---

# 📌 项目介绍

继 [MiniMind](https://github.com/jingyaogong/minimind)（LLM）和 [MiniMind-V](https://github.com/jingyaogong/minimind-v)（VLM）之后，MiniMind-O 是这个系列的第三站。所谓 Omni，就是让一个模型同时具备听、看、说的多模态交互能力：接收文本、语音和视觉信号，输出文本与流式语音。

或许 GPT-4o 让人第一次感受到足够自然的流式语音交互形态，随后 Mini-Omni2、Moshi、GLM-4-Voice、Qwen3-Omni 等开源工作陆续出现。但如果目标不是直接调用这些参数庞大的现成权重，而是从 0 读懂、训练、改动一个完整 Omni 模型，开源社区仍然急缺足够轻量、链路完整的起点。要把语音真正纳入 Omni 模型，一种做法是把 ASR、LLM、TTS 串成级联链路：语音先转文字，LLM 处理后再合成语音。这条路工程上直接，但中间多了一次文本转写，延迟、语气和情绪信息都会受到影响。

MiniMind-O 尝试补上已知的空位：让语音和文本在 hidden state 层面直接连通，在主 backbone 仅 0.1B 的规模下保留端到端 Omni 链路。Talker 侧采用 MTP（Multi-Token Prediction）一次预测多层 Mimi codes，再配合 VAD 支持实时打断与近似双工交互，这是足够实用的工程路线之一。本项目的代码、模型权重、训练数据和技术报告全部完整开源，单张 RTX 3090 上约 2 小时即可跑通 mini 数据集训练。目标依旧：让每个人都能从第一行代码读起，自己动手，从 0 训练一个能听、能看、能思考、能说的模型：

![](images/omni_io_flow.png)

😊 一起感受创造的乐趣吧！

---

#### 🎉 项目包含以下内容

- 提供完整的 MiniMind-O 结构代码：Thinker、独立 Talker、audio / vision projector、Mimi codebook 接口以及 MTP audio head。
- 提供 SFT 全链路训练流程，覆盖 T2A、I2T、A2A 三类数据，支持全参数训练、音频投影层训练、视觉投影层训练与 DDP 多卡训练。
- 提供 mini 与 full 两套训练数据：mini 便于快速入门，单卡 3090 上约 2 小时可跑通；full 与发布权重对应，覆盖中文语音与图像任务。
- 提供多种内置音色、unseen 音色与任意参考音频的音色克隆能力，便于复现音色控制实验。
- 提供完整的推理与 Demo 工具，支持 CLI 推理、Web UI、流式播放、barge-in 打断和电话模式。
- 关键模块均从 0 用 PyTorch 原生实现，不依赖三方高层封装；同时兼容 `transformers` Tokenizer 与原生权重格式。
- 配套技术报告覆盖架构、训练曲线、CER / WER 评估、音色克隆相似度与跨模型对比，链接见顶部 Tech Report 区。

#### 🎉 已发布模型列表

| 模型 | 参数（主干） | Release |
|---|---|---|
| minimind-3o | ~0.1B | 2026.05.05 |
| minimind-3o-moe | ~0.3B-A0.1B | 2026.05.05 |

---

#### 👉 更新日志

<details close>
<summary> <b>🔥 2026-05-05</b> </summary>

- MiniMind-O 首次开源，发布 `minimind-3o`（115M）与 `minimind-3o-moe`（312M-A115M）
- Thinker–Talker 双路径架构，Talker 采用 MTP 预测多层 Mimi codes，支持 24 kHz 流式语音生成与 barge-in 打断
- 音频编解码器采用 Mimi（8 层 codebook，12.5 Hz，24 kHz），Talker 在 codebook 接口上使用共享主体与轻量 adapter
- 语音 / 视觉特征分别由冻结的 SenseVoice-Small 与 SigLIP2 编码，再通过两层 MLP projector 注入 MiniMind 隐空间
- 同步发布 mini 与 full 两套训练数据，mini 单卡 3090 ~2h 即可跑通整条 Thinker–Talker 链路
- 内置 5 个 voice prompt + 7 个 unseen voice prompt，提供音色克隆与电话模式 WebUI

</details>


# 📌 快速开始

<details style="color:rgb(128,128,128)">
<summary>分享本人的软硬件配置（仅供参考）</summary>

* CPU: Intel(R) Core(TM) i9-10980XE CPU @ 3.00GHz
* RAM: 128 GB
* GPU: NVIDIA GeForce RTX 3090(24GB) * 8
* Ubuntu==20.04
* CUDA==12.2
* Python==3.10
* [requirements.txt](./requirements.txt)

</details>

## 第0步（必须）

### 1' 环境准备

```bash
# 克隆仓库代码
git clone --depth 1 https://github.com/jingyaogong/minimind-o
# 安装必要依赖
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 2' 下载资源

```bash
# 下载 SenseVoice-Small 语音编码器到 ./model/SenseVoiceSmall
modelscope download --model gongjy/SenseVoiceSmall --local_dir ./model/SenseVoiceSmall
# 下载 SigLIP2 视觉编码器到 ./model/siglip2-base-p32-256-ve
modelscope download --model gongjy/siglip2-base-p32-256-ve --local_dir ./model/siglip2-base-p32-256-ve
# 下载 Mimi 音频编解码器到 ./model/mimi
modelscope download --model gongjy/mimi --local_dir ./model/mimi
# 下载 CAMPPlus 说话人编码器到 ./model/campplus
modelscope download --model gongjy/campplus --local_dir ./model/campplus
# 下载 MiniMind 语言模型权重到 ./out 目录下（作为训练 Omni 的基座语言模型）
modelscope download --model gongjy/minimind-3o-pytorch llm_768.pth --local_dir ./out
```

注：也可从 [ModelScope Collection](https://modelscope.cn/collections/gongjy/MiniMind-O) 或 [HuggingFace Collection](https://huggingface.co/collections/jingyaogong/minimind-o) 选择对应内容 `git clone`（需LFS）下载，此处不再赘述。

完成后，结构应如下：

```text
minimind-o/
├── model/
│   ├── SenseVoiceSmall/
│   ├── siglip2-base-p32-256-ve/
│   ├── mimi/
│   ├── campplus/
│   └── ...
├── out/
│   └── llm_768.pth
└── ...
```

## Ⅰ 🚀 模型推理

### 1' 下载发布权重

```bash
# 下载发布权重到 ./out 目录下
modelscope download --model gongjy/minimind-3o-pytorch --local_dir ./out
```

### 2' 命令行问答

```bash
python eval_omni.py --load_from model --weight sft_omni
```

如果使用 transformers 格式模型，可先下载模型目录：

```bash
git clone https://huggingface.co/jingyaogong/minimind-3o
python eval_omni.py --load_from minimind-3o
```

### 3' 启动 WebUI（可选）

```bash
# ⚠️ 须先将 transformers 格式模型文件夹复制到 ./scripts/ 目录下，web_demo_omni 脚本会自动扫描该目录下包含权重文件的子文件夹，如不存在则报错
cp -r minimind-3o ./scripts/minimind-3o
cd scripts && python web_demo_omni.py
```

注：`scripts/web_demo_omni.py` 是 Gradio 非实时演示，上传/录音音频会在后端按实际采样率重采样到 16k；实时语音通话请使用 `webui/web_demo.py`。

## Ⅱ 🛠️ 模型训练

<details style="color:rgb(128,128,128)">
<summary>注：提前测试Torch是否可用cuda</summary>

```python
import torch
print(torch.cuda.is_available())
```

如果不可用，请自行去[torch_stable](https://download.pytorch.org/whl/torch_stable.html)下载whl文件安装。

</details>

### 1' 下载数据

快速开始时，推荐从[数据集链接](https://huggingface.co/datasets/jingyaogong/minimind-o_dataset)只下载 `_mini` 数据集，并放到 `./dataset` 下。

### 2' 开始训练

推荐 mini 训练管线如下，默认在 `trainer/` 目录下执行，可直接 `cd trainer && bash train.sh`：

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --master_port 29560 --nproc_per_node 1 train_sft_omni.py --learning_rate 5e-4 --data_path ../dataset/sft_t2a_mini.parquet --epochs 1 --batch_size 40 --use_compile 1 --from_weight llm --save_weight sft_zero --max_seq_len 512 --use_wandb --use_moe 0
CUDA_VISIBLE_DEVICES=0 torchrun --master_port 29560 --nproc_per_node 1 train_sft_omni.py --learning_rate 5e-4 --data_path ../dataset/sft_a2a_mini.parquet --epochs 1 --batch_size 40 --use_compile 0 --from_weight sft_zero --save_weight sft_zero --max_seq_len 640 --mode audio_proj --use_wandb --use_moe 0
CUDA_VISIBLE_DEVICES=0 torchrun --master_port 29560 --nproc_per_node 1 train_sft_omni.py --learning_rate 2e-5 --data_path ../dataset/sft_a2a_mini.parquet --epochs 1 --batch_size 16 --use_compile 0 --from_weight sft_zero --save_weight sft_zero --max_seq_len 768 --use_wandb --use_moe 0
```

### 3' 测试已训练模型（可选）

确保需要测试的模型 `*.pth` 文件已保存于 `./out/` 目录下。

```bash
python eval_omni.py --weight sft_omni
```

# 📌 模型细节

MiniMind-O 的基座语言模型来自孪生项目 [MiniMind](https://github.com/jingyaogong/minimind)，LLM 的结构与训练细节可移步该项目查阅。即使不了解 LLM 细节，也可直接参照上方"快速开始"流程训练一个 MiniMind-O。

## Ⅰ 架构总览

![](./images/architecture.jpg)

MiniMind-O 的主体由 Thinker 和 Talker 两条路径组成。Thinker 负责理解文本、语音和图像输入，并生成语义层面的文本回复；Talker 则在 Thinker 给出的语义条件上，通过 MTP 同步预测多层 Mimi audio codes，最后由音频解码器还原成流式语音。这样做的目的不是把 ASR、LLM、TTS 简单串起来，而是在一个统一序列里同时保留文本推理、语音输出和流式交互能力。

文本输入直接进入语言主干；语音和图像分别经过 Audio Encoder 与 Vision Encoder 提取特征，再映射到 MiniMind 的隐空间中。音色信息由 Speaker Encoder 或参考音频 codes 提供，推理时可以配合 VAD 实现边听边答、实时打断和近似双工交互。更细的 projector 结构、序列排布和训练目标在后文展开，代码层面的实现细节可直接参考 `model/model_omni.py` 与[技术报告](http://arxiv.org/abs/2605.03937)。

![](./images/input_token_layout.jpg)

图中展示了文本 token、语音特征、图像特征和音色条件在输入序列中的布局方式。

## Ⅱ Thinker 侧多模态理解

Thinker 负责统一接收文本、语音和图像信息，并生成语义层面的文本回复。文本 token 直接进入语言主干，语音和图像特征则通过对应 projector 注入到占位符位置，使不同模态最终落到同一条序列中建模。

## Ⅲ 中间层 Bridge

Thinker 向 Talker 传递的表征取自中间层，而不是 embedding 层或最后一层。embedding 层语义信息不足，最后一层又更贴近 next-token prediction 目标；中间层通常已经融合了上下文和跨模态信息，同时还没有被 LM head 过度塑形，更适合作为语音生成的条件。默认 `bridge_layer = num_hidden_layers // 2 - 1`，不同规模下也可以通过配置调整。

## Ⅳ Talker 侧语音生成

Talker 负责把 Thinker 给出的语义状态转成 8 层 Mimi codebook 序列。这里采用 MTP 形式同时预测多个 audio codebook，而不是把每层 codebook 拆成独立的长链路；为了控制 0.1B 模型中的额外参数量，音频 embedding 和输出 head 采用共享主体加轻量 codebook adapter 的形式。这样既保留不同 codebook 的分布差异，也避免为每一层 codebook 复制一整套参数。

## Ⅴ 序列格式与流式解码

![](./images/sequence_format.jpg)

MiniMind-O 将文本 token 与 8 路 audio-code stream 放在同一个训练样本中：Thinker 负责文本序列，Talker 负责音频 code 序列，语音、图像和音色条件都通过占位符或 reference codes 注入。回复开始之后才计算目标文本和目标音频的损失，因此 reference 与 conditioning 区域只提供条件，不作为重构目标。

流式生成时，模型一边产生文本 token，一边通过 MTP 和延迟调度补齐 8 层 Mimi codes。Mimi 解码器可以增量恢复 24 kHz 波形，因此语音播放不必等待完整回答结束。

## Ⅵ 音色控制

音色控制采用 in-context voice cloning 的方式完成：参考音频先被编码成 voice prompt，作为上下文条件喂给 Talker，而不是通过微调权重或改写文本 prompt 来指定音色。模型也可以同时使用 speaker embedding 提供更稳定的说话人约束；推理时更换音色只需要替换这些条件信息，Thinker prompt 与 Talker 权重保持不变。

默认 release 带有 5 个内置 voice prompt（dylan、eric、serena、uncle_fu、vivian），另保留 7 个 unseen prompt 用于评估（arthur、chelsie、cherry、ethan、jennifer、momo、moon）。

## Ⅶ 模块与参数规模

MiniMind-O 所说的 0.1B，指 Thinker、Talker 和两路 projector 组成的可训练主体；落到具体发布版本上，`minimind-3o` 约 113M，`minimind-3o-moe` 约 315M。Audio Encoder、Vision Encoder 和 Speech Codec 属于冻结的外部旁路模型，负责特征提取或音频编解码，合计约 425M 参数，不计入 active MiniMind-O 参数。

下表按发布模型统计主要模块参数，Trainable 参数按 PyTorch 模块统计，tied embedding 去重计入。

| 统计口径 | minimind-3o | minimind-3o-moe |
|---|---:|---:|
| 可训练主体 | 113.13M | 314.89M |
| 冻结外部模块 | 424.70M | 424.70M |
| 运行时总加载 | 537.83M | 739.59M |

| 模块 | 具体实现 | 关键配置 | 状态 / 参数 (~3o / ~3o-moe) |
|---|---|---|---|
| Thinker | MiniMind Transformer | 8 layers, hidden 768 | trainable, 63.91M / 198.42M |
| Talker | 独立 MiniMind blocks | 4 layers, 8 codebook heads | trainable, 47.05M / 114.30M |
| Audio projector | `MMAudioProjector` | 512 → 768 | trainable, 0.99M |
| Vision projector | `MMVisionProjector` | 768 → 768 | trainable, 1.18M |
| Audio encoder | SenseVoice-Small | 16 kHz speech features | frozen, 234.00M |
| Vision encoder | SigLIP2 base-p32-256 | 256×256 image, 64 tokens | frozen, 94.55M |
| Speech codec | Mimi | 8 codebooks, 12.5 Hz, 24 kHz | frozen, 96.15M |
| Speaker condition | CAM++ embedding | 192-d speaker vector | precomputed |

# 📌 实验

## Ⅰ 数据集

数据集下载：[ModelScope](https://www.modelscope.cn/datasets/gongjy/minimind-o_dataset) | [HuggingFace](https://huggingface.co/datasets/jingyaogong/minimind-o_dataset)

所有语音数据都统一转成 Mimi codes 存储，8 层 codebook，帧率 12.5 Hz；图像统一 resize 到 256×256，由 SigLIP2 P32 编码为 64 个 patch token。训练数据主要来自公开 omni / speech instruction 数据，包括 [VoiceAssistant-400K](https://huggingface.co/datasets/gpt-omni/VoiceAssistant-400K)、[UltraChat-300K-SLAM-Omni](https://huggingface.co/datasets/worstchan/UltraChat-300K-SLAM-Omni) 等；同时基于 Qwen3-TTS 进行了大量多说话人音频合成，并用 CAM++ 提取 speaker embedding 作为音色条件。I2T 数据与 [MiniMind-V](https://github.com/jingyaogong/minimind-v) 使用的视觉指令数据来源一致，原始组成和引用可参考该项目说明。

仓库提供 **mini** 与 **full** 两套训练数据。mini 从 full 中按"英文 + 无视觉"筛出，配 `train_sft_omni.py` 的默认 `--data_path` 即可使用；它的目标是用较低成本跑通 Thinker–Talker、Mimi 编解码、序列布局和音色注入链路，而不是复现发布模型的中文语音能力。中文 Talker 要同时处理更复杂的字音映射、韵律停顿和多说话人稳定性，明显比英文更难，不能依赖单卡 3090 约 2 小时的 mini 训练完成。

full 数据集与发布的 `minimind-3o` / `minimind-3o-moe` 权重对应，覆盖中英文 T2A / A2A 与图像 I2T。规模与中英文比例见下表，是论文中 CER / 音色相似度等指标的实际训练源。

其中 T2A 表示 Text-to-Audio，A2A 表示 Audio-to-Audio，I2T 表示 Image-to-Text。

| 数据集 | 子集 | 输入语音 | 输出语音 | 备注 |
|---|---|---|---|---|
| `sft_t2a_mini` | 英文 T2A | — | 约 470.14 h | mini 入门用 |
| `sft_a2a_mini` | 英文 A2A | 约 74.64 h | 约 56.60 h | mini 入门用 |
| `sft_t2a` | 中英 T2A | — | 约 1636.01 h | full 训练 |
| `sft_a2a` | 中英 A2A | 约 1711.97 h | 约 423.40 h | full 训练 |
| `sft_i2t` | 图像 I2T | — | — | full 训练 |

`sft_t2a` 中中文、英文、混合样本占比分别为 45.7%、46.5%、7.8%；`sft_a2a` 中三者分别为 70.8%、21.2%、8.0%。这个分布会直接反映到行为上：短中文和短英文回答通常较稳定，较长英文语音更容易出现读音漂移和漏词。mini 子集只保留英文，因此即便参数量和数据量都收得很紧，单语种内部的 CER 表现仍能维持在可用范围。

## Ⅱ 训练

训练入口是 `train_sft_omni.py`，推荐流程可直接参考 `trainer/train.sh`。当前 full 训练不拆复杂的多阶段预训练，而是按数据流逐步接入能力：

![](./images/training_pipeline.jpg)

- `sft_t2a`：先对齐文本到语音输出，让 Talker 学会在 Thinker 语义条件下生成 Mimi codes；
- `sft_a2a`：再接入语音输入，使模型从 speech instruction 进入同一套 Thinker–Talker 回复链路；
- `sft_i2t`：最后对齐视觉路径，其中 `vision_proj` 模式只更新视觉投影层，避免图像数据过度改写语言和语音能力。

训练模式里，`all` 会更新 MiniMind / Talker / projector，`audio_proj` 和 `vision_proj` 只用于单独对齐对应投影层；SenseVoice-Small、SigLIP2 和 Mimi 始终冻结。Dense 与 MoE 版本沿用同一套数据顺序。mini 命令只用于快速跑通链路，默认单卡 3090 约 2 小时完成；发布权重对应 full 数据训练。

下面给出 full 训练过程中的 T2A 与 A2A loss 曲线（仅供参考）：

![](./images/t2a_training_curves.jpg)

> `sft_t2a`：文本到语音输出链路

![](./images/a2a_training_curves.jpg)

> `sft_a2a`：接入语音输入后的 loss

T2A 曲线已去掉早期不兼容权重 resume 造成的异常尖峰；MoE 总参数更多但 active 参数与 dense 接近，更适合作为容量分配实验参考。

## Ⅲ 模型权重

| 模型格式 | ModelScope | HuggingFace |
|---|---|---|
| PyTorch (`*.pth`) | [minimind-3o-pytorch](https://www.modelscope.cn/models/gongjy/minimind-3o-pytorch) | [minimind-3o-pytorch](https://huggingface.co/jingyaogong/minimind-3o-pytorch) |
| Transformers | [minimind-o collection](https://modelscope.cn/collections/gongjy/MiniMind-O) | [minimind-o collection](https://huggingface.co/collections/jingyaogong/minimind-o) |

> Transformers 版本包含 `minimind-3o` 与 `minimind-3o-moe`，适合直接用于 `eval_omni.py` 和 WebUI 推理；原生 PyTorch 权重主要用于训练、复现实验和继续微调。

# 📌 评估

Omni 模型目前可能还没有统一的评估口径，不同工作的 LLM 主干、音频合成器和系统目标都不一样：有的看重 LLM 本身的知识和推理，会报告 MMLU、HumanEval 等指标；有的看重流式速度和音质，有的强调语音一致性指标，也有的更关注自然交互或更大范围的 Omni 生成。这些工作大都基于 SOTA 开源 LLM 续训，而 MiniMind 的 0.06B 主干在复杂知识问答、数学推理、代码生成或开放式长回答上显然不可能形成竞争力，Talker 的自然度、韵律和稳定性也弱于成规模的系统。

所以这里无法追求综合榜单，而是落实到几项更可复现的局部评估和 use cases：Talker hidden size 消融、音色克隆相似度、相同问题和相同 ASR 流程下的 CER / WER 对比，以及 A2A、I2A 和实时交互样例。CER / WER 主要用来观察文本一致性，音质、自然度和人类偏好则留给定性样例和实际试听判断。

## Ⅰ Talker Hidden Size 消融

如果只看语音生成，Talker 做到 1024 / 2048 维、或者继续加深层数一定会更稳。但 MiniMind-O 要把完整 Omni 链路压在 0.1B 左右，不能把大部分参数都交给声学端。Thinker / Talker 解耦后，语言理解和跨模态融合主要由 Thinker 承担，Talker 只在语义条件上渲染 Mimi codes，这让小 Talker 成为可能。这里的渲染不是只预测语义 token 再交给外部声学生成器，而是由 Talker 直接生成可解码的 Mimi acoustic codes；真正的瓶颈也就在输出端：Talker 面对的是 8 层 Mimi codebook，而不是单一路径的 next-token prediction。

384 维最诱人，dense 版本可以压到 88M 左右；512 维也更轻。但表中结果说明，小并不自动等于划算：短句还能维持，中长句更容易出现漏词、重复和发音漂移。768 维最后留下来，是因为它和 MiniMind 主干维度一致，可以用 Thinker 后 4 层初始化；参数仍在 0.1B 左右，训练成本没有明显增加，一致性却稳定得多。

| Variant | Talker hidden | Params | Avg CER ↓ | Short ↓ | Mid / Long ↓ |
|---|---|---|---|---|---|
| Dense | 768 | 115.29M | **0.0897** | 0.1528 | 0.0874 / 0.0675 |
| Dense | 512 | 96.13M | 0.1745 | 0.2709 | 0.2455 / 0.0976 |
| Dense | 384 | 88.72M | 0.2767 | 0.3904 | 0.1865 / 0.4046 |
| MoE | 768 | 317.05M-A115.33M | **0.0900** | 0.2075 | 0.0533 / 0.0271 |
| MoE | 512 | 261.32M-A96.17M | 0.1265 | 0.0711 | 0.1490 / 0.1464 |
| MoE | 384 | 240.04M-A88.75M | 0.3280 | 0.3757 | 0.2777 / 0.4313 |

Dense 和 MoE 的 CER 不宜直接横向比较：同一问题下，两个 Thinker 生成的内容和长度可能不同，Talker 面对的合成难度也不同。更有意义的是看同一架构内部的趋势，768 都明显优于 512 和 384。

## Ⅱ 音色克隆相似度

音色克隆是当前版本里比较 Beta 的能力。没说错的话，多数开源 Omni 模型只支持固定输出音色，而 minimind-3o 尝试把多音色生成塞到同一套 Talker 里完成。这个目标比“能说话”更难，因为模型不仅要把内容说对，还要在生成 Mimi codes 时保留说话人的音色线索。

目前效果还谈不上高保真克隆，同一个参考音色在不同问题上并不总能保持一致，长句里也容易被发音和节奏问题带偏。但基本的男女声差异、语调倾向和一部分韵律特征是能区分出来的。
下面的 CAM++ speaker embedding 余弦相似度只作为自动化参考：Seen 组来自 `voices.pt` 中 5 个内置音色，Unseen 组来自 `voices_unseen.pt` 中 7 个训练时未见过的音色；每个音色使用同一组文本问题，只替换音色条件。

逐音色细分如下：

| Split | Speaker | Dense ↑ | MoE ↑ |
|---|---|---|---|
| Seen | dylan | 0.6997 | 0.6837 |
| Seen | eric | 0.5289 | 0.4232 |
| Seen | serena | 0.7092 | 0.7041 |
| Seen | uncle_fu | 0.7241 | 0.7337 |
| Seen | vivian | 0.5744 | 0.5888 |
| Unseen | arthur | 0.7171 | 0.6750 |
| Unseen | chelsie | 0.6437 | 0.6240 |
| Unseen | cherry | 0.5689 | 0.5678 |
| Unseen | ethan | 0.4783 | 0.4847 |
| Unseen | jennifer | 0.4749 | 0.4003 |
| Unseen | momo | 0.6470 | 0.5720 |
| Unseen | moon | 0.4282 | 0.6673 |

总体上，minimind-3o 与 minimind-3o-moe 的平均结果接近，也都略高于早期 baseline；这说明音色保持不主要由 inactive expert 容量决定，更直接的影响来自 reference 片段质量、CAM++ embedding 的可分性，以及 Talker 生成音频本身是否稳定。单个音色里，uncle_fu、serena、arthur 这类声音更容易保持住，至少一个版本能超过 0.70；eric、moon 等 outlier 则更容易受生成质量影响。换句话说，这个能力已经能区分一部分音色特征，但距离“给一段参考音频就稳定复刻”的产品级体验还有距离。

### 音色克隆消融实验（音频播放）

为了让试听更直观，这里固定 seed=42、temperature=0.7，对每个音色展示 1 个生成样例，唯一变化的是参考音频 codes 和 speaker embedding。作为对照，下面先给出不施加任何参考音色条件时的 default 输出（朗读文本对所有样例一致）：

https://github.com/user-attachments/assets/b31fd8f2-e3af-4fed-ba19-65424b59bec6

#### Seen 音色
Seen 表示训练数据中出现过的音色，用来观察模型对熟悉说话人的保持情况。

<table>
<tr><th width="100">说话人</th><th width="380">参考音色</th><th width="380">输出结果</th><th width="80">平均</th></tr>
<tr><td>dylan</td><td>

https://github.com/user-attachments/assets/070ea3ab-0e8e-4aa0-84b5-af8d3c4e2725

</td><td>

https://github.com/user-attachments/assets/eb2da7ed-173c-47e9-9431-7bdb5a9b7385

</td><td>0.6712</td></tr>
<tr><td>eric</td><td>

https://github.com/user-attachments/assets/c74aa5dc-1edd-44c1-9546-6e57194c2f60

</td><td>

https://github.com/user-attachments/assets/f3fa8906-4e14-4610-a9d9-c16c915ca1b3

</td><td>0.4430</td></tr>
<tr><td>serena</td><td>

https://github.com/user-attachments/assets/0eeeac87-fa70-4025-b66e-1f0197f2b434

</td><td>

https://github.com/user-attachments/assets/c5901dca-4b2a-47f5-9b30-c89de54f908e

</td><td>0.6600</td></tr>
<tr><td>uncle_fu</td><td>

https://github.com/user-attachments/assets/fdd1bb28-6648-44bf-8bcb-4509e709e347

</td><td>

https://github.com/user-attachments/assets/95b480f1-f015-4712-8d7c-17db465f6584

</td><td>0.6632</td></tr>
<tr><td>vivian</td><td>

https://github.com/user-attachments/assets/f64731c4-67a3-4e18-b7d7-61bf44ef4bdd

</td><td>

https://github.com/user-attachments/assets/3f1cc9bb-16d2-4ce0-a473-40676cf4523e

</td><td>0.5320</td></tr>
</table>

#### Unseen 音色
Unseen 表示训练时没有见过的音色，用来观察模型能否把新的参考音色0样本直接迁移到生成语音里。

<table>
<tr><th width="100">说话人</th><th width="380">参考音色</th><th width="380">输出结果</th><th width="80">平均</th></tr>
<tr><td>arthur</td><td>

https://github.com/user-attachments/assets/3430ecdb-6de8-4fb0-a6a7-ad82bdce01a1

</td><td>

https://github.com/user-attachments/assets/e598dbc2-ba28-4c38-b52d-6fa6c2349a5b

</td><td>0.6479</td></tr>
<tr><td>chelsie</td><td>

https://github.com/user-attachments/assets/f9166af6-3a98-42f3-9cf8-ad105eea87d6

</td><td>

https://github.com/user-attachments/assets/eccca693-4708-409a-88f7-85eb25f66fe6

</td><td>0.5975</td></tr>
<tr><td>cherry</td><td>


https://github.com/user-attachments/assets/e69b9cac-e12f-43ae-a9dc-7e1618ef3a43


</td><td>


https://github.com/user-attachments/assets/bb41cdef-cc92-48fa-a508-76a75d391565


</td><td>0.5418</td></tr>
<tr><td>ethan</td><td>

https://github.com/user-attachments/assets/9c992505-2046-483e-a7cf-50ec18a5e329

</td><td>

https://github.com/user-attachments/assets/98013c5e-f5b5-4e1a-bc0e-a0f0be5d3240

</td><td>0.4323</td></tr>
<tr><td>jennifer</td><td>

https://github.com/user-attachments/assets/924b035d-5c7c-45a5-a8f8-5dbdc18f71db

</td><td>

https://github.com/user-attachments/assets/853d1370-0065-4567-9a71-dc88a6a34d56

</td><td>0.4052</td></tr>
<tr><td>momo</td><td>

https://github.com/user-attachments/assets/7e97f524-da6d-4a2f-9095-e7f99262f4a5

</td><td>

https://github.com/user-attachments/assets/4c193c8f-8750-4424-acba-2bd13089a634

</td><td>0.5968</td></tr>
<tr><td>moon</td><td>

https://github.com/user-attachments/assets/527df88a-adc0-48d3-9a6a-827ca1ba7fb0

</td><td>

https://github.com/user-attachments/assets/3f533e26-1ad8-4ab3-baf1-21267734d3ee

</td><td>0.5874</td></tr>
</table>

## Ⅲ 跨模型英文 T2A 对比

这里选了 20 个英文问题，并统一加上 `Answer briefly in one short sentence` 约束。这样做不是为了考察开放式英文能力，而是尽量把回答长度压到同一范围内；三套模型生成音频后，再统一用 Qwen3-ASR 转写，并与目标文本计算 CER / WER，用来比较 Talker 的文本一致性。

| 长度桶 | [Mini-Omni](https://huggingface.co/gpt-omni/mini-omni) CER/WER | [Mini-Omni2](https://huggingface.co/gpt-omni/mini-omni2) CER/WER | minimind-3o CER/WER |
|---|---|---|---|
| short (≤15w) | 0.0195 / 0.0384 (n=8) | 0.0503 / 0.0584 (n=14) | 0.0531 / 0.0417 (n=8) |
| mid (16–30w) | 0.0038 / 0.0052 (n=12) | 0.0062 / 0.0076 (n=6) | 0.1327 / 0.1420 (n=11) |
| long (31–60w) | — | — | 0.0431 / 0.0508 (n=1) |

≤15 词的短回复里，minimind-3o 已经接近 Mini-Omni2；真正拉开差距的是 16–30 词段。这个长度已经不是简单短语，Talker 需要在一个完整短句里同时维持发音、节奏和词面一致性，也是当前 0.1B Talker 最容易暴露不稳定性的区间。

## Ⅳ 跨模型视觉语言对比

[Mini-Omni](https://huggingface.co/gpt-omni/mini-omni) 不支持 VL 路径，因此这里只比较 [Mini-Omni2](https://huggingface.co/gpt-omni/mini-omni2)（0.5B）和 minimind-3o（0.1B）。9 张合成图像上，两个模型分别生成英文回答，再统一转写并计算 CER / WER，作为视觉到语音链路的一致性参考。

| Model | Params | Avg CER ↓ | Avg WER ↓ |
|---|---|---|---|
| [Mini-Omni2](https://huggingface.co/gpt-omni/mini-omni2) | 0.5B | 0.7609 | 0.9756 |
| minimind-3o | 0.1B | 0.8241 | 1.0293 |

这个数值不能当作开放式图像描述的绝对正确率。视觉描述存在大量等价表达，同义改写和描述顺序都会影响 CER / WER，数值整体偏高是预期现象。在同一自动流程下，minimind-3o 落后于 Mini-Omni2，但仍处在同一数量级，同时参数约为后者的 1/5。

## Ⅴ 样例

![](./images/qual_a2a.jpg)

语音到语音样例直接以真实语音作为输入，由 Thinker 组织语义，再由 Talker 渲染成语音。短回答仍然是当前更稳的区间，中文解释型问题通常能生成较连贯的回答，英文的发音和节奏相对更稳定。

<table>
<tr>
<td>

https://github.com/user-attachments/assets/c85809b2-4787-4656-9c7e-55b693798494

</td>
<td>

https://github.com/user-attachments/assets/354a5eec-c147-4d18-8c7a-942bd2a0b4b0

</td>
</tr>
</table>

![](./images/image2audio_qualitative.jpg)

图像问答样例把视觉编码、文本生成和语音渲染串在同一条链路里。当前模型通常能抓住主体物体和大致场景，但细粒度空间关系、数量和属性仍容易出错，因此更适合作为小模型 omni pipeline 的可复现基线。

<table>
<tr>
<td>

https://github.com/user-attachments/assets/244e08b0-5b12-449e-a7a2-2a2139c5d62d

</td>
<td>

https://github.com/user-attachments/assets/3e8d0a76-282d-4a9d-9726-a954cf80198a

</td>
</tr>
</table>

## Ⅵ 实时交互

![](./images/realtime_interaction.jpg)

最后是实时交互界面。用户停止说话后，Thinker 先完成语义侧的 prefill，Talker 随后开始逐步产生音频 code，Mimi decoder 则边接收 code 边写出 24 kHz 波形。Barge-in 的例子展示了另一条更接近真实对话的路径：当用户在模型说话过程中再次开口，系统会中断当前生成，重新进入 prefill–reply 流程。这里的中断检测仍然只是简单 VAD 阈值，还谈不上语义级打断；但从工程闭环看，系统已经能从 speaking 状态退回 listening 状态，并处理下一轮输入。

### 🧩 未来值得改进的方面

当前模型和大规模 Omni 系统在各方面仍有差距，这一点不需要回避。长语音自然度、复杂视觉推理、开放式英文中长回答和音色稳定性，都还不是它擅长的区间。视觉路径更接近紧凑的 vision-to-speech 链路，MoE 版本也更像一次容量分配实验，而不是同算力最优解。

这些限制也给出了后续方向：更长的 ICL 上下文、更细的 prosody 监督、更强的视觉编码器、更稳定的音色条件，以及对 Bridge layer 和 MTP codebook interface 的系统扫描，都值得继续做。

话说回来，MiniMind-O 的价值也正在这里。它把一个完整 Omni 闭环压到 0.1B 量级，并把代码、权重和主要训练数据放在同一个可检查对象里；这意味着它不只是一个 demo，而是一个足够小、足够透明、可以从头复现和继续改造的基线。对于想理解 Thinker–Talker 解耦、MTP codebook interface、in-context voice cloning 和 middle hidden bridge 这些细节的人来说，它提供的是一套可以真正动手验证的设计经验。

# 📌 致谢

> [!TIP]
> 如果您觉得 `MiniMind-O` 对您有所帮助，可以在 GitHub 上加一个⭐<br/>
> 水平有限难免存在未知的纰漏，欢迎所有人在 Issues 交流指正或提交 PR 改进项目<br/>
> 您的支持就是持续改进项目的动力，谢谢！

## 🤝贡献者

<a href="https://github.com/jingyaogong/minimind-o/graphs/contributors">
  <img width="200" src="https://contrib.rocks/image?repo=jingyaogong/minimind-o" />
</a>

## 😊鸣谢

- [MiniMind](https://github.com/jingyaogong/minimind) / [MiniMind-V](https://github.com/jingyaogong/minimind-v)（基座、数据）
- [Qwen2.5-Omni / Qwen3-Omni](https://github.com/QwenLM/Qwen2.5-Omni)（灵感、数据）
- [Mini-Omni / Mini-Omni2](https://github.com/gpt-omni/mini-omni)（灵感、数据）
- [SLAM-Omni](https://aclanthology.org/2025.findings-acl.115/)（数据）
- [SenseVoice](https://arxiv.org/abs/2407.04051)（组件）
- [Mimi / Moshi](https://arxiv.org/abs/2410.00037)（组件）
- [vLLM-Omni](https://github.com/vllm-project/vllm-omni)（推理、合成数据）
- 其他参考的开源项目与论文（在技术报告中详细列出）

## 🫶支持者

<a href="https://github.com/jingyaogong/minimind-o/stargazers">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://bytecrank.com/nastyox/reporoster/php/stargazersSVG.php?user=jingyaogong&repo=minimind-o&theme=dark"/>
      <source media="(prefers-color-scheme: light)" srcset="https://bytecrank.com/nastyox/reporoster/php/stargazersSVG.php?user=jingyaogong&repo=minimind-o"/>
      <img alt="github contribution grid snake animation" src="https://bytecrank.com/nastyox/reporoster/php/stargazersSVG.php?user=jingyaogong&repo=minimind-o&theme=dark"/>
    </picture>
</a>

<a href="https://github.com/jingyaogong/minimind-o/network/members">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://bytecrank.com/nastyox/reporoster/php/forkersSVG.php?user=jingyaogong&repo=minimind-o&theme=dark"/>
      <source media="(prefers-color-scheme: light)" srcset="https://bytecrank.com/nastyox/reporoster/php/forkersSVG.php?user=jingyaogong&repo=minimind-o"/>
      <img alt="github contribution grid snake animation" src="https://bytecrank.com/nastyox/reporoster/php/forkersSVG.php?user=jingyaogong&repo=minimind-o&theme=dark"/>
    </picture>
</a>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=jingyaogong/minimind-o&type=Date&theme=dark"/>
  <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=jingyaogong/minimind-o&type=Date"/>
  <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=jingyaogong/minimind-o&type=Date&theme=dark"/>
</picture>

# 🎓 引用

如果您觉得 MiniMind-O 对您的研究或工作有所帮助，请引用：

```bibtex
% 引用 MiniMind-O 技术报告：用于讨论模型架构、训练方法与实验结论。
@article{minimind-o-report,
    title   = {MiniMind-O Technical Report: An Open Small-Scale Speech-Native Omni Model}, 
    author  = {Jingyao Gong},
    journal = {arXiv preprint arXiv:2605.03937},
    year    = {2026}
}

% 引用 MiniMind-O GitHub 仓库：用于指代开源代码与发布权重。
@misc{minimind-o,
    title  = {MiniMind-O: Train a Tiny Omni Model from Scratch},
    author = {Jingyao Gong},
    year   = {2026},
    url    = {https://github.com/jingyaogong/minimind-o},
    note   = {GitHub repository, accessed 2026}
}
```

# 📜 许可协议

本仓库遵循 [Apache-2.0 License](LICENSE) 开源协议。
