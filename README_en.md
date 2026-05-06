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
  <h3>"Less is More"</h3>
</div>

<div align="center">

[中文](./README.md) | English

</div>

* This project implements a small end-to-end Omni model from scratch, where a single set of weights jointly handles text / audio / image inputs and produces text / streaming-speech outputs.
* `minimind-3o` has only ~0.1B parameters: it can be trained on a consumer GPU and runs quickly on CPU, making it one of the smallest fully-functional Omni implementations publicly available.
* Two training datasets are released, `mini` and `full`. `mini` runs the full pipeline in about 2 hours on a single RTX 3090 and is intended for getting started; `full` corresponds to the released weights.
* The full codebase and technical report are released, covering the Thinker–Talker dual-path architecture, streaming speech generation, real-time barge-in, near-duplex interaction, voice cloning and a phone-mode WebUI.
* All core algorithmic components are implemented from scratch in native PyTorch and do not rely on high-level abstractions from third-party frameworks.
* MiniMind-O continues the design philosophy of [MiniMind](https://github.com/jingyaogong/minimind) (language) and [MiniMind-V](https://github.com/jingyaogong/minimind-v) (vision-language).

> Note: "about 2 hours" refers to the measured time of running SFT on the mini dataset using a single NVIDIA RTX 3090.

---

<div align="center">

[📄 MiniMind-O Technical Report](http://arxiv.org/abs/2605.03937)

https://github.com/user-attachments/assets/10cbcc5f-4e70-45cf-bdc5-d6361e40bb86

[🔗 Online Demo](https://modelscope.cn/studios/gongjy/MiniMind-O) &nbsp;|&nbsp; [🔗 Video Intro](https://www.bilibili.com/video/BV121421h71e)

</div>

---

# 📌 Project Introduction

After [MiniMind](https://github.com/jingyaogong/minimind) (LLM) and [MiniMind-V](https://github.com/jingyaogong/minimind-v) (VLM), MiniMind-O is the third stop in this series. By "Omni" we mean a model that can listen, see and speak at the same time: it takes text, speech and visual signals as inputs, and produces text together with streaming speech.

GPT-4o was probably the first system that made natural streaming voice interaction feel real. Since then, open-source projects such as Mini-Omni2, Moshi, GLM-4-Voice and Qwen3-Omni have gradually appeared. However, if the goal is not just to call ready-made checkpoints with billions of parameters, but to fully understand, train and modify a complete Omni model from scratch, the open-source community still lacks a sufficiently lightweight starting point with an end-to-end pipeline. A common way to bring speech into an Omni model is to chain ASR, LLM and TTS into a cascade: speech is first transcribed to text, the LLM processes it, and the answer is then synthesized back to speech. This is straightforward from an engineering perspective, but it adds an extra transcription step and noticeably hurts latency, prosody and emotional cues.

MiniMind-O attempts to fill this gap: speech and text are connected directly at the hidden-state level, while the trainable backbone remains only ~0.1B parameters and the end-to-end Omni pipeline is preserved. The Talker side adopts MTP (Multi-Token Prediction) to predict multiple Mimi codebook layers at once, and combines it with VAD to support real-time barge-in and near-duplex interaction—a practical engineering route for a tiny Omni model. The code, model weights, training data and technical report are all open-sourced. A single RTX 3090 can finish training on the mini dataset in about 2 hours. The goal remains the same: let everyone read the project from the first line of code, and train, from scratch, a model that can listen, see, think and speak:

![](images/omni_io_flow.png)

😊 Enjoy building.

---

#### 🎉 What this project provides

- A complete MiniMind-O architecture: Thinker, an independent Talker, audio / vision projectors, the Mimi codebook interface and the MTP audio head.
- A full SFT pipeline that covers T2A, I2T and A2A data, supporting full-parameter training, audio-projector-only training, vision-projector-only training, and DDP multi-GPU training.
- Two training datasets, `mini` and `full`. `mini` is meant for quick onboarding and runs the pipeline in ~2 hours on a single RTX 3090; `full` matches the released weights and covers Chinese speech and image tasks.
- Multiple built-in voice prompts, unseen voice prompts and voice cloning from arbitrary reference audio, making voice-control experiments easy to reproduce.
- A complete inference and demo toolkit: CLI, Web UI, streaming playback, barge-in interruption and a phone-mode demo.
- Key modules are written from scratch in native PyTorch without high-level third-party wrappers, while remaining compatible with `transformers` tokenizers and native weight formats.
- A companion technical report covers architecture, training curves, CER / WER evaluation, voice-cloning similarity and cross-model comparisons. See the Tech Report badge at the top.

#### 🎉 Released models

| Model | Backbone params | Release |
|---|---|---|
| minimind-3o | ~0.1B | 2026.05.05 |
| minimind-3o-moe | ~0.3B-A0.1B | 2026.05.05 |

---

#### 👉 Update Log

<details close>
<summary> <b>🔥 2026-05-05</b> </summary>

- First release of MiniMind-O: `minimind-3o` (115M) and `minimind-3o-moe` (312M-A115M).
- Thinker–Talker dual-path architecture. Talker uses MTP to predict multi-codebook Mimi codes and supports 24 kHz streaming speech generation and barge-in.
- Audio codec is Mimi (8 codebooks, 12.5 Hz, 24 kHz). Talker uses a shared backbone plus lightweight adapters at the codebook interface.
- Speech and visual features are extracted by frozen SenseVoice-Small and SigLIP2 respectively, and injected into the MiniMind hidden space through two-layer MLP projectors.
- Mini and full training datasets are released alongside; mini runs the full Thinker–Talker pipeline in ~2h on a single RTX 3090.
- 5 built-in voice prompts and 7 unseen voice prompts, with voice cloning and a phone-mode WebUI included.

</details>


# 📌 Quick Start

<details style="color:rgb(128,128,128)">
<summary>Reference hardware / software setup</summary>

* CPU: Intel(R) Core(TM) i9-10980XE CPU @ 3.00GHz
* RAM: 128 GB
* GPU: NVIDIA GeForce RTX 3090 (24GB) * 8
* Ubuntu==20.04
* CUDA==12.2
* Python==3.10
* [requirements.txt](./requirements.txt)

</details>

## Step 0 (required)

### 1' Environment

```bash
# Clone the repository
git clone --depth 1 https://github.com/jingyaogong/minimind-o
# Install dependencies
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 2' Download resources

```bash
# Download SenseVoice-Small audio encoder to ./model/SenseVoiceSmall
modelscope download --model gongjy/SenseVoiceSmall --local_dir ./model/SenseVoiceSmall
# Download SigLIP2 vision encoder to ./model/siglip2-base-p32-256-ve
modelscope download --model gongjy/siglip2-base-p32-256-ve --local_dir ./model/siglip2-base-p32-256-ve
# Download Mimi audio codec to ./model/mimi
modelscope download --model gongjy/mimi --local_dir ./model/mimi
# Download CAM++ speaker encoder to ./model/campplus
modelscope download --model gongjy/campplus --local_dir ./model/campplus
# Download MiniMind LLM weights to ./out (used as the language backbone for training Omni)
modelscope download --model gongjy/minimind-3o-pytorch llm_768.pth --local_dir ./out
```

You can also `git clone` the corresponding repos from the [ModelScope Collection](https://modelscope.cn/collections/gongjy/MiniMind-O) or [HuggingFace Collection](https://huggingface.co/collections/jingyaogong/minimind-o) (LFS required); details omitted here.

After downloading, the directory should look like:

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

## Ⅰ 🚀 Inference

### 1' Download released weights

```bash
# Download released weights to ./out
modelscope download --model gongjy/minimind-3o-pytorch --local_dir ./out
```

### 2' Command-line chat

```bash
python eval_omni.py --load_from model --weight sft_omni
```

To use the Transformers-format model, download the model directory first:

```bash
git clone https://huggingface.co/jingyaogong/minimind-3o
python eval_omni.py --load_from minimind-3o
```

### 3' Launch WebUI (optional)

```bash
# ⚠️ Copy the Transformers-format model folder into ./scripts/. The web_demo_omni script
#    automatically scans this directory for sub-folders that contain weight files; it
#    raises an error if none is found.
cp -r minimind-3o ./scripts/minimind-3o
cd scripts && python web_demo_omni.py
```

## Ⅱ 🛠️ Training

<details style="color:rgb(128,128,128)">
<summary>Verify Torch is using CUDA</summary>

```python
import torch
print(torch.cuda.is_available())
```

If unavailable, please download the matching `.whl` from [torch_stable](https://download.pytorch.org/whl/torch_stable.html) and install it manually.

</details>

### 1' Download data

For a quick start, downloading only the `_mini` parquet files from the [dataset link](https://huggingface.co/datasets/jingyaogong/minimind-o_dataset) and placing them under `./dataset` is enough.

### 2' Train

The recommended mini training pipeline is shown below. It is meant to be run from the `trainer/` directory; equivalently, run `cd trainer && bash train.sh`:

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --master_port 29560 --nproc_per_node 1 train_sft_omni.py --learning_rate 5e-4 --data_path ../dataset/sft_t2a_mini.parquet --epochs 1 --batch_size 40 --use_compile 1 --from_weight llm --save_weight sft_zero --max_seq_len 512 --use_wandb --use_moe 0
CUDA_VISIBLE_DEVICES=0 torchrun --master_port 29560 --nproc_per_node 1 train_sft_omni.py --learning_rate 5e-4 --data_path ../dataset/sft_a2a_mini.parquet --epochs 1 --batch_size 40 --use_compile 0 --from_weight sft_zero --save_weight sft_zero --max_seq_len 640 --mode audio_proj --use_wandb --use_moe 0
CUDA_VISIBLE_DEVICES=0 torchrun --master_port 29560 --nproc_per_node 1 train_sft_omni.py --learning_rate 2e-5 --data_path ../dataset/sft_a2a_mini.parquet --epochs 1 --batch_size 16 --use_compile 0 --from_weight sft_zero --save_weight sft_zero --max_seq_len 768 --use_wandb --use_moe 0
```

### 3' Test the trained model (optional)

Make sure the model `*.pth` to be tested is placed under `./out/`.

```bash
python eval_omni.py --weight sft_omni
```

# 📌 Model Details

The language backbone of MiniMind-O comes from the sister project [MiniMind](https://github.com/jingyaogong/minimind). For LLM architecture and training details, please refer to that repository. Even without going into the LLM internals, you can still follow the Quick Start section above to train MiniMind-O end-to-end.

## Ⅰ Architecture overview

![](./images/architecture.jpg)

MiniMind-O consists of two paths: Thinker and Talker. Thinker is responsible for understanding text, speech and image inputs and producing a semantic-level text reply. Talker takes the semantic conditions from Thinker and uses MTP to jointly predict multi-codebook Mimi audio codes, which the audio decoder finally restores into streaming speech. The point is not to chain ASR, LLM and TTS together, but to keep text reasoning, speech generation and streaming interaction inside a single unified sequence.

Text inputs go directly into the language backbone; speech and images are first encoded by the Audio Encoder and Vision Encoder respectively, and then projected into the MiniMind hidden space. Voice information is provided either by a Speaker Encoder or by reference-audio codes; combined with VAD at inference time, this enables listen-while-speaking, real-time barge-in and near-duplex interaction. Later sections describe the projectors, sequence layout and training objectives in more detail; for code-level details, please refer to `model/model_omni.py` and the [technical report](http://arxiv.org/abs/2605.03937).

![](./images/input_token_layout.jpg)

The figure above shows how text tokens, speech features, image features and voice conditions are laid out in the input sequence.

## Ⅱ Multimodal understanding on the Thinker side

Thinker receives text, speech and image information uniformly and produces a semantic-level text reply. Text tokens enter the language backbone directly, while speech and image features are injected into placeholder positions through their respective projectors, so that all modalities are eventually modeled within the same sequence.

## Ⅲ Middle-layer Bridge

The representation passed from Thinker to Talker is taken from a middle layer rather than the embedding layer or the final layer. Embedding layers carry too little semantic information, while the final layer is overly shaped towards next-token prediction. A middle layer typically already fuses contextual and cross-modal information without being over-tuned by the LM head, which makes it a better conditioning source for speech generation. By default `bridge_layer = num_hidden_layers // 2 - 1`, and it can be adjusted through configuration at different scales.

## Ⅳ Speech generation on the Talker side

Talker turns the semantic states from Thinker into 8 streams of Mimi codebook codes. It uses MTP to predict multiple audio codebooks simultaneously, instead of running each codebook through a separate long path. To control the additional parameter count inside a 0.1B model, the audio embedding and output head share a common backbone with lightweight per-codebook adapters. This preserves the distributional differences between codebooks while avoiding a full parameter copy for each codebook layer.

## Ⅴ Sequence format and streaming decoding

![](./images/sequence_format.jpg)

MiniMind-O places text tokens and 8 audio-code streams in the same training sample: Thinker handles the text sequence, Talker handles the audio-code sequence, and speech / image / voice conditions are injected through placeholders or reference codes. Loss on target text and target audio is computed only after the reply starts; reference and conditioning regions serve only as conditions and are not part of the reconstruction target.

For streaming generation, the model emits text tokens while simultaneously filling in 8 layers of Mimi codes via MTP and a delay schedule. The Mimi decoder can incrementally reconstruct the 24 kHz waveform, so playback does not have to wait for the full reply to finish.

## Ⅵ Voice control

Voice control is realized through in-context voice cloning: reference audio is first encoded into a voice prompt, and then fed to Talker as a contextual condition, instead of fine-tuning weights or rewriting the text prompt to specify a voice. The model can additionally use a speaker embedding to provide a more stable speaker constraint; switching the voice at inference time only requires changing these conditioning inputs, while the Thinker prompt and Talker weights remain unchanged.

The default release ships with 5 built-in voice prompts (dylan, eric, serena, uncle_fu, vivian), and reserves 7 unseen prompts for evaluation (arthur, chelsie, cherry, ethan, jennifer, momo, moon).

## Ⅶ Modules and parameter scale

The "0.1B" referenced for MiniMind-O denotes the trainable backbone composed of Thinker, Talker and the two projectors. For the released checkpoints, `minimind-3o` is about 113M and `minimind-3o-moe` is about 315M. The Audio Encoder, Vision Encoder and Speech Codec are frozen external side modules used only for feature extraction or audio (de)coding; together they contain about 425M parameters and are not counted as active MiniMind-O parameters.

The table below counts the main module sizes per released model. Trainable counts are based on PyTorch modules, with tied embeddings deduplicated.

| Counting scope | minimind-3o | minimind-3o-moe |
|---|---:|---:|
| Trainable backbone | 113.13M | 314.89M |
| Frozen external modules | 424.70M | 424.70M |
| Total loaded at runtime | 537.83M | 739.59M |

| Module | Implementation | Key configuration | Status / params (~3o / ~3o-moe) |
|---|---|---|---|
| Thinker | MiniMind Transformer | 8 layers, hidden 768 | trainable, 63.91M / 198.42M |
| Talker | Standalone MiniMind blocks | 4 layers, 8 codebook heads | trainable, 47.05M / 114.30M |
| Audio projector | `MMAudioProjector` | 512 → 768 | trainable, 0.99M |
| Vision projector | `MMVisionProjector` | 768 → 768 | trainable, 1.18M |
| Audio encoder | SenseVoice-Small | 16 kHz speech features | frozen, 234.00M |
| Vision encoder | SigLIP2 base-p32-256 | 256×256 image, 64 tokens | frozen, 94.55M |
| Speech codec | Mimi | 8 codebooks, 12.5 Hz, 24 kHz | frozen, 96.15M |
| Speaker condition | CAM++ embedding | 192-d speaker vector | precomputed |

# 📌 Experiments

## Ⅰ Datasets

Dataset download: [ModelScope](https://www.modelscope.cn/datasets/gongjy/minimind-o_dataset) | [HuggingFace](https://huggingface.co/datasets/jingyaogong/minimind-o_dataset)

All speech data is stored uniformly as Mimi codes (8 codebooks, 12.5 Hz frame rate). Images are resized uniformly to 256×256 and encoded by SigLIP2 P32 into 64 patch tokens. The training data mainly comes from public Omni / speech-instruction corpora, including [VoiceAssistant-400K](https://huggingface.co/datasets/gpt-omni/VoiceAssistant-400K), [UltraChat-300K-SLAM-Omni](https://huggingface.co/datasets/worstchan/UltraChat-300K-SLAM-Omni) and others. A large amount of multi-speaker audio is additionally synthesized with Qwen3-TTS, and CAM++ is used to extract speaker embeddings as voice conditions. The I2T data follows the same source as the visual instruction data used in [MiniMind-V](https://github.com/jingyaogong/minimind-v); please refer to that project for the original composition and citations.

The repository ships two training sets, **mini** and **full**. The mini set is filtered from full using the "English + no-vision" criteria and works with `train_sft_omni.py` using the default `--data_path`. Its goal is to verify the Thinker–Talker pipeline, Mimi (de)coding, sequence layout and voice-injection path at low cost, rather than to reproduce the Chinese speech ability of the released models. A Chinese Talker has to handle more complex grapheme-to-phoneme mapping, prosodic pauses and multi-speaker stability, which is clearly harder than English and cannot be expected to converge within ~2 hours on a single RTX 3090.

The full set corresponds to the released `minimind-3o` / `minimind-3o-moe` checkpoints and covers Chinese-English T2A / A2A as well as image-to-text. Sizes and language ratios are listed below; this is the actual training source behind the CER / voice-similarity numbers reported in the paper.

T2A means Text-to-Audio, A2A means Audio-to-Audio, and I2T means Image-to-Text.

| Dataset | Subset | Input speech | Output speech | Note |
|---|---|---|---|---|
| `sft_t2a_mini` | English T2A | — | ~470.14 h | mini onboarding |
| `sft_a2a_mini` | English A2A | ~74.64 h | ~56.60 h | mini onboarding |
| `sft_t2a` | zh+en T2A | — | ~1636.01 h | full training |
| `sft_a2a` | zh+en A2A | ~1711.97 h | ~423.40 h | full training |
| `sft_i2t` | Image I2T | — | — | full training |

In `sft_t2a`, Chinese / English / mixed samples account for 45.7% / 46.5% / 7.8% respectively; in `sft_a2a` the ratios are 70.8% / 21.2% / 8.0%. This distribution is directly reflected in behavior: short Chinese and short English replies are usually stable, while longer English speech is more prone to mispronunciation and word omissions. The mini subset keeps only English, so even with a tight budget on parameters and data, the within-language CER stays in a usable range.

## Ⅱ Training

The training entry point is `train_sft_omni.py`, and the recommended pipeline can be found in `trainer/train.sh`. Full training is not split into multiple complex pretraining stages; instead, capabilities are introduced incrementally along the data flow:

![](./images/training_pipeline.jpg)

- `sft_t2a`: align text with speech output first, so that Talker learns to generate Mimi codes under Thinker's semantic conditions;
- `sft_a2a`: bring in speech inputs, so that the model can enter the same Thinker–Talker reply path from speech instructions;
- `sft_i2t`: align the visual path last; the `vision_proj` mode updates only the vision projector to avoid image data overwriting language and speech abilities.

Among training modes, `all` updates MiniMind / Talker / projectors, while `audio_proj` and `vision_proj` are used solely to align the corresponding projector. SenseVoice-Small, SigLIP2 and Mimi are kept frozen throughout. The Dense and MoE variants share the same data ordering. The mini commands are meant only to make the pipeline runnable end-to-end and finish in ~2 hours on a single RTX 3090 by default; the released weights correspond to full training.

T2A and A2A loss curves during full training are shown below for reference:

![](./images/t2a_training_curves.jpg)

> `sft_t2a`: text-to-speech-output path

![](./images/a2a_training_curves.jpg)

> `sft_a2a`: loss after speech inputs are added

Early spikes caused by an incompatible weight resume have been removed from the T2A curve. The MoE variant has more total parameters but a similar number of active parameters compared with Dense, which makes it more useful as a capacity-allocation reference.

## Ⅲ Model weights

| Format | ModelScope | HuggingFace |
|---|---|---|
| PyTorch (`*.pth`) | [minimind-3o-pytorch](https://www.modelscope.cn/models/gongjy/minimind-3o-pytorch) | [minimind-3o-pytorch](https://huggingface.co/jingyaogong/minimind-3o-pytorch) |
| Transformers | [minimind-o collection](https://modelscope.cn/collections/gongjy/MiniMind-O) | [minimind-o collection](https://huggingface.co/collections/jingyaogong/minimind-o) |

> The Transformers version contains both `minimind-3o` and `minimind-3o-moe` and is suitable for direct use with `eval_omni.py` and the WebUI. The native PyTorch weights are mainly intended for training, reproducing experiments and continued fine-tuning.

# 📌 Evaluation

There is currently no unified evaluation protocol for Omni models: different works differ in the LLM backbone, the audio synthesizer and the system goal. Some focus on the LLM's own knowledge and reasoning and report MMLU, HumanEval and related benchmarks; some emphasize streaming latency and audio quality; some highlight speech-consistency metrics; and others focus on natural interaction or broader Omni generation. Most of these systems are continually trained from a state-of-the-art open-source LLM, while MiniMind's 0.06B backbone is clearly not competitive on complex knowledge QA, math reasoning, code generation or long open-ended replies, and the Talker's naturalness, prosody and stability are also weaker than those of full-scale systems.

Therefore, the goal here is not to chase a comprehensive leaderboard, but to focus on a few more reproducible local evaluations and use cases: a Talker hidden-size ablation, voice-cloning similarity, CER / WER comparisons under identical questions and identical ASR pipelines, and qualitative A2A, I2A and real-time interaction examples. CER / WER are mainly used to inspect text consistency, while audio quality, naturalness and human preference are left to qualitative samples and actual listening tests.

## Ⅰ Talker hidden-size ablation

If only speech generation is considered, scaling Talker to 1024 / 2048 hidden size or stacking more layers would obviously be more stable. But MiniMind-O has to fit the entire Omni pipeline within ~0.1B parameters, and cannot afford to allocate most of the budget to the acoustic side. Once Thinker and Talker are decoupled, language understanding and cross-modal fusion are mainly carried by Thinker, while Talker only renders Mimi codes from semantic conditions; this makes a small Talker possible. The rendering here is not "predict semantic tokens and hand them off to an external acoustic model"—Talker directly produces decodable Mimi acoustic codes, so the real bottleneck is at the output side: it has to handle 8 Mimi codebooks rather than a single next-token-prediction stream.

384-d is tempting, since the dense version compresses to ~88M; 512-d is also lighter. But the table below shows that smaller does not automatically mean better allocated: short utterances remain acceptable, but medium-to-long ones are more prone to word drops, repetitions and pronunciation drift. 768-d was kept in the end because it matches the MiniMind backbone width and can be initialized from the last 4 layers of Thinker; the parameter count remains around 0.1B, training cost does not increase noticeably, and consistency is clearly more stable.

| Variant | Talker hidden | Params | Avg CER ↓ | Short ↓ | Mid / Long ↓ |
|---|---|---|---|---|---|
| Dense | 768 | 115.29M | **0.0897** | 0.1528 | 0.0874 / 0.0675 |
| Dense | 512 | 96.13M | 0.1745 | 0.2709 | 0.2455 / 0.0976 |
| Dense | 384 | 88.72M | 0.2767 | 0.3904 | 0.1865 / 0.4046 |
| MoE | 768 | 317.05M-A115.33M | **0.0900** | 0.2075 | 0.0533 / 0.0271 |
| MoE | 512 | 261.32M-A96.17M | 0.1265 | 0.0711 | 0.1490 / 0.1464 |
| MoE | 384 | 240.04M-A88.75M | 0.3280 | 0.3757 | 0.2777 / 0.4313 |

Dense and MoE CERs should not be compared directly across architectures: under the same question, the two Thinkers may produce different content with different lengths, leading to different synthesis difficulty for Talker. What matters more is the within-architecture trend: 768 clearly outperforms 512 and 384 in both cases.

## Ⅱ Voice-cloning similarity

Voice cloning is one of the more beta-quality features in this release. To our knowledge, most open-source Omni models support only fixed output voices, while minimind-3o tries to fit multi-voice generation into a single Talker. This goal is harder than simply "being able to talk", because the model needs not only to say the right content, but also to preserve speaker cues while generating Mimi codes.

Quality has not yet reached high-fidelity cloning: the same reference voice does not always stay consistent across questions, and longer utterances can drift because of pronunciation and rhythm issues. But basic male / female differences, intonation tendencies and parts of the prosody are distinguishable.

The CAM++ speaker-embedding cosine similarity below is only an automatic reference. Seen comes from the 5 built-in voices in `voices.pt`; Unseen comes from 7 voices in `voices_unseen.pt` that were never seen during training. Each voice uses the same set of text questions and only the voice condition is swapped.

Per-speaker breakdown:

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

Overall, minimind-3o and minimind-3o-moe land at similar averages, both slightly above the early baseline. This suggests that voice retention is not primarily determined by inactive expert capacity; the more direct factors are reference-clip quality, the separability of CAM++ embeddings, and the stability of Talker generation itself. Per-speaker, voices like uncle_fu, serena and arthur are easier to preserve, with at least one variant exceeding 0.70; outliers like eric and moon are more sensitive to generation quality. In other words, this capability already separates some speaker characteristics, but is still some distance away from a product-level "given a reference clip, faithfully reproduce its timbre" experience.

### Voice-cloning ablation samples (audio playback)

For a more direct listening test, seed=42 and temperature=0.7 are fixed, and one generated sample is shown per voice. The only variables are the reference audio codes and speaker embedding. As a control, the default output without any reference voice condition is shown first (the spoken text is identical across all samples):

https://github.com/user-attachments/assets/b31fd8f2-e3af-4fed-ba19-65424b59bec6

#### Seen voices
"Seen" means voices that appeared in training data, used to inspect how well the model preserves familiar speakers.

<table>
<tr><th width="100">Speaker</th><th width="380">Reference</th><th width="380">Output</th><th width="80">Avg</th></tr>
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

#### Unseen voices
"Unseen" means voices not seen during training, used to inspect zero-shot transfer of a new reference voice into generated speech.

<table>
<tr><th width="100">Speaker</th><th width="380">Reference</th><th width="380">Output</th><th width="80">Avg</th></tr>
<tr><td>arthur</td><td>

https://github.com/user-attachments/assets/46e720ba-c413-4731-8275-a9a2cb483690

</td><td>

https://github.com/user-attachments/assets/f489b463-c7d7-4ad6-9e4d-53333172c66c

</td><td>0.6479</td></tr>
<tr><td>chelsie</td><td>

https://github.com/user-attachments/assets/3430ecdb-6de8-4fb0-a6a7-ad82bdce01a1

</td><td>

https://github.com/user-attachments/assets/e598dbc2-ba28-4c38-b52d-6fa6c2349a5b

</td><td>0.5975</td></tr>
<tr><td>cherry</td><td>

https://github.com/user-attachments/assets/f9166af6-3a98-42f3-9cf8-ad105eea87d6

</td><td>

https://github.com/user-attachments/assets/eccca693-4708-409a-88f7-85eb25f66fe6

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

## Ⅲ Cross-model English T2A comparison

We selected 20 English questions, all constrained by `Answer briefly in one short sentence`. The intent is not to evaluate open-ended English ability, but to keep response lengths within a similar range. The three models then synthesize audio, which is uniformly transcribed by Qwen3-ASR; CER / WER between transcription and target text are used to compare Talker-side textual consistency.

| Length bucket | [Mini-Omni](https://huggingface.co/gpt-omni/mini-omni) CER/WER | [Mini-Omni2](https://huggingface.co/gpt-omni/mini-omni2) CER/WER | minimind-3o CER/WER |
|---|---|---|---|
| short (≤15w) | 0.0195 / 0.0384 (n=8) | 0.0503 / 0.0584 (n=14) | 0.0531 / 0.0417 (n=8) |
| mid (16–30w) | 0.0038 / 0.0052 (n=12) | 0.0062 / 0.0076 (n=6) | 0.1327 / 0.1420 (n=11) |
| long (31–60w) | — | — | 0.0431 / 0.0508 (n=1) |

For replies of ≤15 words, minimind-3o is already close to Mini-Omni2; the gap really opens up at 16–30 words. This length is no longer a simple phrase, and Talker must keep pronunciation, rhythm and surface form consistent in a complete short sentence simultaneously. This is also the regime where the current 0.1B Talker most easily exposes its instability.

## Ⅳ Cross-model vision-language comparison

[Mini-Omni](https://huggingface.co/gpt-omni/mini-omni) does not support a VL path, so the comparison is between [Mini-Omni2](https://huggingface.co/gpt-omni/mini-omni2) (0.5B) and minimind-3o (0.1B). On 9 synthetic images, both models generate English answers, which are then uniformly transcribed and used to compute CER / WER as a vision-to-speech consistency reference.

| Model | Params | Avg CER ↓ | Avg WER ↓ |
|---|---|---|---|
| [Mini-Omni2](https://huggingface.co/gpt-omni/mini-omni2) | 0.5B | 0.7609 | 0.9756 |
| minimind-3o | 0.1B | 0.8241 | 1.0293 |

These numbers should not be read as the absolute correctness of open-ended image description. Image captioning has many equivalent expressions, and synonym choices and word order both affect CER / WER, so high absolute values are expected. Under the same automatic pipeline, minimind-3o trails behind Mini-Omni2 but stays in the same order of magnitude, with roughly 1/5 the parameters.

## Ⅴ Qualitative samples

![](./images/qual_a2a.jpg)

In speech-to-speech samples, the input is real speech, Thinker organizes the semantics, and Talker renders speech. Short replies are again the more stable regime; Chinese explanatory questions usually produce coherent answers, while English pronunciation and rhythm are relatively more stable.

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

The image-QA samples chain visual encoding, text generation and speech rendering inside the same path. The current model usually captures the main object and the rough scene, but fine-grained spatial relations, counts and attributes are still often wrong, which makes it more suitable as a reproducible baseline for tiny-model Omni pipelines.

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

## Ⅵ Real-time interaction

![](./images/realtime_interaction.jpg)

This is the real-time interaction interface. Once the user stops speaking, Thinker first finishes the semantic-side prefill, Talker then starts to emit audio codes incrementally, and the Mimi decoder writes the 24 kHz waveform as it receives codes. The barge-in example shows another path that is closer to a real conversation: when the user starts speaking again while the model is talking, the system interrupts the current generation and re-enters the prefill–reply flow. Interruption detection here is still based on a simple VAD threshold, not yet semantic-level barge-in; but from an engineering loop-closure perspective, the system can already fall back from speaking to listening and process the next turn.

### 🧩 Possible future improvements

The current model still has clear gaps compared with large-scale Omni systems, and there is no need to gloss over them. Long-form speech naturalness, complex visual reasoning, open-ended English mid/long replies and voice stability are not its strong areas. The visual path is closer to a compact vision-to-speech link, and the MoE variant looks more like a capacity-allocation experiment than a same-FLOP optimum.

These limitations also point to several follow-ups: longer ICL contexts, finer prosody supervision, stronger vision encoders, more stable voice conditions, and systematic sweeps over the Bridge layer and the MTP codebook interface—all of which are worth continuing.

That said, the value of MiniMind-O lies exactly here. It compresses an entire Omni loop into the 0.1B regime, and ships code, weights and the main training data inside the same inspectable artifact. This means it is not just a demo, but a baseline small enough, transparent enough, and reproducible enough to rebuild from scratch and modify further. For people who want to understand Thinker–Talker decoupling, the MTP codebook interface, in-context voice cloning, and the middle-hidden bridge, it offers a set of design choices that can actually be verified by hand.

# 📌 Acknowledgements

> [!TIP]
> If you find `MiniMind-O` helpful, consider giving us a ⭐ on GitHub.<br/>
> Given limited bandwidth there will inevitably be unknown bugs. Discussions, corrections and PRs in Issues are welcome.<br/>
> Your support is what keeps the project moving—thank you!

## 🤝 Contributors

<a href="https://github.com/jingyaogong/minimind-o/graphs/contributors">
  <img width="200" src="https://contrib.rocks/image?repo=jingyaogong/minimind-o" />
</a>

## 😊 Credits

- [MiniMind](https://github.com/jingyaogong/minimind) / [MiniMind-V](https://github.com/jingyaogong/minimind-v) (backbone, data)
- [Qwen2.5-Omni / Qwen3-Omni](https://github.com/QwenLM/Qwen2.5-Omni) (inspiration, data)
- [Mini-Omni / Mini-Omni2](https://github.com/gpt-omni/mini-omni) (inspiration, data)
- [SLAM-Omni](https://aclanthology.org/2025.findings-acl.115/) (data)
- [SenseVoice](https://arxiv.org/abs/2407.04051) (component)
- [Mimi / Moshi](https://arxiv.org/abs/2410.00037) (component)

## 🫶 Supporters

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

# 🎓 Citation

If MiniMind-O helps your research or work, please cite:

```bibtex
% Cite the technical report when referencing the model design or experimental results.
@article{minimind-o-report,
    title   = {MiniMind-O Technical Report: An Open Small-Scale Speech-Native Omni Model}, 
    author  = {Jingyao Gong},
    journal = {arXiv preprint arXiv:2605.03937},
    year    = {2026}
}

% Cite the GitHub repo when referencing the open-source codebase or released weights.
@misc{minimind-o,
    title  = {MiniMind-O: Train a Tiny Omni Model from Scratch},
    author = {Jingyao Gong},
    year   = {2026},
    url    = {https://github.com/jingyaogong/minimind-o},
    note   = {GitHub repository, accessed 2026}
}
```

# 📜 License

This repository is released under the [Apache-2.0 License](LICENSE).
