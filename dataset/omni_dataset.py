import torch
import os
import math
import random
import numpy as np
import soundfile as sf
import librosa
import json
import io
from PIL import Image
from scipy.signal import resample
from torch.utils.data import Dataset
import pyarrow as pa
import pyarrow.parquet as pq

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def pre_processing_chat(conversations, add_system_ratio=0.2):
    if any(conv.get('tools') for conv in conversations): return conversations

    SYSTEM_PROMPTS = [
        "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
        "你是minimind，一个小巧但有用的语言模型。",
        "你是一个专业的AI助手，请提供有价值的回答。",
        "你是minimind，请尽力帮助用户解决问题。",
        "你是一个可靠的AI，请给出准确的回答。",
        "You are a helpful AI assistant.",
        "You are minimind, a lightweight intelligent assistant.",
        "You are a friendly chatbot. Please answer the user's questions carefully.",
        "You are a knowledgeable AI. Try your best to provide accurate information.",
        "You are minimind, a small but useful language model."
    ]
    if conversations[0].get('role') != 'system':
        if random.random() < add_system_ratio:
            return [{'role': 'system', 'content': random.choice(SYSTEM_PROMPTS)}] + conversations
    return conversations

def post_processing_chat(prompt_content, empty_think_ratio=0.2):
    if '<think>\n\n</think>\n\n' in prompt_content and random.random() > empty_think_ratio:
        prompt_content = prompt_content.replace('<think>\n\n</think>\n\n', '')
    return prompt_content


class OmniDataset(Dataset):
    def __init__(self, data_path, tokenizer, audio_processor=None, vision_processor=None,
                 max_length=1200, audio_special_token='<|audio_pad|>', image_special_token='<|image_pad|>',
                 audio_stop_token=2050,  # <|audio_stop|>
                 audio_pad_token=2049,  # <|audio_pad|>
                 audio_spk_token=2051,  # <|audio_spk|>
                 audio_vocab_size=2112,  # 2048 mimi codes + 64 special tokens
                 scheduled_sampling=0.05,
                 image_token_len=64):
        super().__init__()
        tables = [pa.Table.from_batches(pq.ParquetFile(p.strip()).iter_batches()) for p in data_path.split(',')]
        tables = [t.cast(pa.schema([f.with_type(pa.large_string()) if pa.types.is_string(f.type) else f for f in t.schema])) for t in tables]
        self.table = pa.concat_tables(tables, promote_options='default')
        self.tokenizer = tokenizer
        self.audio_processor = audio_processor
        self.vision_processor = vision_processor
        self.max_length = max_length
        self.audio_token = audio_special_token
        self.image_token_len = image_token_len
        self.image_token = image_special_token * image_token_len
        self.audio_stop_token = audio_stop_token
        self.audio_pad_token = audio_pad_token
        self.audio_spk_token = audio_spk_token
        self.audio_vocab_size = audio_vocab_size
        self.scheduled_sampling_prob = scheduled_sampling
        self.text_vocab_size = len(tokenizer)
        self.image_token_id = tokenizer.encode(image_special_token, add_special_tokens=False)[0]
        self.audio_token_id = tokenizer.encode(audio_special_token, add_special_tokens=False)[0]
        self.think_end_ids = tokenizer.encode('</think>\n\n', add_special_tokens=False)
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.table)

    @staticmethod
    def process_audio(audio_path, audio_processor):
        """加载音频并预处理成fbank，返回 (fbank (T,560), valid_len=encoder输出帧数)"""
        wav, sr = sf.read(audio_path)
        if wav.ndim > 1: wav = wav.mean(axis=1)
        if sr != 16000: wav = librosa.resample(wav.astype(float), orig_sr=sr, target_sr=16000)
        inputs = audio_processor(wav.astype(np.float32), sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
        valid_len = inputs.attention_mask.sum().item()
        return inputs.input_features.squeeze(0), valid_len

    def augment_wav(self, wav, sr=16000):
        # 随机变速(0.7~1.6x)：改变音频时长和音调，覆盖快/慢语速
        if random.random() < 0.5:
            speed = random.uniform(0.7, 1.6)
            wav = resample(wav, int(len(wav) / speed)).astype(np.float32)
        # 随机加噪：叠加轻微高斯白噪声，模拟录音环境差异
        if random.random() < 0.3:
            noise = np.random.randn(len(wav)).astype(np.float32) * random.uniform(0.001, 0.01)
            wav = wav + noise
        # 随机音量：缩放振幅0.8~1.2倍，模拟说话音量变化
        if random.random() < 0.3:
            wav = wav * random.uniform(0.8, 1.2)
        # 随机时间遮蔽：将0.25秒片段置零，模拟短暂静音/丢包
        if random.random() < 0.2 and len(wav) > sr:
            start = random.randint(0, len(wav) - sr // 4)
            wav[start:start + sr // 4] = 0
        # 随机低通滤波：移动平均模糊高频，模拟电话/低质量麦克风
        if random.random() < 0.2:
            k = random.choice([3, 5, 7])
            wav = np.convolve(wav, np.ones(k) / k, mode='same').astype(np.float32)
        # 随机混响：合成指数衰减脉冲响应卷积，模拟房间反射/回声
        if random.random() < 0.3:
            ir_len = int(sr * random.uniform(0.05, 0.2))
            ir = np.random.randn(ir_len).astype(np.float32) * np.exp(-np.linspace(0, 10, ir_len))
            ir[0] = 1.0
            ir /= np.sqrt(np.sum(ir ** 2) + 1e-6)
            wav = np.convolve(wav, ir, mode='same').astype(np.float32)
        # 随机粉红噪声：1/f噪声模拟房间环境底噪（空调/远处人声）
        if random.random() < 0.2:
            pink = np.cumsum(np.random.randn(len(wav))).astype(np.float32)
            pink /= np.max(np.abs(pink)) + 1e-6
            wav = wav + pink * random.uniform(0.003, 0.015)
        return np.clip(wav, -1.0, 1.0).astype(np.float32)

    def augment_mel(self, fbank):
        # fbank: (T, 560) — SenseVoice LFR 后的特征（时间维在前，频率维在后）
        T, D = fbank.shape
        # SpecAugment频率遮蔽：随机抹掉1~64维，防止对特定频段过拟合
        if random.random() < 0.5:
            f = random.randint(1, 64)
            f0 = random.randint(0, D - f)
            fbank[:, f0:f0 + f] = 0
        # SpecAugment时间遮蔽：随机抹掉1~min(10,T)帧，提升对不完整输入的容错
        if random.random() < 0.5 and T > 1:
            t = random.randint(1, min(10, T))
            t0 = random.randint(0, T - t)
            fbank[t0:t0 + t, :] = 0
        return fbank

    def load_audio_inputs(self, audio_bytes):
        if not audio_bytes: return None, 0
        wav, sr = sf.read(io.BytesIO(audio_bytes))
        if wav.ndim > 1: wav = wav.mean(axis=1)
        if sr != 16000: wav = librosa.resample(wav.astype(float), orig_sr=sr, target_sr=16000)
        wav = self.augment_wav(wav.astype(np.float32))
        inputs = self.audio_processor(wav, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
        valid_len = inputs.attention_mask.sum().item()
        return self.augment_mel(inputs.input_features.squeeze(0)), valid_len

    def load_image_inputs(self, image_bytes):
        if not image_bytes or self.vision_processor is None: return None
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        inputs = self.vision_processor(images=image, return_tensors="pt")
        if hasattr(inputs, 'keys'): return {k: v for k, v in inputs.items()}
        return inputs.pixel_values

    def create_chat_prompt(self, conversations, audio_features_length=0):
        conversations = pre_processing_chat(conversations)
        messages = []
        is_last_user = lambda i: i == max(j for j, t in enumerate(conversations) if t['role'] == 'user')
        for idx, turn in enumerate(conversations):
            role, content = turn['role'], turn['content']
            if role == 'user' and is_last_user(idx) and audio_features_length > 0:
                ap = self.audio_token * audio_features_length
                r = random.random()
                if r < 0.4: content = ap
                elif r < 0.6: content = content
                elif r < 0.8: content = ap + '\n\n' + content
                else: content = content + '\n\n' + ap
            if '<image>' in content:
                r = random.random()
                if r < 0.2: content = '<image>\n' + content.replace('<image>', '').strip()
                elif r < 0.4: content = '<image>\n\n' + content.replace('<image>', '').strip()
                elif r < 0.6: content = content.replace('<image>', '').strip() + '\n' + '<image>'
                else: content = content.replace('<image>', '').strip() + '\n\n' + '<image>'
            messages.append({"role": role, "content": content})
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return post_processing_chat(prompt)

    
    def generate_text_labels(self, input_ids):
        labels = [-100] * len(input_ids)
        ranges = []
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                ranges.append((start, end))
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    labels[j] = input_ids[j]
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return labels, ranges
    
    def apply_scheduled_sampling(self, input_ids, audio_labels, text_labels):
        """Scheduled Sampling: 用随机值替代部分GT，让模型学会从错误历史中恢复"""
        if self.scheduled_sampling_prob <= 0:
            return input_ids
        audio_mask = (audio_labels != -100).any(dim=0) & (torch.rand(input_ids.size(1)) < self.scheduled_sampling_prob)
        for i in range(8):
            input_ids[i] = torch.where(audio_mask, torch.randint(0, self.audio_vocab_size, input_ids[i].shape), input_ids[i])
        # 保护 image token 的连续性
        text_mask = (text_labels != -100) & (input_ids[8] != self.image_token_id) & (torch.rand(input_ids.size(1)) < self.scheduled_sampling_prob)
        input_ids[8] = torch.where(text_mask, torch.randint(0, self.text_vocab_size, input_ids[8].shape), input_ids[8])
        return input_ids

    def __getitem__(self, index: int):
        conversations = json.loads(self.table['conversations'][index].as_py())
        question_audios = self.table['question_audios'][index].as_py() if 'question_audios' in self.table.column_names else []
        answer_audios = self.table['answer_audios'][index].as_py() if 'answer_audios' in self.table.column_names else []
        image_bytes = self.table['image_bytes'][index].as_py() if 'image_bytes' in self.table.column_names else []
        if image_bytes and not isinstance(image_bytes, list): image_bytes = [image_bytes]
        ref_audios = self.table['ref_audios'][index].as_py() if 'ref_audios' in self.table.column_names else []
        spk_emb_raw = self.table['spk_emb'][index].as_py() if 'spk_emb' in self.table.column_names else []
        
        # 随机截断到某一轮（每轮=user+assistant）
        asst_indices = [i for i, t in enumerate(conversations) if t['role'] == 'assistant']
        if len(asst_indices) > 1:
            rand_idx = random.randint(0, len(asst_indices) - 1)
            # 从随机轮次开始，向前回退直到长度安全
            for i in range(rand_idx, -1, -1):
                conversations = conversations[:asst_indices[i] + 1]
                test_prompt = self.create_chat_prompt(conversations, 0)
                if len(self.tokenizer(test_prompt).input_ids) + 100 < self.max_length:
                    break
        
        # 加载最后一个user的图像（按user轮次索引访问，与audio一致）
        pixel_values = None
        user_count = sum(1 for t in conversations if t['role'] == 'user')
        if image_bytes and len(image_bytes) > 0 and self.vision_processor:
            pixel_values = self.load_image_inputs(image_bytes[0])
        
        # 只加载最后一个user的audio（按user轮次索引访问）
        audio_inputs, audio_len, audio_features_length = None, 0, 0
        user_count = sum(1 for t in conversations if t['role'] == 'user')
        if question_audios and user_count > 0 and user_count <= len(question_audios) and self.audio_processor:
            audio_bytes = question_audios[user_count - 1]
            if audio_bytes:
                mel, valid_len = self.load_audio_inputs(audio_bytes)
                if mel is not None:
                    audio_inputs = mel.unsqueeze(0)
                    audio_len = valid_len
                    audio_features_length = valid_len or 1
        
        # 混合训练时，无音频样本返回dummy tensor保持batch索引尽可能对齐 (SenseVoice: T x 560)
        if audio_inputs is None and self.audio_processor:
            audio_inputs = torch.zeros(1, 1, 560)
            audio_len = 0
        if pixel_values is None and self.vision_processor:
            pixel_values = {'pixel_values': torch.zeros(1, 3, 256, 256)}
        
        # 从answer_audios获取最后一个assistant的音频codes
        last_audio_codes = None
        asst_count = sum(1 for t in conversations if t['role'] == 'assistant')
        if answer_audios and asst_count > 0 and asst_count <= len(answer_audios):
            tokens = answer_audios[asst_count - 1]
            if tokens:
                audio_codes_8layers = [[] for _ in range(8)]
                for i in range(0, len(tokens) - 7, 8):
                    for j in range(8): audio_codes_8layers[j].append(tokens[i + j])
                for layer in audio_codes_8layers: layer.append(self.audio_stop_token)
                last_audio_codes = audio_codes_8layers
        
        # 生成prompt (text input_ids)
        prompt = self.create_chat_prompt(conversations, audio_features_length)
        if pixel_values is not None: prompt = prompt.replace('<image>', self.image_token)
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        
        # PAD input_ids到max_length
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        
        # 生成labels（只训练最后一个assistant）
        text_labels, assistant_ranges = self.generate_text_labels(input_ids)
        for start, end in assistant_ranges[:-1]:
            text_labels[start:end] = [-100] * (end - start)
        
        # 生成7层audio targets（只填充最后一个assistant）
        Y_audio_layers = [[self.audio_pad_token] * self.max_length for _ in range(8)]
        audio_labels = [[-100] * self.max_length for _ in range(8)]
        if assistant_ranges and last_audio_codes:
            assistant_start, assistant_end = assistant_ranges[-1]
            for pos in range(assistant_start, min(assistant_end, assistant_start + 50)):
                if input_ids[pos:pos + len(self.think_end_ids)] == self.think_end_ids:
                    assistant_start = pos + len(self.think_end_ids)
                    break
            # spk_emb 占位 + ref_codes 右对齐（50% 概率 drop ref_codes，只保留 spk）
            has_spk = bool(spk_emb_raw)
            has_ref = bool(ref_audios) and random.random() > 0.5
            spk_reserve = 1 if has_spk else 0
            if has_ref:
                ref_codes = [[] for _ in range(8)]
                for i in range(0, len(ref_audios) - 7, 8):
                    for j in range(8): ref_codes[j].append(ref_audios[i + j])
                ref_len = len(ref_codes[0])
                ref_start = max(spk_reserve, assistant_start - ref_len)
                for layer_idx in range(8):
                    codes = ref_codes[layer_idx][-(assistant_start - ref_start):] if ref_len > (assistant_start - ref_start) else ref_codes[layer_idx]
                    for i, code in enumerate(codes):
                        Y_audio_layers[layer_idx][ref_start + i] = code
            else:
                ref_start = assistant_start
            if has_spk and ref_start > 0:
                spk_pos = ref_start - 1
                for layer_idx in range(8):
                    Y_audio_layers[layer_idx][spk_pos] = self.audio_spk_token
            # target codes 填充到 assistant_start 之后（参与 loss）
            for layer_idx in range(8):
                codes = last_audio_codes[layer_idx]
                start_pos = assistant_start + layer_idx + 1
                for i, code in enumerate(codes):
                    if start_pos + i < self.max_length:
                        Y_audio_layers[layer_idx][start_pos + i] = code
                        audio_labels[layer_idx][start_pos + i] = code
        
        # 构造9路输入：input_ids = (9, T) = 8路audio + 1路text
        X_audio = torch.tensor([layer[:-1] for layer in Y_audio_layers], dtype=torch.long)  # (8, T-1)
        X_text = torch.tensor(input_ids[:-1], dtype=torch.long)  # (T-1,)
        input_ids = torch.cat((X_audio, X_text.unsqueeze(0)), dim=0)  # (9, T-1)
        text_labels = torch.tensor(text_labels[1:], dtype=torch.long)  # (T-1,)
        audio_labels = torch.tensor([layer[1:] for layer in audio_labels], dtype=torch.long)  # (8, T-1)
        
        input_ids = self.apply_scheduled_sampling(input_ids, audio_labels, text_labels)
        spk_emb = torch.tensor(spk_emb_raw, dtype=torch.float32) if spk_emb_raw else torch.zeros(192)
        return input_ids, text_labels, audio_labels, audio_inputs, audio_len, pixel_values, spk_emb


# 测试parquet数据读取
if __name__ == '__main__':
    for path in ['sft_a2a.parquet']:
        if not os.path.exists(path): continue
        t = pa.Table.from_batches(pq.ParquetFile(path).iter_batches())
        conversations = json.loads(t['conversations'][0].as_py())
        answer_audios = t['answer_audios'][0].as_py() if 'answer_audios' in t.column_names else []
        user_msg = conversations[0]
        asst_msg = conversations[1] if len(conversations) > 1 else {}
        print(f'{path}: {len(t)}条, 列{t.column_names}')
        print(f'  User: {user_msg["content"][:50]}...')
        print(f'  Asst: {asst_msg.get("content", "")[:50]}...')
        if answer_audios:
            print(f'  answer_audios: {len(answer_audios)}轮, 首轮{len(answer_audios[0])}tokens')