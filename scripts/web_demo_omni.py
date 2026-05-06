import os
import sys

__package__ = "scripts"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import base64
import io
import tempfile
import warnings
import logging
import contextlib
import librosa
import soundfile as sf
import numpy as np
import torch
import gradio as gr
from queue import Queue
from threading import Thread, Lock
from PIL import Image
from pydub import AudioSegment
from transformers import AutoTokenizer, AutoModelForCausalLM, MimiModel, TextStreamer
from model.model_omni import MiniMindOmni, OmniConfig
from dataset.omni_dataset import OmniDataset
from trainer.trainer_utils import setup_seed, log_model_params
logging.getLogger().setLevel(logging.ERROR)
with contextlib.redirect_stdout(io.StringIO()):
    from funasr import AutoModel
    from funasr.utils.postprocess_utils import rich_transcription_postprocess

warnings.filterwarnings('ignore')

model_lock = Lock()

model, tokenizer, device, mimi_model, asr_model = None, None, None, None, None
voices_data = {}
builtin_voices, clone_voices = set(), set()


def frames_to_mimi(frames):
    codes = [f for f in frames if f and len(f) == 8]
    if not codes:
        return None
    return torch.tensor(codes, dtype=torch.long).T.unsqueeze(0)


def decode_mimi(mimi_codes):
    if mimi_codes is None or mimi_codes.numel() == 0:
        return None
    filtered = torch.where(mimi_codes >= 2049, torch.zeros_like(mimi_codes), mimi_codes).to(next(mimi_model.parameters()).device)
    with torch.no_grad():
        audio = mimi_model.decode(filtered).audio_values
    return audio.squeeze().float().cpu().numpy()


def scan_hf_models(base_dir):
    models = {}
    base_dir = os.path.abspath(base_dir)
    for d in sorted(os.listdir(base_dir), reverse=True):
        full_path = os.path.join(base_dir, d)
        if not os.path.isdir(full_path) or d.startswith('.') or d.startswith('_'):
            continue
        files = set(os.listdir(full_path))
        has_model = bool(files & {'pytorch_model.bin', 'model.safetensors', 'pytorch_model.bin.index.json', 'model.safetensors.index.json'})
        if has_model:
            models[d] = full_path
    return models


def load_hf_model(model_path):
    global model, tokenizer
    with model_lock:
        [sys.modules.pop(k) for k in list(sys.modules) if 'transformers_modules' in k]
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
        vision_encoder, vision_processor = MiniMindOmni.load_vision(args.vision_model)
        audio_encoder, audio_processor = MiniMindOmni.load_sensevoice(args.audio_encoder)
        object.__setattr__(model, 'vision_encoder', vision_encoder)
        object.__setattr__(model, 'vision_processor', vision_processor)
        object.__setattr__(model, 'audio_encoder', audio_encoder)
        object.__setattr__(model, 'audio_processor', audio_processor)
        model = model.half().eval().to(device)
        if model.audio_encoder:
            model.audio_encoder.to(device)
        if model.vision_encoder:
            model.vision_encoder.to(device)
        name = os.path.basename(model_path)
        params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f'已加载 {name}，参数量：{params:.2f}M')
        return f"已加载: {name} ({params:.2f}M)"


def load_voices():
    global voices_data, builtin_voices, clone_voices
    voices_data = {}
    builtin_voices, clone_voices = set(), set()
    for name, is_builtin in [('voices.pt', True), ('voices_unseen.pt', False)]:
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'model', 'speaker', name)
        if os.path.exists(path):
            data = torch.load(path, map_location=device)
            for speaker, v in data.items():
                if speaker not in voices_data:
                    voices_data[speaker] = v
                    (builtin_voices if is_builtin else clone_voices).add(speaker)


def chat_stream(prompt, audio_input=None, image_input=None, voice_name="default", history=None, temperature=0.85, max_tokens=512):
    audio_inputs, audio_lens, pixel_values, ref_codes, spk_emb = None, None, None, None, None
    asr_result = [None]
    asr_thread = None

    if audio_input is not None:
        sr, samples = audio_input
        if len(samples.shape) > 1:
            samples = samples.mean(axis=1)
        if samples.dtype != np.float32:
            samples = samples.astype(np.float32) / max(np.abs(samples).max(), 1)
        if sr != 16000:
            samples = librosa.resample(samples.astype(float), orig_sr=sr, target_sr=16000).astype(np.float32)
        inputs = model.audio_processor(samples, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
        mel = inputs.input_features.squeeze(0)
        valid_len = inputs.attention_mask.sum().item()
        audio_inputs = mel.unsqueeze(0).to(device)
        audio_lens = torch.tensor([valid_len], device=device)
        audio_token_len = valid_len or 1
        prompt = model.config.audio_special_token * audio_token_len
        samples_for_asr = samples.copy()
        def _do_asr():
            r = asr_model.generate(input=samples_for_asr, cache={}, language='auto', use_itn=True)
            asr_result[0] = rich_transcription_postprocess(r[0]['text']).strip() if r else ''
            print(f'[ASR] {asr_result[0]}')
        asr_thread = Thread(target=_do_asr)
        asr_thread.start()

    if image_input is not None:
        image = Image.open(image_input).convert('RGB') if isinstance(image_input, str) else image_input.convert('RGB')
        pixel_values = {k: v.to(device) for k, v in model.vision_processor(images=image, return_tensors="pt").items()}
        prompt = (prompt + "\n\n" if prompt else "") + model.config.image_special_token * model.config.image_token_len

    if voice_name != "default" and voice_name in voices_data:
        v = voices_data[voice_name]
        ref_codes = v['ref_codes'].unsqueeze(0).to(device)
        spk_emb = v['spk_emb'].half().unsqueeze(0).to(device) if 'spk_emb' in v else None

    messages = (history or []) + [{"role": "user", "content": prompt}]
    open_thinking = bool(args.open_thinking)
    inputs_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, open_thinking=open_thinking)
    x = torch.tensor(tokenizer(inputs_text).data['input_ids'], dtype=torch.long, device=device)[None, ...]

    audio_frames = []
    with torch.no_grad():
        with model_lock:
            res = model.generate(x, tokenizer.eos_token_id, max_new_tokens=max_tokens,
                                 temperature=temperature, top_p=args.top_p, stream=True,
                                 return_audio_codes=True, open_thinking=open_thinking,
                                 audio_inputs=audio_inputs, audio_lens=audio_lens, pixel_values=pixel_values,
                                 ref_codes=ref_codes, spk_emb=spk_emb)
            history_idx = 0
            for y, audio_frame in res:
                text_chunk = None
                if y is not None:
                    answer = tokenizer.decode(y[0].tolist(), skip_special_tokens=True)
                    if answer and answer[-1] != '�' and len(answer) > history_idx:
                        text_chunk = answer[history_idx:]
                        history_idx = len(answer)
                if audio_frame:
                    audio_frames.append(audio_frame)
                yield text_chunk, None, None

    if asr_thread:
        asr_thread.join()

    if audio_frames and mimi_model:
        yield None, "loading_audio", None
        mimi_codes = frames_to_mimi(audio_frames)
        audio_np = decode_mimi(mimi_codes)
        if audio_np is not None:
            yield None, (24000, audio_np), asr_result[0]
            return
    yield None, None, asr_result[0]


def launch_gradio(server_name="0.0.0.0", server_port=8888):
    voice_choices = [("default", "default")]
    for s in sorted(builtin_voices): voice_choices.append((f"[内置] {s}", s))
    for s in sorted(clone_voices): voice_choices.append((f"[克隆] {s}", s))

    def respond(message, audio, voice, chat_history, model_history, max_turns):
        text = message.get("text", "") if isinstance(message, dict) else (message or "")
        files = message.get("files", []) if isinstance(message, dict) else []
        img_path = next((f for f in files if any(f.lower().endswith(e) for e in ('.png','.jpg','.jpeg','.gif','.bmp','.webp'))), None)

        if not text and audio is None and img_path is None:
            yield chat_history + [{"role": "assistant", "content": "请输入文本、上传图片或录制音频"}], gr.update(), gr.update(), model_history, ""
            return

        if audio is not None:
            sr, samples = audio
            display_samples = samples.mean(axis=1) if len(samples.shape) > 1 else samples
            wav_path = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name
            sf.write(wav_path, display_samples, sr)
            chat_history = chat_history + [{"role": "user", "content": {"path": wav_path}}]
        elif img_path:
            chat_history = chat_history + [{"role": "user", "content": {"path": img_path}}, {"role": "user", "content": text or "请描述这张图片"}]
        else:
            chat_history = chat_history + [{"role": "user", "content": text}]

        response_text = ""
        final_audio = None
        asr_text = None
        chat_history = chat_history + [{"role": "assistant", "content": ""}]
        yield chat_history, gr.update(value=None), None, model_history, ""

        hist = model_history[-(max_turns * 2):] if max_turns > 0 else []
        for text_chunk, audio_data, asr in chat_stream(
            text or "", audio_input=audio, image_input=img_path,
            voice_name=voice, history=hist, temperature=0.7, max_tokens=512
        ):
            if text_chunk:
                response_text += text_chunk
                chat_history[-1]["content"] = response_text
                yield chat_history, gr.update(), None, model_history, gr.update()
            if audio_data == "loading_audio":
                yield chat_history, gr.update(), None, model_history, '<div style="text-align:center;padding:10px 0;color:#aaa;font-size:13px;animation:pulse 1.5s ease-in-out infinite">正在组装语音（受限于 Gradio 特性，此处为非流式解码）...</div>'
            elif audio_data:
                final_audio = audio_data
            if asr is not None:
                asr_text = asr

        user_text = asr_text if asr_text else (text or "")
        if user_text:
            model_history = model_history + [{"role": "user", "content": user_text}]
        if response_text:
            model_history = model_history + [{"role": "assistant", "content": response_text}]

        yield chat_history, final_audio if final_audio else gr.update(), None, model_history, ""

    with gr.Blocks(title="MiniMind-O", js="()=>{new MutationObserver(()=>{const m=document.getElementById('mic-box');if(!m)return;const h=!!m.querySelector('audio');document.body.classList.toggle('has-audio',h);const t=document.querySelector('textarea');if(t){t.placeholder=h?'已加载语音，点击发送':'输入文本';t.disabled=h}}).observe(document.body,{childList:true,subtree:true})}", css=".app{padding-top:6px!important} #component-0{gap:6px!important} #component-1{padding:2px 0!important;margin:0!important;min-height:0!important;border:none!important} #component-1 .padding{padding:0!important} #chatbox img{max-width:120px!important;max-height:120px!important;border-radius:8px} textarea{overflow-y:hidden!important;height:auto!important;min-height:30px!important;max-height:60px!important} #mic-box{max-height:150px!important;overflow:hidden!important} #mic-box .wrap span.or,#mic-box .wrap span:first-child{display:none!important} #mic-box .wrap{font-size:0!important;min-height:40px!important;padding:8px!important} #mic-box .wrap::after{content:'上传/录音';font-size:14px!important} #mic-box .mic-select{display:none!important} .has-audio textarea{opacity:0.4!important;pointer-events:none!important} .has-audio .upload-button,.has-audio [data-testid='upload-button']{display:none!important} @keyframes pulse{0%,100%{opacity:1}50%{opacity:0.4}}") as demo:
        gr.HTML('<div style="text-align:center;margin:2px 0"><span style="font-size:1.2rem;font-weight:bold;font-style:italic">MiniMind-O</span> <span style="color:#999;font-size:0.8rem">text / image / audio → text + audio</span></div>')

        chatbot = gr.Chatbot(label="", height=380, elem_id="chatbox", type="messages")
        model_history = gr.State([])
        audio_status = gr.HTML("", elem_id="audio-status")
        audio_out = gr.Audio(label="语音回复", autoplay=True, elem_id="audio-out")
        with gr.Row(equal_height=True):
            with gr.Column(scale=0, min_width=160):
                aud = gr.Audio(sources=["upload", "microphone"], type="numpy", show_label=False, elem_id="mic-box")
            with gr.Column(scale=4):
                msg = gr.MultimodalTextbox(placeholder="输入文本", show_label=False, submit_btn="发送")
        with gr.Row():
            voice_dd = gr.Dropdown(choices=voice_choices, value="default", label="音色选择", scale=0, min_width=140)
            turns_dd = gr.Dropdown(choices=[0, 2, 4, 6, 8], value=0, label="多轮记忆", scale=0, min_width=120)
            if model_dict:
                model_dd = gr.Dropdown(choices=list(model_dict.keys()), value=current_model_name, label="模型选择", scale=1, min_width=180)
                status = gr.Textbox(value=f"已加载: {current_model_name}", label="状态", interactive=False, scale=2)
                model_dd.change(lambda n: load_hf_model(model_dict[n]), [model_dd], [status])

        msg.submit(respond, [msg, aud, voice_dd, chatbot, model_history, turns_dd], [chatbot, audio_out, aud, model_history, audio_status])

    demo.queue().launch(server_name=server_name, server_port=server_port)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MiniMind-O Gradio Demo")
    parser.add_argument('--load_from', default='./', type=str, help="transformers模型扫描目录")
    parser.add_argument('--audio_encoder', default='../model/SenseVoiceSmall', type=str)
    parser.add_argument('--vision_model', default='../model/siglip2-base-p32-256-ve', type=str)
    parser.add_argument('--mimi_path', default='../model/mimi', type=str)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str)
    parser.add_argument('--open_thinking', default=0, type=int, choices=[0, 1])
    parser.add_argument('--top_p', default=0.85, type=float)
    parser.add_argument('--port', default=8888, type=int)
    args = parser.parse_args()

    device = args.device
    model_dict = scan_hf_models(args.load_from)
    if not model_dict:
        print(f"未在 {os.path.abspath(args.load_from)} 找到 transformers 模型")
        exit(1)
    current_model_name = list(model_dict.keys())[0]
    load_hf_model(model_dict[current_model_name])

    try:
        mimi_model = MimiModel.from_pretrained(args.mimi_path).eval()
        print('Mimi model loaded')
    except Exception:
        mimi_model = None
        print('Mimi model not found, audio output disabled')

    with contextlib.redirect_stdout(io.StringIO()):
        asr_model = AutoModel(model=args.audio_encoder, trust_remote_code=True, device=device, disable_update=True)
    load_voices()
    print(f'Voices loaded: {list(voices_data.keys()) or "none"}')
    launch_gradio(server_port=args.port)
