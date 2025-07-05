import os
import tempfile
import torch
import torchaudio
import json
import re
import ast
from email.mime.text import MIMEText
import smtplib
from google.cloud import storage
import google.generativeai as genai
from safetensors.torch import load_file
from pathlib import Path
from App.midi_model.midi_model import MIDIModel, MIDIModelConfig
from App.midi_model.midi_tokenizer import MIDITokenizerV1, MIDITokenizerV2
from App.midi_model.MIDI import score2midi, midi2score, score2opus



def load_model_from_best_checkpoint(path):
    from safetensors.torch import load_file as safe_load_file

    cfg_path = Path(path).parent / 'config.json'
    cfg = MIDIModelConfig.from_json_file(cfg_path)

    model = MIDIModel(config=cfg)
    tokenizer = model.tokenizer

    state = safe_load_file(path)
    model.load_state_dict(state, strict=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device, dtype=torch.bfloat16 if device == "cuda" else torch.float32)
    model.eval()
    return model, tokenizer


def generate_music_from_prompt_batch(prompt_dict, n=4):
    import numpy as np
    import torch
    from pathlib import Path
    from safetensors.torch import load_file as safe_load_file
    from App.midi_model.MIDI import score2midi, midi2score

    from App.midi_model.midi_synthesizer import MidiSynthesizer
    from huggingface_hub import hf_hub_download

    path = download_best_model_gcs()
    model, tokenizer = load_model_from_best_checkpoint(path)

    # Synthé audio
    sf = hf_hub_download('skytnt/midi-model', 'soundfont.sf2')
    synth = MidiSynthesizer(sf)

    # Préparation des paramètres
    instruments = prompt_dict.get("instruments", ["Acoustic Grand"])
    drum_kit = prompt_dict.get("drum_kit", "Standard")
    bpm = prompt_dict.get("bpm", 120)
    time_sig = prompt_dict.get("time_signature", "4/4")
    key_sig = prompt_dict.get("key_signature", "C")
    seed = prompt_dict.get("seed", 42)
    random_seed = prompt_dict.get("random_seed", True)
    duration_s = prompt_dict.get("duration_s", 30)
    temperature = prompt_dict.get("temperature", 1.0)
    top_p = prompt_dict.get("top_p", 0.9)
    top_k = prompt_dict.get("top_k", 20)
    allow_cc = prompt_dict.get("allow_cc", True)

    from streamlit_app import run_generation, chunk_and_pad

    EVENTS_PER_SECOND = 16
    gen_events = duration_s * EVENTS_PER_SECOND
    key_sig_idx = ["C","G","D","A","E","B","F#","C#","F","Bb","Eb","Ab","Db","Gb","Cb"].index(key_sig)

    mid_seq = run_generation(
        model=model,
        tokenizer=tokenizer,
        synth=synth,
        pool=None,
        instruments=instruments,
        drum_kit=drum_kit,
        bpm=bpm,
        time_sig=time_sig,
        key_sig_idx=key_sig_idx,
        seed=seed,
        random_seed=random_seed,
        gen_events=gen_events,
        temp=temperature,
        top_p=top_p,
        top_k=top_k,
        allow_cc=allow_cc,
        progress_gen=None  # Pas de Streamlit ici
    )

    # Sauvegarde des fichiers .wav
    from concurrent.futures import ThreadPoolExecutor
    import wave
    os.makedirs("media", exist_ok=True)
    urls, filenames = [], []
    group_id = str(abs(hash(str(prompt_dict))) % 1000000)

    for i, seq in enumerate(mid_seq):
        midi_seq = chunk_and_pad(seq, tokenizer.max_token_seq, tokenizer.pad_id)
        midi_data = tokenizer.detokenize(midi_seq)
        midi_bytes = score2midi(midi_data)
        audio = synth.synthesis(score2opus(midi_data))
        audio = (audio * np.iinfo(np.int16).max).astype(np.int16)
        filename = f"gen_{group_id}_v{i+1}.wav"
        filepath = os.path.join("media", filename)

        with wave.open(filepath, 'wb') as wf:
            wf.setnchannels(audio.shape[1] if audio.ndim > 1 else 1)
            wf.setsampwidth(2)
            wf.setframerate(synth.sample_rate)
            wf.writeframes(audio.tobytes())

        urls.append(f"/media/{filename}")
        filenames.append(filename)

    return group_id, urls, filenames



def extract_as_dict(text):
    try:
        return json.loads(text)
    except Exception:
        pass
    try:
        return ast.literal_eval(text)
    except Exception:
        pass
    matches = re.findall(r'\{[\s\S]*\}', text)
    if matches:
        raw = matches[0]
        try:
            return json.loads(raw)
        except Exception:
            pass
        try:
            return ast.literal_eval(raw)
        except Exception:
            pass
    return text


def prompt_to_config(prompt, model_name="gemini-1.5-pro-latest"):
    prompt2 = """
    Context:
    The project "midi-model" (https://github.com/SkyTNT/midi-model) is a toolkit for transformer-based training on MIDI files. Remove model_path, output_midi_path, add time.
    Task:
    Generate a JSON object (not markdown, not explanation), only raw JSON, for midi generation, with instruments, bpm, time, and all other necessary fields.
    """
    full_prompt = prompt2 + prompt
    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(full_prompt)
        return extract_as_dict(response.text)
    except Exception as e:
        print(f"Erreur lors du prompt-to-config Gemini : {e}")
        return None


def send_email(sender_email, sender_password, receiver_email, subject, body):
    try:
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = sender_email
        msg['To'] = receiver_email

        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, receiver_email, msg.as_bytes())

        print("📧 Email envoyé avec succès !")
    except Exception as e:
        print(f"❌ Erreur lors de l'envoi d'email : {e}")
