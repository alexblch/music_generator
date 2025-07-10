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
from App.midi_model.MIDI import score2midi, midi2score, score2opus, Number2patch
from openai import OpenAI
import logging
import numpy as np 
from tqdm import tqdm
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import wave
from pathlib import Path

logger = logging.getLogger(__name__)

# --- Mappings globaux ---
patch2number = {v: k for k, v in Number2patch.items()}
number2drum_kits = {
    -1: "None", 0: "Standard", 8: "Room", 16: "Power",
    24: "Electric", 25: "TR-808", 32: "Jazz", 40: "Blush", 48: "Orchestra"
}
drum_kits2number = {v: k for k, v in number2drum_kits.items()}

# Constantes
EVENTS_PER_SECOND = 16
BATCH_SIZE       = 4
AUDIO_DIR        = 'audio'
BUCKET           = 'pastorageesgi'
MODEL_BLOB       = 'models/bestmodel/model.safetensors'
CONFIG_BLOB      = 'models/bestmodel/config.json'


# --- Extraction des params via GPT function call ---
INSTRUMENTS      = list(patch2number.keys())
DRUM_KIT_LIST    = list(drum_kits2number.keys())
TIME_SIG_LIST    = ["auto","2/4","3/4","4/4","5/4","6/8","7/8","9/8","12/8"]
KEY_SIG_LIST     = ["C","G","D","A","E","B","F#","C#","F","Bb","Eb","Ab","Db","Gb","Cb"]

# --- GCS Utils ---
def download_from_gcs(bucket_name, blob_name, dest):
    logger.info(f"Downloading {blob_name} from GCS bucket {bucket_name} to {dest}")
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob   = bucket.blob(blob_name)
    blob.download_to_filename(dest)

def download_best_model_gcs():
    client = storage.Client()
    bucket = client.bucket("pastorageesgi")

    temp_dir = tempfile.mkdtemp()

    # Télécharger model.safetensors
    model_blob = bucket.blob("models/bestmodel/model.safetensors")
    local_model_path = os.path.join(temp_dir, "model.safetensors")
    model_blob.download_to_filename(local_model_path)

    # Télécharger config.json
    config_blob = bucket.blob("models/bestmodel/config.json")
    local_config_path = os.path.join(temp_dir, "config.json")
    config_blob.download_to_filename(local_config_path)

    print(f"✔️ Modèle & config téléchargés depuis GCS: models/bestmodel/")
    return local_model_path  # On retourne le chemin vers model.safetensors


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

def get_params_from_prompt(client: OpenAI, prompt: str) -> dict:
    logger.info("Requesting generation parameters from GPT-4…")
    system_msg = "Map a short musical description to valid MIDI generation parameters."
    functions = [
        {
            "name": "extract_music_params",
            "description": "Return MIDI generation parameters",
            "parameters": {
                "type": "object",
                "properties": {
                    "instruments":    {"type":"array","items":{"type":"string","enum":INSTRUMENTS}},
                    "drum_kit":       {"type":"string","enum":DRUM_KIT_LIST},
                    "bpm":            {"type":"integer","minimum":0,"maximum":255},
                    "time_signature": {"type":"string","enum":TIME_SIG_LIST},
                    "key_signature":  {"type":"string","enum":KEY_SIG_LIST},
                    "seed":           {"type":"integer"},
                    "random_seed":    {"type":"boolean"},
                    "duration_s":     {"type":"integer","minimum":1,"maximum":120},
                    "temperature":    {"type":"number","minimum":0.1,"maximum":1.2},
                    "top_p":          {"type":"number","minimum":0.1,"maximum":1.0},
                    "top_k":          {"type":"integer","minimum":1,"maximum":128},
                    "allow_cc":       {"type":"boolean"}
                },
                "required": [
                    "instruments","drum_kit","bpm","time_signature","key_signature",
                    "seed","random_seed","duration_s","temperature","top_p","top_k","allow_cc"
                ]
            }
        }
    ]
    resp = client.chat.completions.create(
        model="gpt-4-0613",
        messages=[
            {"role":"system","content":system_msg},
            {"role":"user",  "content":prompt}
        ],
        functions=functions,
        function_call={"name":"extract_music_params"},
        temperature=0.7
    )
    params = json.loads(resp.choices[0].message.function_call.arguments)
    logger.info(f"Received parameters: {params}")
    return params


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
        
        
# --- Copie de run_generation de main.py avec tqdm logs ---
@torch.inference_mode()
def generate(
    model, tokenizer, prompt, batch_size, max_len,
    temp, top_p, top_k, disable_patch_change,
    disable_control_change, disable_channels, generator,
    total_events
):
    # identical to main.py generate but yields blocks
    from transformers import DynamicCache
    import torch.nn.functional as F
    if disable_channels is not None:
        disable_channels = [tokenizer.parameter_ids["channel"][c] for c in disable_channels]
    else:
        disable_channels = []
    max_ts = tokenizer.max_token_seq
    input_tensor = torch.from_numpy(prompt[:, -max_ts:])\
                         .to(dtype=torch.long, device=model.device)\
                         .unsqueeze(1)
    cur_len = input_tensor.shape[1]
    cache1  = DynamicCache()
    past_len= 0

    logger.info("Starting MIDI token generation loop…")
    pbar = tqdm(total=total_events, desc="Generating MIDI events")
    while cur_len < max_len:
        hidden = model.forward(input_tensor[:, past_len:], cache=cache1)[:, -1]
        next_block = None
        event_names = [""] * batch_size
        cache2 = DynamicCache()
        for i in range(max_ts):
            mask = torch.zeros((batch_size, tokenizer.vocab_size),
                               dtype=torch.int64, device=model.device)
            for b in range(batch_size):
                if next_block is None:
                    mask_ids = list(tokenizer.event_ids.values()) + [tokenizer.eos_id]
                    if disable_patch_change:
                        mask_ids.remove(tokenizer.event_ids["patch_change"])
                    if disable_control_change:
                        mask_ids.remove(tokenizer.event_ids["control_change"])
                    mask[b, mask_ids] = 1
                else:
                    params = tokenizer.events[event_names[b]]
                    if i > len(params):
                        mask[b, tokenizer.pad_id] = 1
                        continue
                    pname = params[i-1]
                    ids   = tokenizer.parameter_ids[pname]
                    if pname=="channel":
                        ids = [c for c in ids if c not in disable_channels]
                    mask[b, ids] = 1
            
            x = None if next_block is None else next_block[:, -1:]
            logits = model.forward_token(hidden, x, cache=cache2)[:, -1:]
            scores = torch.softmax(logits / temp, dim=-1) * mask.unsqueeze(1)
            samples = model.sample_top_p_k(scores, top_p, top_k, generator=generator)

            if next_block is None:
                next_block = samples
                for b in range(batch_size):
                    eid = samples[b].item()
                    event_names[b] = None if eid==tokenizer.eos_id else tokenizer.id_events[eid]
            else:
                next_block = torch.cat([next_block, samples], dim=1)
                if all((event_names[b] is None) or
                       (len(tokenizer.events[event_names[b]])==next_block.shape[1]-1)
                       for b in range(batch_size)):
                    break
        if next_block.shape[1] < max_ts:
            pad_amt = max_ts - next_block.shape[1]
            next_block = F.pad(next_block, (0,pad_amt), value=tokenizer.pad_id)

        next_block = next_block.unsqueeze(1)
        input_tensor = torch.cat([input_tensor, next_block], dim=1)
        past_len = cur_len
        cur_len  += 1
        pbar.update(1)
        yield next_block[:,0].cpu().numpy()
    pbar.close()


def run_generation(
    model, tokenizer, synth, pool,
    instruments, drum_kit, bpm, time_sig, key_sig_idx,
    seed, random_seed, gen_events,
    temp, top_p, top_k, allow_cc
):
    logger.info("Running generation with parameters")
    # Build patches
    patches = {i: patch2number[instr] for i,instr in enumerate(instruments)}
    if drum_kit!="None": patches[9] = drum_kits2number[drum_kit]
    # Build initial sequence
    max_ts = tokenizer.max_token_seq
    base = [tokenizer.bos_id] + [tokenizer.pad_id]*(max_ts-1)
    init = base.copy()
    if tokenizer.version=="v2" and time_sig!="auto":
        nn,dd = map(int,time_sig.split('/'))
        init += tokenizer.event2tokens(["time_signature",0,0,0,nn-1,{2:1,4:2,8:3}[dd]])
    if tokenizer.version=="v2" and key_sig_idx>0:
        sf=(key_sig_idx-1)//2-7; mi=(key_sig_idx-1)%2
        init += tokenizer.event2tokens(["key_signature",0,0,0,sf+7,mi])
    if bpm>0: init += tokenizer.event2tokens(["set_tempo",0,0,0,bpm])
    for idx,(c,p) in enumerate(patches.items()):
        init += tokenizer.event2tokens(["patch_change",0,0,idx+1,c,p])
    prompt = np.array([init]*BATCH_SIZE, dtype=np.int64)
    # RNG
    gen = torch.Generator(device=model.device)
    seed_val = random.randint(0,2**31-1) if random_seed else seed
    gen.manual_seed(seed_val)
    # Generate
    seqs = [seq.copy() for seq in prompt.tolist()]
    logger.info("Starting block-wise generation…")
    for block in generate(
        model, tokenizer, prompt, BATCH_SIZE,
        len(init)+gen_events, temp, top_p, top_k,
        disable_patch_change=bool(patches),
        disable_control_change=not allow_cc,
        disable_channels=[c for c in range(16) if c not in patches],
        generator=gen,
        total_events=gen_events
    ):
        for i in range(len(seqs)): seqs[i].extend(block[i].tolist())
    logger.info("Generation complete.")
    return seqs



def synthesize_audios_django(
    midi_paths: list[str],
    synth,
    pool,
    duration_s: int,
    output_dir: str
) -> list[str]:
    """
    Prend une liste de chemins MIDI locaux,
    synthétise en WAV (tronçonnage à duration_s secondes),
    écrit dans output_dir, et renvoie la liste de fichiers WAV générés.
    """
    os.makedirs(output_dir, exist_ok=True)
    futures = {}
    for idx, path in enumerate(midi_paths, start=1):
        with open(path, 'rb') as f:
            midi_bytes = f.read()
        score     = midi2score(midi_bytes)
        midi_opus = score2opus(score)
        futures[pool.submit(synth.synthesis, midi_opus)] = idx

    wav_filenames = []
    for fut in as_completed(futures):
        idx = futures[fut]
        pcm = fut.result()       # np.ndarray
        sr  = synth.sample_rate  # e.g. 44100

        # Tronçonnage à duration_s secondes
        max_samples = int(duration_s * sr)
        pcm = pcm[:max_samples] if pcm.ndim == 1 else pcm[:max_samples, :]

        # Passage en int16 si nécessaire
        if pcm.dtype == np.float32:
            pcm = (pcm * np.iinfo(np.int16).max).astype(np.int16)

        # Nombre de canaux
        if pcm.ndim == 1:
            nchannels = 1
        else:
            # transpose si besoin
            if pcm.shape[0] < pcm.shape[1]:
                pcm = pcm.T
            nchannels = pcm.shape[1]

        wav_name = f"out_{idx}.wav"
        wav_path = os.path.join(output_dir, wav_name)
        with wave.open(wav_path, 'wb') as wf:
            wf.setnchannels(nchannels)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(pcm.tobytes())

        wav_filenames.append(wav_name)

    return wav_filenames


def upload_to_gcs(local_path, bucket_name, blob_name):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_path)

    
    
def extract_index(filename):
    match = re.search(r"out_(\d+)\.wav", filename)
    return int(match.group(1)) if match else float('inf')