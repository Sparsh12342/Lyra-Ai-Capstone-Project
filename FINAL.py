#!/usr/bin/env python3
"""
Lyra-TF-Pipeline (TensorFlow + PyTorch demo)

End-to-end script that:
  1) Fetches and cleans a YouTube transcript
  2) Translates text using a TensorFlow NMT model (served via TF Serving OR local SavedModel)
  3) Separates a song into vocals/instrumental with Demucs (PyTorch)
  4) Detects vocal activity windows (VAD) with librosa
  5) Synthesizes speech with a TensorFlow TTS model (FastSpeech2-like) using a duration target per segment
  6) Aligns synthesized speech into the detected vocal windows
  7) Mixes aligned TTS with the instrumental to produce a final track

Assumptions / Notes:
- You already have exported TensorFlow SavedModels for:
    * NMT (translation) — signature takes a batch of strings and returns translated strings
    * TTS (acoustic+vocoder) — signature takes "text" and optional "target_duration_s" and returns mono audio float32 and sample_rate
- Alternatively, you are serving them behind TF Serving REST endpoints.
- This is a scaffold; adapt the input/output keys to match your actual SavedModel signatures.
- Demucs is used only for source separation (PyTorch). Everything else is TF / classic Python DSP.

Author: you
"""

import os
import re
import json
import math
import time
import argparse
from typing import List, Tuple, Optional

import numpy as np
import soundfile as sf
import librosa
import requests

# TensorFlow for local SavedModel loading (optional)
import tensorflow as tf

# PyTorch + Demucs for source separation
import torch
from demucs.pretrained import get_model as demucs_get_model
from demucs.apply import apply_model as demucs_apply_model
from demucs.audio import AudioFile as DemucsAudioFile, save_audio as demucs_save_audio

from youtube_transcript_api import YouTubeTranscriptApi

# ------------------------------
# Configuration
# ------------------------------
USE_TF_SERVING_NMT = True
USE_TF_SERVING_TTS = True

# If using TF Serving, set endpoints here.
NMT_SERVING_URL = os.environ.get("NMT_SERVING_URL", "http://localhost:8501/v1/models/nmt:predict")
TTS_SERVING_URL = os.environ.get("TTS_SERVING_URL", "http://localhost:8502/v1/models/tts:predict")

# If loading local SavedModels, set paths here.
NMT_SAVEDMODEL_DIR = os.environ.get("NMT_SAVEDMODEL_DIR", "./saved_models/nmt")
TTS_SAVEDMODEL_DIR = os.environ.get("TTS_SAVEDMODEL_DIR", "./saved_models/tts")

# Demucs model name (downloaded on first use)
DEMUCS_MODEL_NAME = os.environ.get("DEMUCS_MODEL_NAME", "htdemucs")

# Output sampling rate for TTS (if model doesn't specify)
DEFAULT_TTS_SR = 22050

# ------------------------------
# Utilities
# ------------------------------

def remove_music_markers(text: str) -> str:
    markers = ["[Music]", "♪", "[music]", "[Música]", "[música]"]
    for m in markers:
        text = text.replace(m, "")
    return text


def fetch_youtube_transcript(video_id: str) -> Optional[str]:
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        joined = " ".join([seg["text"] for seg in transcript])
        return remove_music_markers(joined).strip()
    except Exception as e:
        print(f"[ERR] Could not fetch transcript: {e}")
        return None


# ------------------------------
# NMT: TensorFlow (Serving or local SavedModel)
# ------------------------------
class NMTTranslator:
    def __init__(self, use_serving: bool, serving_url: str = None, savedmodel_dir: str = None):
        self.use_serving = use_serving
        self.serving_url = serving_url
        self.model = None
        if not use_serving:
            if savedmodel_dir is None:
                raise ValueError("savedmodel_dir required when not using TF Serving for NMT")
            print(f"[NMT] Loading local SavedModel from: {savedmodel_dir}")
            self.model = tf.saved_model.load(savedmodel_dir)
            self.infer = self.model.signatures.get("serving_default")
            if self.infer is None:
                raise RuntimeError("NMT SavedModel missing 'serving_default' signature")

    def translate(self, texts: List[str]) -> List[str]:
        if self.use_serving:
            payload = {"instances": [{"inputs": t} for t in texts]}
            resp = requests.post(self.serving_url, json=payload, timeout=120)
            if resp.status_code != 200:
                raise RuntimeError(f"NMT Serving error {resp.status_code}: {resp.text}")
            data = resp.json()
            # Expecting {"predictions": ["..."]} or objects with "outputs"
            preds = data.get("predictions", [])
            # Normalize to list[str]
            out = []
            for p in preds:
                if isinstance(p, dict):
                    out.append(p.get("outputs", ""))
                else:
                    out.append(str(p))
            return out
        else:
            # Local SavedModel: accept a tf.string tensor batch
            inputs = tf.constant(texts, dtype=tf.string)
            outputs = self.infer(inputs=inputs)
            # The output key depends on your signature; try common names
            for key in ("outputs", "translations", "predictions", "output_0"):
                if key in outputs:
                    arr = outputs[key].numpy()
                    return [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in arr]
            raise RuntimeError("NMT SavedModel outputs not recognized; adjust keys in code.")


# ------------------------------
# TTS: TensorFlow (Serving or local SavedModel)
# ------------------------------
class TTSSynthesizer:
    def __init__(self, use_serving: bool, serving_url: str = None, savedmodel_dir: str = None):
        self.use_serving = use_serving
        self.serving_url = serving_url
        self.model = None
        self.infer = None
        if not use_serving:
            if savedmodel_dir is None:
                raise ValueError("savedmodel_dir required when not using TF Serving for TTS")
            print(f"[TTS] Loading local SavedModel from: {savedmodel_dir}")
            self.model = tf.saved_model.load(savedmodel_dir)
            self.infer = self.model.signatures.get("serving_default")
            if self.infer is None:
                raise RuntimeError("TTS SavedModel missing 'serving_default' signature")

    def synthesize(self, text: str, target_duration_s: Optional[float] = None) -> Tuple[np.ndarray, int]:
        """
        Returns: (audio_float32_mono, sample_rate)
        Expected model behaviors (adjust to your signatures):
          - Inputs: {"text": <string>, "target_duration_s": <float> (optional)}
          - Outputs: {"audio": float32[time], "sample_rate": int}
        """
        if self.use_serving:
            instance = {"text": text}
            if target_duration_s is not None:
                instance["target_duration_s"] = float(target_duration_s)
            payload = {"instances": [instance]}
            resp = requests.post(self.serving_url, json=payload, timeout=300)
            if resp.status_code != 200:
                raise RuntimeError(f"TTS Serving error {resp.status_code}: {resp.text}")
            data = resp.json()
            pred = data.get("predictions", [])[0]
            if isinstance(pred, dict):
                audio = np.array(pred.get("audio", []), dtype=np.float32)
                sr = int(pred.get("sample_rate", DEFAULT_TTS_SR))
            else:
                # If server returns a flat list with a parallel key in outputs
                audio = np.array(pred, dtype=np.float32)
                sr = DEFAULT_TTS_SR
            return audio, sr
        else:
            inputs = {"text": tf.constant([text])}
            if target_duration_s is not None:
                inputs["target_duration_s"] = tf.constant([float(target_duration_s)], dtype=tf.float32)
            outputs = self.infer(**inputs)
            # Normalize keys per your SavedModel
            audio_key = None
            for k in ("audio", "waveform", "output_0"):
                if k in outputs:
                    audio_key = k
                    break
            if audio_key is None:
                raise RuntimeError("TTS SavedModel: couldn't find audio in outputs. Adjust keys.")
            audio = outputs[audio_key].numpy()[0].astype(np.float32)
            sr = DEFAULT_TTS_SR
            for k in ("sample_rate", "sr"):
                if k in outputs:
                    sr = int(outputs[k].numpy()[0])
            return audio, sr


# ------------------------------
# Demucs separation (PyTorch)
# ------------------------------

def demucs_separate(input_path: str, output_dir: str = "separated_output", model_name: str = DEMUCS_MODEL_NAME) -> Tuple[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(input_path))[0]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Demucs] Loading {model_name} on {device}")
    model = demucs_get_model(model_name)
    model.to(device)

    print(f"[Demucs] Reading: {input_path}")
    wav = DemucsAudioFile(input_path).read(streams=0, samplerate=model.samplerate, channels=model.audio_channels)
    wav = wav.unsqueeze(0).to(device)

    with torch.no_grad():
        print("[Demucs] Separating sources…")
        sources = demucs_apply_model(model, wav)
    sources = sources.cpu()

    if "vocals" not in model.sources:
        raise RuntimeError("Demucs model does not expose a 'vocals' stem")

    vocals_idx = model.sources.index("vocals")

    vocals = sources[0, vocals_idx]
    instrumental = torch.zeros_like(sources[0, 0])
    for i, name in enumerate(model.sources):
        if i != vocals_idx:
            instrumental += sources[0, i]

    vocals_path = os.path.join(output_dir, f"{base}_vocals.wav")
    instr_path = os.path.join(output_dir, f"{base}_instrumental.wav")

    demucs_save_audio(vocals, vocals_path, model.samplerate)
    demucs_save_audio(instrumental, instr_path, model.samplerate)

    print(f"[Demucs] Saved vocals: {vocals_path}")
    print(f"[Demucs] Saved instrumental: {instr_path}")
    return vocals_path, instr_path


# ------------------------------
# VAD + alignment utilities
# ------------------------------

def detect_vocal_segments(vocals_path: str, threshold: float = 0.01, min_duration: float = 0.05, gap_tolerance: float = 0.2) -> List[Tuple[float, float]]:
    y, sr = librosa.load(vocals_path, mono=True)
    rms = librosa.feature.rms(y=y)[0]
    # Convert frame index to time. rms uses hop_length=512 by default.
    hop_length = 512
    time_per_frame = hop_length / sr

    segments: List[Tuple[float, float]] = []
    in_seg = False
    start_t = 0.0
    last_end = 0.0

    for i, energy in enumerate(rms):
        t = i * time_per_frame
        if energy > threshold and not in_seg:
            # new or continuation within gap
            if last_end > 0 and (t - last_end) <= gap_tolerance:
                in_seg = True
            else:
                start_t = t
                in_seg = True
        elif energy <= threshold and in_seg:
            end_t = t
            if (end_t - start_t) >= min_duration:
                segments.append((start_t, end_t))
                last_end = end_t
            in_seg = False

    if in_seg:
        end_t = len(rms) * time_per_frame
        if (end_t - start_t) >= min_duration:
            segments.append((start_t, end_t))

    # Merge small gaps
    merged: List[Tuple[float, float]] = []
    if segments:
        cur_s, cur_e = segments[0]
        for s, e in segments[1:]:
            if s - cur_e <= gap_tolerance:
                cur_e = e
            else:
                merged.append((cur_s, cur_e))
                cur_s, cur_e = s, e
        merged.append((cur_s, cur_e))
    return merged


def split_text_to_segments(text: str, n_segments: int) -> List[str]:
    """Naively split text into n roughly-equal sentence chunks."""
    # Split on sentence boundaries
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    if not sentences:
        return [text]
    # If fewer sentences than segments, pad by merging spaces
    chunks: List[str] = []
    # Greedy pack sentences into n chunks by length
    target = max(1, len(" ".join(sentences)) // max(1, n_segments))
    cur = ""
    for s in sentences:
        if len(cur) + len(s) + 1 <= target or len(chunks) + 1 == n_segments:
            cur = (cur + " " + s).strip()
        else:
            chunks.append(cur)
            cur = s
    if cur:
        chunks.append(cur)
    # Adjust to exactly n_segments
    if len(chunks) > n_segments:
        # merge extras into last
        head, tail = chunks[:n_segments-1], chunks[n_segments-1:]
        chunks = head + [" ".join(tail)]
    elif len(chunks) < n_segments:
        # pad with empty strings
        chunks += [""] * (n_segments - len(chunks))
    return chunks


def place_audio_segments(template_len: int, sr: int, segments: List[Tuple[float, float]], audio_chunks: List[np.ndarray]) -> np.ndarray:
    out = np.zeros(template_len, dtype=np.float32)
    for (start_s, end_s), chunk in zip(segments, audio_chunks):
        a = int(start_s * sr)
        b = int(end_s * sr)
        seg_len = b - a
        if seg_len <= 0:
            continue
        if len(chunk) > seg_len:
            chunk = librosa.util.fix_length(chunk, seg_len, mode='edge')[:seg_len]
        elif len(chunk) < seg_len:
            chunk = librosa.util.fix_length(chunk, seg_len)
        out[a:b] = chunk
    # normalize if needed
    mx = np.max(np.abs(out))
    if mx > 1.0:
        out = out / mx
    return out


# ------------------------------
# Mixing utilities
# ------------------------------

def resample_audio(data: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return data
    return librosa.resample(y=data, orig_sr=orig_sr, target_sr=target_sr)


def mix_tracks(vocals: np.ndarray, instrumental: np.ndarray, sr: int, vocals_vol: float = 0.7, instr_vol: float = 0.5) -> np.ndarray:
    # Ensure stereo for both
    def to_stereo(x):
        if x.ndim == 1:
            return np.stack([x, x], axis=-1)
        return x

    v = to_stereo(vocals)
    i = to_stereo(instrumental)

    # Match lengths
    n = min(v.shape[0], i.shape[0])
    v = v[:n]
    i = i[:n]

    mix = vocals_vol * v + instr_vol * i
    mx = np.max(np.abs(mix))
    if mx > 1.0:
        mix = mix / mx
    return mix


# ------------------------------
# Main orchestration
# ------------------------------

def run_pipeline(
    video_id: str,
    song_path: str,
    target_lang_hint: str = "en",
    vocals_vol: float = 0.7,
    instr_vol: float = 0.5,
    out_dir: str = "outputs",
):
    os.makedirs(out_dir, exist_ok=True)

    # 1) Transcript
    print("[1/7] Fetching transcript…")
    transcript = fetch_youtube_transcript(video_id)
    if not transcript:
        raise SystemExit("Transcript fetch failed.")

    # 2) Translation via NMT (TF)
    print("[2/7] Translating text with TensorFlow NMT…")
    nmt = NMTTranslator(USE_TF_SERVING_NMT, NMT_SERVING_URL, NMT_SAVEDMODEL_DIR)
    translated_list = nmt.translate([transcript])
    translated = translated_list[0] if translated_list else ""
    if not translated:
        raise SystemExit("NMT produced empty output.")

    # 3) Separate song with Demucs
    print("[3/7] Separating song with Demucs…")
    vocals_wav, instr_wav = demucs_separate(song_path, output_dir=os.path.join(out_dir, "separated"))

    # 4) Detect vocal segments
    print("[4/7] Detecting vocal segments…")
    segments = detect_vocal_segments(vocals_wav)
    print(f"[VAD] {len(segments)} segments detected")

    # 5) TTS synth per segment with duration targets
    print("[5/7] Synthesizing TTS with TensorFlow TTS…")
    tts = TTSSynthesizer(USE_TF_SERVING_TTS, TTS_SERVING_URL, TTS_SAVEDMODEL_DIR)

    # Load reference to get total length & sr
    ref_y, ref_sr = librosa.load(vocals_wav, mono=True)

    # Split translated text into len(segments) chunks
    chunks = split_text_to_segments(translated, len(segments))

    tts_chunks: List[np.ndarray] = []
    for (start_s, end_s), text_chunk in zip(segments, chunks):
        dur = max(0.05, end_s - start_s)
        # Synthesize with a target duration; model should pace output accordingly
        audio, sr = tts.synthesize(text_chunk, target_duration_s=dur)
        if sr != ref_sr:
            audio = resample_audio(audio, sr, ref_sr)
            sr = ref_sr
        # Loudness trim
        if np.max(np.abs(audio)) > 0:
            audio = audio / max(1.0, np.max(np.abs(audio))) * 0.8
        tts_chunks.append(audio.astype(np.float32))

    # 6) Place TTS chunks into the vocal timeline
    print("[6/7] Aligning TTS to vocal timeline…")
    placed = place_audio_segments(len(ref_y), ref_sr, segments, tts_chunks)

    # Save aligned TTS mono
    aligned_tts_path = os.path.join(out_dir, "aligned_tts.wav")
    sf.write(aligned_tts_path, placed, ref_sr)
    print(f"[SAVE] Aligned TTS: {aligned_tts_path}")

    # 7) Mix with instrumental
    print("[7/7] Mixing aligned TTS with instrumental…")
    instr_y, instr_sr = librosa.load(instr_wav, mono=False)  # keep stereo if possible
    if instr_y.ndim == 1:
        instr_y = np.stack([instr_y, instr_y], axis=-1)
    if instr_sr != ref_sr:
        # Resample each channel
        instr_y = np.stack([resample_audio(instr_y[:, ch], instr_sr, ref_sr) for ch in range(instr_y.shape[1])], axis=-1)
        instr_sr = ref_sr

    final_mix = mix_tracks(placed, instr_y, ref_sr, vocals_vol=vocals_vol, instr_vol=instr_vol)
    final_out = os.path.join(out_dir, "final_translated_song.wav")
    sf.write(final_out, final_mix, ref_sr)
    print(f"[DONE] Final mix saved: {final_out}")


def parse_args():
    ap = argparse.ArgumentParser(description="Lyra TF Pipeline")
    ap.add_argument("--video_id", required=True, help="YouTube video ID with captions")
    ap.add_argument("--song", required=True, help="Path to the original song file (.mp3/.wav)")
    ap.add_argument("--vocals_vol", type=float, default=0.7)
    ap.add_argument("--instr_vol", type=float, default=0.5)
    ap.add_argument("--out", default="outputs")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        video_id=args.video_id,
        song_path=args.song,
        vocals_vol=args.vocals_vol,
        instr_vol=args.instr_vol,
        out_dir=args.out,
    )
