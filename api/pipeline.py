

import os
import re
from typing import List, Tuple, Optional

import numpy as np
import soundfile as sf
import librosa
import requests

# ----- TensorFlow is OPTIONAL (only for local SavedModels). Do not import by default.
try:
    import tensorflow as tf  # noqa: F401
except Exception:
    tf = None  # type: ignore

# PyTorch + Demucs for source separation
import torch
from demucs.pretrained import get_model as demucs_get_model
from demucs.apply import apply_model as demucs_apply_model
from demucs.audio import AudioFile as DemucsAudioFile, save_audio as demucs_save_audio

from youtube_transcript_api import YouTubeTranscriptApi
import yt_dlp


# ------------------------------
# Configuration
# ------------------------------
USE_TF_SERVING_NMT = True
USE_TF_SERVING_TTS = True

# If using TF Serving, set endpoints here.
NMT_SERVING_URL = os.environ.get("NMT_SERVING_URL", "http://localhost:8501/v1/models/nmt:predict")
TTS_SERVING_URL = os.environ.get("TTS_SERVING_URL", "http://localhost:8502/v1/models/tts:predict")

# If loading local SavedModels, set paths here (only used when USE_TF_SERVING_* = False)
NMT_SAVEDMODEL_DIR = os.environ.get("NMT_SAVEDMODEL_DIR", "./saved_models/nmt")
TTS_SAVEDMODEL_DIR = os.environ.get("TTS_SAVEDMODEL_DIR", "./saved_models/tts")

# Demucs model name (downloaded on first use)
DEMUCS_MODEL_NAME = os.environ.get("DEMUCS_MODEL_NAME", "htdemucs")

# Output sampling rate for TTS (fallback if model/server doesn’t specify)
DEFAULT_TTS_SR = 22050

# Default translation language hint (overridable via API)
_TARGET_LANG_HINT = os.environ.get("TARGET_LANG_HINT", "en")


def set_target_language_hint(lang: str):
    """Set the target language hint used for NMT Serving payloads."""
    global _TARGET_LANG_HINT
    _TARGET_LANG_HINT = (lang or "en").strip()


# ------------------------------
# Utilities
# ------------------------------
def remove_music_markers(text: str) -> str:
    for m in ["[Music]", "♪", "[music]", "[Música]", "[música]"]:
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


def parse_video_id_from_url(url: str) -> str:
    """Extract the 11-char YouTube ID from common URL forms."""
    m = re.search(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})", url)
    if not m:
        raise ValueError("Could not parse YouTube video ID from URL.")
    return m.group(1)


# ------------------------------
# NMT: TensorFlow (Serving or local SavedModel)
# ------------------------------
class NMTTranslator:
    def __init__(self, use_serving: bool, serving_url: Optional[str] = None, savedmodel_dir: Optional[str] = None):
        self.use_serving = use_serving
        self.serving_url = serving_url
        self.infer = None

        if not use_serving:
            if tf is None:
                raise RuntimeError(
                    "TensorFlow is not installed. Either set USE_TF_SERVING_NMT=True or install tensorflow."
                )
            if not savedmodel_dir:
                raise ValueError("savedmodel_dir required when not using TF Serving for NMT")
            print(f"[NMT] Loading local SavedModel from: {savedmodel_dir}")
            model = tf.saved_model.load(savedmodel_dir)
            self.infer = model.signatures.get("serving_default")
            if self.infer is None:
                raise RuntimeError("NMT SavedModel missing 'serving_default' signature")

    def translate(self, texts: List[str]) -> List[str]:
        if self.use_serving:
            # Include a target language hint if your server supports it
            payload = {"instances": [{"inputs": t, "target_lang": _TARGET_LANG_HINT} for t in texts]}
            resp = requests.post(self.serving_url, json=payload, timeout=120)
            if resp.status_code != 200:
                raise RuntimeError(f"NMT Serving error {resp.status_code}: {resp.text}")
            data = resp.json()
            preds = data.get("predictions", [])
            out = []
            for p in preds:
                if isinstance(p, dict):
                    out.append(p.get("outputs", ""))
                else:
                    out.append(str(p))
            return out
        else:
            inputs = tf.constant(texts, dtype=tf.string)  # type: ignore
            outputs = self.infer(inputs=inputs)  # type: ignore
            for key in ("outputs", "translations", "predictions", "output_0"):
                if key in outputs:
                    arr = outputs[key].numpy()
                    return [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in arr]
            raise RuntimeError("NMT SavedModel outputs not recognized; adjust keys.")


# ------------------------------
# TTS: TensorFlow (Serving or local SavedModel)
# ------------------------------
class TTSSynthesizer:
    def __init__(self, use_serving: bool, serving_url: Optional[str] = None, savedmodel_dir: Optional[str] = None):
        self.use_serving = use_serving
        self.serving_url = serving_url
        self.infer = None

        if not use_serving:
            if tf is None:
                raise RuntimeError(
                    "TensorFlow is not installed. Either set USE_TF_SERVING_TTS=True or install tensorflow."
                )
            if not savedmodel_dir:
                raise ValueError("savedmodel_dir required when not using TF Serving for TTS")
            print(f"[TTS] Loading local SavedModel from: {savedmodel_dir}")
            model = tf.saved_model.load(savedmodel_dir)  # type: ignore
            self.infer = model.signatures.get("serving_default")
            if self.infer is None:
                raise RuntimeError("TTS SavedModel missing 'serving_default' signature")

    def synthesize(self, text: str, target_duration_s: Optional[float] = None) -> Tuple[np.ndarray, int]:
        """
        Returns: (audio_float32_mono, sample_rate)
        Expected server/model behaviors (adjust to your signatures):
          - Inputs: {"text": <string>, "target_duration_s": <float> (optional)}
          - Outputs: {"audio": float32[time], "sample_rate": int}
        """
        if self.use_serving:
            instance = {"text": text}
            if target_duration_s is not None:
                instance["target_duration_s"] = float(target_duration_s)
            resp = requests.post(self.serving_url, json={"instances": [instance]}, timeout=300)
            if resp.status_code != 200:
                raise RuntimeError(f"TTS Serving error {resp.status_code}: {resp.text}")
            pred = resp.json().get("predictions", [])[0]
            if isinstance(pred, dict):
                audio = np.array(pred.get("audio", []), dtype=np.float32)
                sr = int(pred.get("sample_rate", DEFAULT_TTS_SR))
            else:
                audio = np.array(pred, dtype=np.float32)
                sr = DEFAULT_TTS_SR
            return audio, sr
        else:
            inputs = {"text": tf.constant([text])}  # type: ignore
            if target_duration_s is not None:
                inputs["target_duration_s"] = tf.constant([float(target_duration_s)], dtype=tf.float32)  # type: ignore
            outputs = self.infer(**inputs)  # type: ignore

            audio_key = next((k for k in ("audio", "waveform", "output_0") if k in outputs), None)
            if audio_key is None:
                raise RuntimeError("TTS SavedModel: couldn't find audio in outputs. Adjust keys.")
            audio = outputs[audio_key].numpy()[0].astype(np.float32)

            sr = DEFAULT_TTS_SR
            for k in ("sample_rate", "sr"):
                if k in outputs:
                    try:
                        sr = int(outputs[k].numpy()[0])
                    except Exception:
                        pass
            return audio, sr


# ------------------------------
# Demucs separation (PyTorch)
# ------------------------------
def demucs_separate(input_path: str, output_dir: str) -> Tuple[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(input_path))[0]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Demucs] Loading {DEMUCS_MODEL_NAME} on {device}")
    model = demucs_get_model(DEMUCS_MODEL_NAME)
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

    out_vocals = os.path.join(output_dir, f"{base}_vocals.wav")
    out_instr  = os.path.join(output_dir, f"{base}_instrumental.wav")
    demucs_save_audio(vocals, out_vocals, model.samplerate)
    demucs_save_audio(instrumental, out_instr, model.samplerate)

    print(f"[Demucs] Saved vocals: {out_vocals}")
    print(f"[Demucs] Saved instrumental: {out_instr}")
    return out_vocals, out_instr


# ------------------------------
# VAD + alignment utilities
# ------------------------------
def detect_vocal_segments(vocals_path: str, threshold: float = 0.01, min_duration: float = 0.05, gap_tolerance: float = 0.2) -> List[Tuple[float, float]]:
    y, sr = librosa.load(vocals_path, mono=True)
    rms = librosa.feature.rms(y=y)[0]
    hop_length = 512  # librosa default inside rms
    time_per_frame = hop_length / sr

    segments: List[Tuple[float, float]] = []
    in_seg = False
    start_t = 0.0
    last_end = 0.0

    for i, energy in enumerate(rms):
        t = i * time_per_frame
        if energy > threshold and not in_seg:
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
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    if not sentences:
        return [text]
    chunks: List[str] = []
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
    if len(chunks) > n_segments:
        head, tail = chunks[:n_segments-1], chunks[n_segments-1:]
        chunks = head + [" ".join(tail)]
    elif len(chunks) < n_segments:
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
    def to_stereo(x):
        if x.ndim == 1:
            return np.stack([x, x], axis=-1)
        return x

    v = to_stereo(vocals)
    i = to_stereo(instrumental)

    n = min(v.shape[0], i.shape[0])
    v = v[:n]
    i = i[:n]

    mix = vocals_vol * v + instr_vol * i
    mx = np.max(np.abs(mix))
    if mx > 1.0:
        mix = mix / mx
    return mix


# ------------------------------
# Download audio
# ------------------------------
def download_best_audio(youtube_url: str, out_dir: str) -> str:
    """Download best audio and convert to WAV via ffmpeg (requires ffmpeg on PATH)."""
    os.makedirs(out_dir, exist_ok=True)
    outtmpl = os.path.join(out_dir, "%(id)s.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": outtmpl,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "postprocessors": [
            {"key": "FFmpegExtractAudio", "preferredcodec": "wav", "preferredquality": "192"}
        ],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=True)
        video_id = info.get("id")
    wav_path = os.path.join(out_dir, f"{video_id}.wav")
    if not os.path.exists(wav_path):
        raise RuntimeError("Failed to download/convert audio (is ffmpeg installed?).")
    return wav_path


# ------------------------------
# Main orchestration (YouTube URL)
# ------------------------------
def run_pipeline_from_youtube(
    youtube_url: str,
    vocals_vol: float = 0.7,
    instr_vol: float = 0.5,
    out_dir: str = "outputs",
) -> str:
    """
    End-to-end from a YouTube URL. Returns path to final WAV.
    """
    os.makedirs(out_dir, exist_ok=True)
    video_id = parse_video_id_from_url(youtube_url)

    # 1) Transcript
    print("[1/8] Fetching transcript…")
    transcript = fetch_youtube_transcript(video_id)
    if not transcript:
        raise RuntimeError("Transcript fetch failed.")

    # 2) Translation via NMT (Serving-first)
    print("[2/8] Translating text…")
    nmt = NMTTranslator(USE_TF_SERVING_NMT, NMT_SERVING_URL, NMT_SAVEDMODEL_DIR)
    translated = nmt.translate([transcript])[0]
    if not translated:
        raise RuntimeError("NMT produced empty output.")

    # 3) Download audio
    print("[3/8] Downloading audio…")
    dl_dir = os.path.join(out_dir, "download")
    song_wav = download_best_audio(youtube_url, dl_dir)

    # 4) Separate song with Demucs
    print("[4/8] Separating with Demucs…")
    vocals_wav, instr_wav = demucs_separate(song_wav, output_dir=os.path.join(out_dir, "separated"))

    # 5) Detect vocal segments
    print("[5/8] Detecting vocal segments…")
    segments = detect_vocal_segments(vocals_wav)
    print(f"[VAD] {len(segments)} segments detected")

    # 6) TTS per segment (Serving-first)
    print("[6/8] Synthesizing TTS…")
    tts = TTSSynthesizer(USE_TF_SERVING_TTS, TTS_SERVING_URL, TTS_SAVEDMODEL_DIR)

    ref_y, ref_sr = librosa.load(vocals_wav, mono=True)
    chunks_txt = split_text_to_segments(translated, len(segments))

    tts_chunks: List[np.ndarray] = []
    for (start_s, end_s), txt in zip(segments, chunks_txt):
        dur = max(0.05, end_s - start_s)
        audio, sr = tts.synthesize(txt, target_duration_s=dur)
        if sr != ref_sr:
            audio = resample_audio(audio, sr, ref_sr)
        if np.max(np.abs(audio)) > 0:
            audio = audio / max(1.0, np.max(np.abs(audio))) * 0.8
        tts_chunks.append(audio.astype(np.float32))

    # 7) Align into timeline
    print("[7/8] Aligning TTS to vocal timeline…")
    placed = place_audio_segments(len(ref_y), ref_sr, segments, tts_chunks)
    aligned_tts_path = os.path.join(out_dir, "aligned_tts.wav")
    sf.write(aligned_tts_path, placed, ref_sr)
    print(f"[SAVE] Aligned TTS: {aligned_tts_path}")

    # 8) Mix with instrumental
    print("[8/8] Mixing with instrumental…")
    instr_y, instr_sr = librosa.load(instr_wav, mono=False)
    if instr_y.ndim == 1:
        instr_y = np.stack([instr_y, instr_y], axis=-1)
    if instr_sr != ref_sr:
        instr_y = np.stack(
            [resample_audio(instr_y[:, ch], instr_sr, ref_sr) for ch in range(instr_y.shape[1])],
            axis=-1,
        )

    final_mix = mix_tracks(placed, instr_y, ref_sr, vocals_vol=vocals_vol, instr_vol=instr_vol)
    final_out = os.path.join(out_dir, "final_translated_song.wav")
    sf.write(final_out, final_mix, ref_sr)
    print(f"[DONE] Final mix: {final_out}")
    return final_out


# ------------------------------
# Optional: legacy entry (video_id + local song path)
# ------------------------------
def run_pipeline(
    video_id: str,
    song_path: str,
    vocals_vol: float = 0.7,
    instr_vol: float = 0.5,
    out_dir: str = "outputs",
):
    """
    Keeps compatibility with the older CLI signature (video_id + song file path).
    """
    os.makedirs(out_dir, exist_ok=True)

    print("[1/7] Fetching transcript…")
    transcript = fetch_youtube_transcript(video_id)
    if not transcript:
        raise RuntimeError("Transcript fetch failed.")

    print("[2/7] Translating text…")
    nmt = NMTTranslator(USE_TF_SERVING_NMT, NMT_SERVING_URL, NMT_SAVEDMODEL_DIR)
    translated = nmt.translate([transcript])[0]
    if not translated:
        raise RuntimeError("NMT produced empty output.")

    print("[3/7] Separating with Demucs…")
    vocals_wav, instr_wav = demucs_separate(song_path, output_dir=os.path.join(out_dir, "separated"))

    print("[4/7] Detecting vocal segments…")
    segments = detect_vocal_segments(vocals_wav)
    print(f"[VAD] {len(segments)} segments detected")

    print("[5/7] Synthesizing TTS…")
    tts = TTSSynthesizer(USE_TF_SERVING_TTS, TTS_SERVING_URL, TTS_SAVEDMODEL_DIR)
    ref_y, ref_sr = librosa.load(vocals_wav, mono=True)
    chunks_txt = split_text_to_segments(translated, len(segments))

    tts_chunks: List[np.ndarray] = []
    for (start_s, end_s), txt in zip(segments, chunks_txt):
        dur = max(0.05, end_s - start_s)
        audio, sr = tts.synthesize(txt, target_duration_s=dur)
        if sr != ref_sr:
            audio = resample_audio(audio, sr, ref_sr)
        if np.max(np.abs(audio)) > 0:
            audio = audio / max(1.0, np.max(np.abs(audio))) * 0.8
        tts_chunks.append(audio.astype(np.float32))

    print("[6/7] Aligning TTS…")
    placed = place_audio_segments(len(ref_y), ref_sr, segments, tts_chunks)
    sf.write(os.path.join(out_dir, "aligned_tts.wav"), placed, ref_sr)

    print("[7/7] Mixing…")
    instr_y, instr_sr = librosa.load(instr_wav, mono=False)
    if instr_y.ndim == 1:
        instr_y = np.stack([instr_y, instr_y], axis=-1)
    if instr_sr != ref_sr:
        instr_y = np.stack(
            [resample_audio(instr_y[:, ch], instr_sr, ref_sr) for ch in range(instr_y.shape[1])],
            axis=-1,
        )

    final_mix = mix_tracks(placed, instr_y, ref_sr, vocals_vol=vocals_vol, instr_vol=instr_vol)
    final_out = os.path.join(out_dir, "final_translated_song.wav")
    sf.write(final_out, final_mix, ref_sr)
    print(f"[DONE] Final mix: {final_out}")
    return final_out
