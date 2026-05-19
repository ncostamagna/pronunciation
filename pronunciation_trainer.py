#!/usr/bin/env python3
"""
pronunciation_trainer.py

Fully local English pronunciation trainer CLI.
Pipeline: Mic → Whisper → Wav2Vec2 → eSpeak NG → Phoneme Diff → Ollama/Mistral → Feedback
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# Standard library
# ─────────────────────────────────────────────────────────────────────────────
import argparse
import os
import re
import sys
import tempfile
import time
import uuid
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Third-party — fail fast with helpful messages
# ─────────────────────────────────────────────────────────────────────────────
try:
    import numpy as np
    import sounddevice as sd
    from scipy.io import wavfile
except ImportError as exc:
    sys.exit(f"Missing audio dep: {exc}\nRun: pip install -r requirements.txt")

try:
    import torch
    from faster_whisper import WhisperModel
    from transformers import (
        Wav2Vec2ForCTC,
        Wav2Vec2FeatureExtractor,
        Wav2Vec2PhonemeCTCTokenizer,
    )
except ImportError as exc:
    sys.exit(f"Missing ML dep: {exc}\nRun: conda env create -f environment.yml")

try:
    import shutil as _shutil
    # phonemizer 3.x uses ctypes to load libespeak-ng directly (not the binary).
    # On macOS with Homebrew the library lives in /opt/homebrew/lib which is
    # outside the conda env's default search path, so ctypes.find_library fails.
    # We point it at the Homebrew dylib explicitly before importing phonemize.
    _LIB_CANDIDATES = [
        "/opt/homebrew/lib/libespeak-ng.dylib",         # macOS arm64/x86_64
        "/opt/homebrew/opt/espeak-ng/lib/libespeak-ng.dylib",
        "/usr/lib/libespeak-ng.so.1",                   # Linux
        "/usr/lib/x86_64-linux-gnu/libespeak-ng.so.1",
    ]
    try:
        from phonemizer.backend.espeak.wrapper import EspeakWrapper as _EW
        for _lib in _LIB_CANDIDATES:
            if os.path.exists(_lib):
                _EW.set_library(_lib)
                break
    except Exception:
        pass
    from phonemizer import phonemize as _phonemize_fn
    from phonemizer.backend.espeak.espeak import EspeakBackend as _EspeakBE
    _PHONEMIZER_OK = _EspeakBE.is_available()
except ImportError:
    _PHONEMIZER_OK = False

try:
    import requests
except ImportError:
    sys.exit("Missing: requests. Run: pip install requests")

try:
    from colorama import Fore, Style, init as _colorama_init
    _colorama_init(autoreset=True)
except ImportError:
    class _FallbackColor:
        def __getattr__(self, _): return ""
    Fore = Style = _FallbackColor()  # type: ignore[assignment]

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
SAMPLE_RATE           = 16000
OLLAMA_URL            = "http://localhost:11434/api/generate"
OLLAMA_MODEL          = "mistral"
WHISPER_SIZE          = "small"
# Outputs IPA phonemes directly from audio — much more accurate than base-960h
WAV2VEC2_PHO_ID       = "facebook/wav2vec2-lv-60-espeak-cv-ft"
# wav2vec2 CNN feature extractor stride: 320 samples = 20 ms at 16 kHz
WAV2VEC2_FRAME_STRIDE = 320

# ─────────────────────────────────────────────────────────────────────────────
# Built-in corpus — 20 sentences targeting Spanish-speaker pain points:
#   /θ/ /ð/ (th unvoiced/voiced), /v/ vs /b/, /r/, vowel length (/ɪ/→/iː/, /ʊ/→/uː/)
# ─────────────────────────────────────────────────────────────────────────────
BUILT_IN_SENTENCES: List[Dict] = [
    # ── Beginner ──────────────────────────────────────────────────────────────
    {"text": "Think before you speak.",
     "difficulty": "beginner", "focus": "/θ/"},
    {"text": "This is the best thing I have ever seen.",
     "difficulty": "beginner", "focus": "/θ/ and /ð/"},
    {"text": "I want to visit the village.",
     "difficulty": "beginner", "focus": "/v/ vs /b/"},
    {"text": "The weather is very beautiful today.",
     "difficulty": "beginner", "focus": "/ð/, /v/ vs /b/"},
    {"text": "Put the book on the bed.",
     "difficulty": "beginner", "focus": "final consonants, /b/"},
    {"text": "The ship is not the same as the sheep.",
     "difficulty": "beginner", "focus": "/ɪ/ vs /iː/"},
    {"text": "Three brothers live in a big village.",
     "difficulty": "beginner", "focus": "/θ/, /v/ vs /b/"},
    # ── Intermediate ──────────────────────────────────────────────────────────
    {"text": "Whether the weather is cold or hot, she always brings a vest.",
     "difficulty": "intermediate", "focus": "/ð/, /v/ vs /b/"},
    {"text": "She sells seashells by the seashore.",
     "difficulty": "intermediate", "focus": "/ʃ/ vs /s/"},
    {"text": "The full moon looks like a fool's golden pool.",
     "difficulty": "intermediate", "focus": "/ʊ/ vs /uː/"},
    {"text": "I think the northern road goes through the forest.",
     "difficulty": "intermediate", "focus": "/θ/, /ð/, /r/"},
    {"text": "The red vehicle drove through the valley last night.",
     "difficulty": "intermediate", "focus": "/θ/, /v/ vs /b/, /r/"},
    {"text": "Both brothers thought carefully about their theory.",
     "difficulty": "intermediate", "focus": "/θ/ and /ð/"},
    # ── Advanced ──────────────────────────────────────────────────────────────
    {"text": "The breathtaking view from the northern cliffs was thoroughly refreshing.",
     "difficulty": "advanced", "focus": "/θ/, /ð/, /r/"},
    {"text": "Three thousand threads were thoroughly woven throughout the entire fabric.",
     "difficulty": "advanced", "focus": "/θ/, /r/"},
    {"text": "The rhythm of the rain on the roof was rather relaxing.",
     "difficulty": "advanced", "focus": "/r/, /θ/"},
    {"text": "Whether you breathe through your mouth or nose, the effect differs greatly.",
     "difficulty": "advanced", "focus": "voiced/unvoiced /θ/ /ð/"},
    {"text": "The valuable vintage vehicle veered dangerously towards the ravine.",
     "difficulty": "advanced", "focus": "/v/ vs /b/, /r/"},
    {"text": "Rural road repairs rarely resolve themselves without thorough planning.",
     "difficulty": "advanced", "focus": "/r/, /θ/"},
    {"text": "The ship's crew ate sheep stew and drank fruit juice throughout the voyage.",
     "difficulty": "advanced", "focus": "vowels, /v/ vs /b/, final consonants"},
]

# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class WordResult:
    word: str
    correct: bool
    user_phonemes: str
    ref_phonemes: str
    wrong_pairs: List[Tuple[str, str]]   # (user_chunk, ref_chunk) per mismatch


@dataclass
class SentenceResult:
    sentence: str
    word_results: List[WordResult]
    correct_count: int
    total_count: int

    @property
    def score(self) -> float:
        return self.correct_count / self.total_count if self.total_count else 0.0

    @property
    def pct(self) -> int:
        return int(self.score * 100)


@dataclass
class SessionStats:
    sentences: List[SentenceResult] = field(default_factory=list)
    # ref_phoneme_char → list of words where it was missed
    phoneme_errors: Dict[str, List[str]] = field(
        default_factory=lambda: defaultdict(list)
    )

    def record(self, result: SentenceResult) -> None:
        self.sentences.append(result)
        for wr in result.word_results:
            if not wr.correct:
                for _u, ref_chunk in wr.wrong_pairs:
                    for ch in ref_chunk:       # each IPA char in the missed chunk
                        if ch.strip():
                            self.phoneme_errors[ch].append(wr.word)

    @property
    def overall_score(self) -> float:
        tot_c = sum(s.correct_count for s in self.sentences)
        tot_t = sum(s.total_count   for s in self.sentences)
        return tot_c / tot_t if tot_t else 0.0

    def top_missed(self, n: int = 3) -> List[Tuple[str, List[str], int]]:
        ranked = sorted(self.phoneme_errors.items(), key=lambda kv: len(kv[1]), reverse=True)
        out = []
        for ph, words in ranked[:n]:
            unique = list(dict.fromkeys(words))   # deduplicated, order-preserved
            out.append((ph, unique, len(words)))
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Terminal helpers
# ─────────────────────────────────────────────────────────────────────────────
def c(text: str, color: str = "", bold: bool = False, end: str = "\n") -> None:
    prefix = (Style.BRIGHT if bold else "") + color
    print(f"{prefix}{text}{Style.RESET_ALL}", end=end, flush=True)


def sep(width: int = 62) -> None:
    c("─" * width, Fore.CYAN)


def header(title: str) -> None:
    sep()
    c(f"  {title}", Fore.CYAN, bold=True)
    sep()


# ─────────────────────────────────────────────────────────────────────────────
# Sentence splitting
# ─────────────────────────────────────────────────────────────────────────────
_ABBR = re.compile(
    r"\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|vs|etc|e\.g|i\.e|Fig|Vol|No|pp"
    r"|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.",
    re.IGNORECASE,
)
_DECIMAL = re.compile(r"(\d+)\.(\d+)")
_MARK_A  = "\x00A\x00"
_MARK_D  = "\x00D\x00"


def split_sentences(text: str) -> List[str]:
    text = _ABBR.sub(lambda m: m.group().replace(".", _MARK_A), text)
    text = _DECIMAL.sub(lambda m: f"{m.group(1)}{_MARK_D}{m.group(2)}", text)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"‘“])", text)
    result = []
    for p in parts:
        p = p.replace(_MARK_A, ".").replace(_MARK_D, ".").strip()
        if p:
            result.append(p)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────
def load_models():
    """Load faster-whisper + wav2vec2-lv-60-espeak-cv-ft. First run ~1.5 GB download."""
    c("\n  Loading models — first run downloads ~1.5 GB.\n", Fore.YELLOW)

    c("  ⏳ Whisper (small)...", Fore.YELLOW, end="\r")
    device       = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    w_model = WhisperModel(WHISPER_SIZE, device=device, compute_type=compute_type)
    c("  ✅ Whisper ready.                        ", Fore.GREEN)

    c("  ⏳ Wav2Vec2 phoneme model (lv-60)...     ", Fore.YELLOW, end="\r")
    feat_ext  = Wav2Vec2FeatureExtractor.from_pretrained(WAV2VEC2_PHO_ID)
    # Tokenizer needs espeak to ENCODE text → phoneme IDs (reference IPA).
    # espeak library is available via PHONEMIZER_ESPEAK_LIBRARY / set_library above,
    # but the tokenizer uses a subprocess check for the binary name 'espeak'.
    # If init_backend fails (espeak binary not found), fall back to no-op so the
    # tokenizer still loads for DECODE-only use.
    try:
        tokenizer = Wav2Vec2PhonemeCTCTokenizer.from_pretrained(WAV2VEC2_PHO_ID)
    except Exception:
        _orig = Wav2Vec2PhonemeCTCTokenizer.init_backend
        Wav2Vec2PhonemeCTCTokenizer.init_backend = lambda self, lang: None
        try:
            tokenizer = Wav2Vec2PhonemeCTCTokenizer.from_pretrained(WAV2VEC2_PHO_ID)
        finally:
            Wav2Vec2PhonemeCTCTokenizer.init_backend = _orig
    pho_model = Wav2Vec2ForCTC.from_pretrained(WAV2VEC2_PHO_ID)
    pho_model.eval()
    c("  ✅ Phoneme model ready.                  ", Fore.GREEN)

    return w_model, (feat_ext, tokenizer), pho_model


# ─────────────────────────────────────────────────────────────────────────────
# Session folder management
# ─────────────────────────────────────────────────────────────────────────────
SESSIONS_DIR = Path("sessions")

def new_session_dir() -> Path:
    """Create sessions/<uuid>/ and return its path."""
    path = SESSIONS_DIR / str(uuid.uuid4())
    path.mkdir(parents=True, exist_ok=True)
    return path

def save_sentence_files(session_dir: Path, idx: int, phrase: str, audio: np.ndarray) -> None:
    """Save phrase.txt and me.wav inside sessions/<uuid>/<idx:02d>/."""
    folder = session_dir / f"{idx:02d}"
    folder.mkdir(exist_ok=True)
    (folder / "phrase.txt").write_text(phrase, encoding="utf-8")
    wavfile.write(str(folder / "me.wav"), SAMPLE_RATE, (audio * 32767).astype(np.int16))


# ─────────────────────────────────────────────────────────────────────────────
# Audio recording
# ─────────────────────────────────────────────────────────────────────────────
def record_audio(duration: int) -> np.ndarray:
    """Record from the default microphone. Returns float32 array at SAMPLE_RATE."""
    print()
    input(f"{Fore.CYAN}  Press Enter when ready...{Style.RESET_ALL}")

    for i in (3, 2, 1):
        c(f"    {i}...", Fore.YELLOW, end="\r")
        time.sleep(1)

    c("  🔴 Recording...              ", Fore.RED, bold=True, end="\r")

    audio = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
    )

    start = time.time()
    while time.time() - start < duration:
        remain = duration - (time.time() - start)
        c(f"  🔴 Recording... {remain:.1f}s left  ", Fore.RED, bold=True, end="\r")
        time.sleep(0.1)

    sd.wait()
    c("  ✅ Done!                      ", Fore.GREEN)
    return audio.flatten()


# ─────────────────────────────────────────────────────────────────────────────
# Transcription — faster-whisper with word-level timestamps
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class WordSpan:
    word: str
    start: float   # seconds into the audio
    end: float


def transcribe_with_timestamps(audio: np.ndarray, model) -> Tuple[str, List[WordSpan]]:
    """Return (full_text, per-word timing) using faster-whisper word timestamps."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp = f.name
    try:
        wavfile.write(tmp, SAMPLE_RATE, (audio * 32767).astype(np.int16))
        segments, _ = model.transcribe(tmp, language="en", beam_size=5, word_timestamps=True)
        spans: List[WordSpan] = []
        text_parts: List[str] = []
        for seg in segments:
            text_parts.append(seg.text)
            if seg.words:
                for w in seg.words:
                    spans.append(WordSpan(w.word.strip(), w.start, w.end))
        return " ".join(text_parts).strip().lower(), spans
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Phoneme extraction — wav2vec2-lv-60-espeak-cv-ft on full audio
#
# Strategy: run the model ONCE on the entire recording (good context),
# then slice the frame-level token predictions per word using Whisper timestamps.
# Running on tiny isolated word clips produced garbage — models need context.
# ─────────────────────────────────────────────────────────────────────────────
def audio_to_ipa_frames(audio: np.ndarray, pho_processor, model) -> "torch.Tensor":
    """Run wav2vec2 on full audio. Returns raw argmax token IDs per frame."""
    feat_ext, _ = pho_processor
    inputs = feat_ext(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits   # [1, T, vocab]
    return torch.argmax(logits, dim=-1)[0]  # [T]


def frames_to_ipa(frame_ids: "torch.Tensor", start_sec: float, end_sec: float,
                  tokenizer) -> str:
    """CTC-decode the frame range [start_sec, end_sec] to an IPA string."""
    s = max(0, int(start_sec * SAMPLE_RATE / WAV2VEC2_FRAME_STRIDE))
    e = min(len(frame_ids), int(end_sec * SAMPLE_RATE / WAV2VEC2_FRAME_STRIDE) + 1)
    if e <= s:
        return ""
    return tokenizer.decode(frame_ids[s:e].tolist()).strip()


# ─────────────────────────────────────────────────────────────────────────────
# Reference IPA — tokenizer encoding (same vocab as acoustic model output)
# ─────────────────────────────────────────────────────────────────────────────
_tok_cache: Dict[str, str] = {}

def word_ipa(word: str, tokenizer=None) -> str:
    """
    Convert one word to IPA using:
    - tokenizer encoding (preferred — same vocab as wav2vec2 output)
    - phonemizer eSpeak fallback
    Returns '' if neither is available.
    """
    clean = re.sub(r"[^a-zA-Z']", "", word).lower()
    if not clean:
        return ""
    if clean in _tok_cache:
        return _tok_cache[clean]

    # Tokenizer path: uses the same IPA vocabulary as the acoustic model
    if tokenizer is not None:
        try:
            ids = tokenizer(clean)["input_ids"]
            ipa = tokenizer.decode(ids).strip()
            _tok_cache[clean] = ipa
            return ipa
        except Exception:
            pass

    # eSpeak fallback (if tokenizer unavailable)
    if _PHONEMIZER_OK:
        try:
            out = _phonemize_fn(
                clean,
                backend="espeak",
                language="en-us",
                with_stress=False,
                preserve_punctuation=False,
                njobs=1,
            )
            ipa = out.strip().replace(" ", "")
            _tok_cache[clean] = ipa
            return ipa
        except Exception:
            pass

    return ""


# ─────────────────────────────────────────────────────────────────────────────
# Phoneme comparison helpers
# ─────────────────────────────────────────────────────────────────────────────
def _tokenize(s: str) -> List[str]:
    """Split sentence into clean lowercase word tokens."""
    return [
        re.sub(r"[^a-zA-Z']", "", w).lower()
        for w in s.split()
        if re.sub(r"[^a-zA-Z']", "", w)
    ]


def _align_words(
    user: List[str], ref: List[str]
) -> List[Tuple[Optional[str], Optional[str]]]:
    """Align two word lists with SequenceMatcher. Returns (user_w, ref_w) pairs."""
    sm = SequenceMatcher(None, user, ref, autojunk=False)
    pairs: List[Tuple[Optional[str], Optional[str]]] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        u_chunk, r_chunk = user[i1:i2], ref[j1:j2]
        if tag == "equal":
            pairs.extend(zip(u_chunk, r_chunk))
        elif tag == "replace":
            for u, r in zip(u_chunk, r_chunk):
                pairs.append((u, r))
            for u in u_chunk[len(r_chunk):]:
                pairs.append((u, None))
            for r in r_chunk[len(u_chunk):]:
                pairs.append((None, r))
        elif tag == "delete":
            for u in u_chunk:
                pairs.append((u, None))
        elif tag == "insert":
            for r in r_chunk:
                pairs.append((None, r))
    return pairs


def _phoneme_diff(user_ph: str, ref_ph: str) -> Tuple[bool, List[Tuple[str, str]]]:
    """
    Character-level diff of two IPA strings.
    Returns (is_correct, [(user_chunk, ref_chunk), ...]) for every mismatch.
    """
    if user_ph == ref_ph:
        return True, []
    sm = SequenceMatcher(None, list(user_ph), list(ref_ph), autojunk=False)
    wrong: List[Tuple[str, str]] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != "equal":
            u = "".join(list(user_ph)[i1:i2])
            r = "".join(list(ref_ph)[j1:j2])
            wrong.append((u, r))
    return False, wrong


# ─────────────────────────────────────────────────────────────────────────────
# Core analysis — word timestamps + per-word IPA from audio
# ─────────────────────────────────────────────────────────────────────────────
def analyze_pronunciation(
    ref_sentence: str,
    audio: np.ndarray,
    whisper_text: str,
    word_spans: List[WordSpan],
    pho_processor,
    pho_model,
) -> SentenceResult:
    ref_tokens  = _tokenize(ref_sentence)
    user_tokens = _tokenize(whisper_text)
    aligned     = _align_words(user_tokens, ref_tokens)

    _, tokenizer = pho_processor

    # Run wav2vec2 once on full audio — better than per-word clips (needs context).
    frame_ids = audio_to_ipa_frames(audio, pho_processor, pho_model)

    span_map: Dict[str, Tuple[float, float]] = {}
    for span in word_spans:
        key = re.sub(r"[^a-zA-Z']", "", span.word).lower()
        if key and key not in span_map:
            span_map[key] = (span.start, span.end)

    results: List[WordResult] = []
    correct_count = total_count = 0

    for user_w, ref_w in aligned:
        if ref_w is None:
            continue

        total_count += 1
        ref_ph = word_ipa(ref_w, tokenizer)

        if user_w is None:
            results.append(WordResult(
                word=ref_w, correct=False,
                user_phonemes="", ref_phonemes=ref_ph,
                wrong_pairs=[("", ref_ph)],
            ))
            continue

        # Fast path: if Whisper transcribed the word correctly, pronunciation is OK.
        # The IPA of the same word via the same tokenizer would be identical.
        if user_w == ref_w:
            correct_count += 1
            results.append(WordResult(
                word=ref_w, correct=True,
                user_phonemes=ref_ph, ref_phonemes=ref_ph,
                wrong_pairs=[],
            ))
            continue

        # Word mismatch — get acoustic IPA from full-audio frame slice.
        timing = span_map.get(user_w)
        if timing is not None:
            user_ph = frames_to_ipa(frame_ids, timing[0], timing[1], tokenizer)
        else:
            user_ph = ""

        # Fallback: tokenizer-encode the Whisper transcript word
        if not user_ph:
            user_ph = word_ipa(user_w, tokenizer)

        # If reference IPA unavailable, fall back to word equality (already false here)
        if not ref_ph:
            results.append(WordResult(
                word=ref_w, correct=False,
                user_phonemes=user_ph or user_w,
                ref_phonemes=ref_w,
                wrong_pairs=[(user_ph or user_w, ref_w)],
            ))
            continue

        if not user_ph:
            user_ph = user_w

        ok, wrong = _phoneme_diff(user_ph, ref_ph)
        if ok:
            correct_count += 1
        results.append(WordResult(
            word=ref_w, correct=ok,
            user_phonemes=user_ph, ref_phonemes=ref_ph,
            wrong_pairs=wrong,
        ))

    return SentenceResult(
        sentence=ref_sentence,
        word_results=results,
        correct_count=correct_count,
        total_count=total_count,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Ollama integration
# ─────────────────────────────────────────────────────────────────────────────
_ollama_status: Optional[bool] = None  # None = not checked yet


def ollama_ok() -> bool:
    global _ollama_status
    if _ollama_status is None:
        try:
            r = requests.get("http://localhost:11434/api/tags", timeout=3)
            _ollama_status = r.status_code == 200
        except Exception:
            _ollama_status = False
    return bool(_ollama_status)


def _ollama_ask(prompt: str, timeout: int = 60) -> str:
    try:
        r = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=timeout,
        )
        if r.status_code == 200:
            return r.json().get("response", "").strip()
    except Exception:
        pass
    return ""


def get_word_feedback(wrong_words: List[WordResult]) -> str:
    """Generate coaching feedback for mispronounced words via Ollama."""
    if not wrong_words:
        return ""
    lines = []
    for w in wrong_words:
        if w.wrong_pairs:
            u = w.user_phonemes or "?"
            lines.append(f"- '{w.word}': said /{u}/, correct /{w.ref_phonemes}/")
    if not lines:
        return ""
    prompt = (
        "You are a friendly English pronunciation coach. "
        "Give brief actionable feedback (1-2 sentences per word). "
        "Use IPA. Focus on tongue/lip position.\n\n"
        "Errors:\n" + "\n".join(lines) + "\n\nFeedback:"
    )
    return _ollama_ask(prompt)


def get_minimal_pairs(phonemes: List[str]) -> str:
    """Ask Ollama for minimal pair practice suggestions."""
    if not phonemes:
        return ""
    ph_str = ", ".join(f"/{p}/" for p in phonemes[:3])
    prompt = (
        f"English pronunciation coach: student struggles with {ph_str}. "
        "Suggest 3-4 minimal pairs per sound. "
        "Format: /sound/: word1/word2, word3/word4. Brief."
    )
    return _ollama_ask(prompt, timeout=45)


# ─────────────────────────────────────────────────────────────────────────────
# Display helpers
# ─────────────────────────────────────────────────────────────────────────────
_PAIR_FALLBACK: Dict[str, str] = {
    "θ": "think/sink, three/tree, both/boat, thin/tin",
    "ð": "this/dis, breathe/breed, other/udder, though/dough",
    "v": "very/berry, vine/bine, vote/boat, veil/bail",
    "r": "rice/lice, red/led, right/light, wrong/long",
    "ɹ": "rice/lice, red/led, right/light",
    "ɪ": "ship/sheep, sit/seat, bit/beat, fill/feel",
    "uː": "full/fool, pull/pool, look/Luke",
}


def show_sentence_result(result: SentenceResult, feedback: str) -> None:
    print()
    for wr in result.word_results:
        if wr.correct:
            c(f'  ✅  "{wr.word}"', Fore.GREEN, end="")
            c(f"  → /{wr.ref_phonemes}/ ✓", Fore.GREEN)
        else:
            c(f'  ❌  "{wr.word}"', Fore.RED, end="")
            if wr.user_phonemes:
                c(f"  → you said /{wr.user_phonemes}/, correct is /{wr.ref_phonemes}/", Fore.RED)
            else:
                c(f"  → correct /{wr.ref_phonemes}/ (not detected in audio)", Fore.RED)

    pct   = result.pct
    col   = Fore.GREEN if pct >= 80 else (Fore.YELLOW if pct >= 50 else Fore.RED)
    bar   = "█" * (pct // 5) + "░" * (20 - pct // 5)
    print()
    c(f"  📊 Score: {result.correct_count}/{result.total_count} sounds correct ({pct}%)", col, bold=True)
    c(f"     [{bar}]", col)

    if feedback:
        print()
        c("  💡 Coach:", Fore.YELLOW, bold=True)
        for line in feedback.splitlines():
            if line.strip():
                c(f"     {line.strip()}", Fore.YELLOW)


def show_summary(stats: SessionStats, use_llm: bool) -> None:
    print()
    header("SESSION SUMMARY")

    pct = int(stats.overall_score * 100)
    col = Fore.GREEN if pct >= 80 else (Fore.YELLOW if pct >= 50 else Fore.RED)
    c(f"  Sentences completed:  {len(stats.sentences)}", Fore.CYAN)
    c(f"  Overall score:        {pct}%", col, bold=True)

    top = stats.top_missed(3)
    if top:
        print()
        c("  🔁 Most missed sounds:", Fore.MAGENTA, bold=True)
        for ph, words, count in top:
            wlist = ", ".join(words[:5])
            c(f"     /{ph}/  →  missed {count}×  ({wlist})", Fore.MAGENTA)

        print()
        if use_llm and ollama_ok():
            c("  ⏳ Generating practice tips...", Fore.YELLOW, end="\r")
            tips = get_minimal_pairs([ph for ph, _, _ in top])
            if tips:
                c("  💡 Practice these minimal pairs:   ", Fore.YELLOW, bold=True)
                for line in tips.splitlines():
                    if line.strip():
                        c(f"     {line.strip()}", Fore.YELLOW)
            else:
                _show_fallback_pairs(top)
        else:
            _show_fallback_pairs(top)

    sep()


def _show_fallback_pairs(top: List[Tuple[str, List[str], int]]) -> None:
    c("  💡 Minimal pairs to practice:", Fore.YELLOW, bold=True)
    for ph, _, _ in top:
        pairs = _PAIR_FALLBACK.get(ph)
        if pairs:
            c(f"     /{ph}/: {pairs}", Fore.YELLOW)


# ─────────────────────────────────────────────────────────────────────────────
# Per-sentence loop
# ─────────────────────────────────────────────────────────────────────────────
def run_one_sentence(
    sentence: str,
    idx: int,
    total: int,
    w_model,
    wv_proc,
    wv_model,
    duration: int,
    use_llm: bool,
    session_dir: Optional[Path] = None,
) -> Optional[SentenceResult]:
    """
    Display → record → analyze → show → prompt.
    Returns SentenceResult or None if user quits.
    """
    while True:
        sep()
        c(f"  Sentence {idx} of {total}", Fore.CYAN)
        print()
        c(f'  👉  "{sentence}"', Fore.WHITE, bold=True)

        try:
            audio = record_audio(duration)
        except KeyboardInterrupt:
            return None

        if session_dir is not None:
            save_sentence_files(session_dir, idx, sentence, audio)

        c("\n  ⚙️  Analyzing...", Fore.CYAN)

        whisper_text, word_spans = transcribe_with_timestamps(audio, w_model)

        c(f"  📝 Heard: \"{whisper_text}\"", Fore.CYAN)

        result = analyze_pronunciation(
            sentence, audio, whisper_text, word_spans, wv_proc, wv_model
        )
        feedback = ""
        if use_llm and ollama_ok():
            bad = [wr for wr in result.word_results if not wr.correct and wr.wrong_pairs]
            if bad:
                feedback = get_word_feedback(bad)

        show_sentence_result(result, feedback)

        print()
        sep()
        choice = input(
            f"  {Fore.CYAN}[R]{Style.RESET_ALL} Retry  "
            f"{Fore.CYAN}[N]{Style.RESET_ALL} Next  "
            f"{Fore.CYAN}[Q]{Style.RESET_ALL} Quit  → "
        ).strip().lower()

        if choice == "q":
            return None
        if choice != "r":
            return result
        # 'r' → fall through to top of while


# ─────────────────────────────────────────────────────────────────────────────
# Session runner
# ─────────────────────────────────────────────────────────────────────────────
def run_session(sentences: List[str], args: argparse.Namespace) -> None:
    total = len(sentences)
    if not total:
        c("No sentences to practice.", Fore.RED)
        return

    c(f"\n  📄 {total} sentence(s) loaded.", Fore.CYAN)

    session_dir = new_session_dir()
    c(f"  💾 Session saved to: sessions/{session_dir.name}/", Fore.CYAN)

    w_model, wv_proc, wv_model = load_models()

    use_llm = not args.no_llm
    if use_llm:
        if ollama_ok():
            c("  ✅ Ollama connected.", Fore.GREEN)
        else:
            c("  ⚠️  Ollama not running — switching to --no-llm mode.", Fore.YELLOW)
            use_llm = False

    if not _PHONEMIZER_OK:
        c("  ⚠️  phonemizer not installed — phoneme analysis disabled (word-level only).", Fore.YELLOW)

    stats = SessionStats()

    for i, sentence in enumerate(sentences, 1):
        result = run_one_sentence(
            sentence, i, total,
            w_model, wv_proc, wv_model,
            args.duration, use_llm,
            session_dir=session_dir,
        )
        if result is None:
            c("\n  Session ended early.", Fore.YELLOW)
            break
        stats.record(result)

    if stats.sentences:
        show_summary(stats, use_llm)
    else:
        c("\n  No sentences completed.", Fore.YELLOW)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pronunciation_trainer.py",
        description="Fully local English pronunciation trainer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pronunciation_trainer.py                          # built-in sentences
  python pronunciation_trainer.py --difficulty beginner    # filter by difficulty
  python pronunciation_trainer.py --file my_text.txt       # load from file
  python pronunciation_trainer.py --paste                  # type/paste text
  python pronunciation_trainer.py --no-llm                 # skip Ollama
  python pronunciation_trainer.py --duration 8             # 8-second recording
        """,
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--file",  metavar="PATH", help="Load sentences from text file")
    mode.add_argument("--paste", action="store_true", help="Paste or type text interactively")

    p.add_argument("--no-llm",    action="store_true", help="Skip Ollama; show raw phoneme diffs only")
    p.add_argument("--duration",  type=int, default=5, metavar="SEC",
                   help="Recording length in seconds (default: 5)")
    p.add_argument("--difficulty", choices=["beginner", "intermediate", "advanced"],
                   help="Filter built-in sentences by difficulty")
    return p


def _get_sentences(args: argparse.Namespace) -> List[str]:
    if args.file:
        if not os.path.isfile(args.file):
            c(f"File not found: {args.file}", Fore.RED)
            sys.exit(1)
        with open(args.file, encoding="utf-8") as f:
            text = f.read()
        sentences = split_sentences(text)
        c(f"  📂 Loaded {len(sentences)} sentence(s) from '{args.file}'.", Fore.CYAN)
        return sentences

    if args.paste:
        c("  📋 Paste or type your text. Press Enter twice when done.\n", Fore.CYAN)
        lines: List[str] = []
        blank_streak = 0
        while True:
            try:
                line = input()
            except EOFError:
                break
            if line == "":
                blank_streak += 1
                if blank_streak >= 2:
                    break
            else:
                blank_streak = 0
                lines.append(line)
        sentences = split_sentences(" ".join(lines))
        c(f"\n  📄 Detected {len(sentences)} sentence(s).", Fore.CYAN)
        return sentences

    # Built-in mode
    pool = BUILT_IN_SENTENCES
    if args.difficulty:
        pool = [s for s in pool if s["difficulty"] == args.difficulty]
    if not pool:
        c(f"No built-in sentences for difficulty '{args.difficulty}'.", Fore.RED)
        sys.exit(1)
    sentences = [s["text"] for s in pool]
    label = f" [{args.difficulty}]" if args.difficulty else ""
    c(f"  📚 Built-in mode: {len(sentences)} sentence(s){label}.", Fore.CYAN)
    return sentences


def main() -> None:
    print()
    c("═" * 62, Fore.CYAN)
    c("  🎤  Pronunciation Trainer  —  fully local, no cloud APIs", Fore.CYAN, bold=True)
    c("═" * 62, Fore.CYAN)
    print()

    args      = _build_parser().parse_args()
    sentences = _get_sentences(args)

    if not sentences:
        c("No sentences to practice.", Fore.RED)
        sys.exit(1)

    run_session(sentences, args)

    print()
    c("  Goodbye! Keep practicing. 💪", Fore.CYAN, bold=True)
    print()


if __name__ == "__main__":
    main()
