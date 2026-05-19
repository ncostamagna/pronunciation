#!/usr/bin/env python3
"""
test_sessions.py
Run the full pronunciation analysis pipeline on saved session audio files.
No mic needed — reads me.wav + phrase.txt from each session subfolder.
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from scipy.io import wavfile
import numpy as np

# Import from the trainer
sys.path.insert(0, str(Path(__file__).parent))
from pronunciation_trainer import (
    load_models,
    transcribe_with_timestamps,
    analyze_pronunciation,
    show_sentence_result,
    audio_to_ipa_frames,
    word_ipa,
    _tokenize,
    SAMPLE_RATE,
)

SESSIONS_DIR = Path("sessions")


def load_wav(path: Path) -> np.ndarray:
    rate, data = wavfile.read(str(path))
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32767.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483647.0
    if data.ndim > 1:
        data = data[:, 0]
    return data


def collect_samples():
    samples = []
    for session_dir in sorted(SESSIONS_DIR.iterdir()):
        if not session_dir.is_dir():
            continue
        for sub in sorted(session_dir.iterdir()):
            if not sub.is_dir():
                continue
            phrase_file = sub / "phrase.txt"
            wav_file = sub / "me.wav"
            if phrase_file.exists() and wav_file.exists():
                samples.append((sub, phrase_file.read_text(encoding="utf-8").strip(), wav_file))
    return samples


def main():
    samples = collect_samples()
    if not samples:
        print("No samples found in sessions/")
        sys.exit(1)

    print(f"\nFound {len(samples)} recorded samples\n")
    print("Loading models...")
    w_model, wv_proc, wv_model = load_models()
    print()

    for sub, phrase, wav_path in samples:
        print("=" * 70)
        print(f"Session: {sub.parent.name[:8]}.../{sub.name}")
        print(f'Phrase:  "{phrase}"')

        audio = load_wav(wav_path)
        whisper_text, word_spans = transcribe_with_timestamps(audio, w_model)
        print(f'Heard:   "{whisper_text}"')

        # Show full-sentence IPA from wav2vec2
        _, tokenizer = wv_proc
        frame_ids = audio_to_ipa_frames(audio, wv_proc, wv_model)
        full_user_ipa = tokenizer.decode(frame_ids.tolist()).strip()
        ref_words = _tokenize(phrase)
        full_ref_ipa = " | ".join(f"{w}={word_ipa(w, tokenizer)}" for w in ref_words)
        print(f"  user IPA (full): {full_user_ipa}")
        print(f"  ref  IPA (text): {full_ref_ipa}")

        for strictness in ("lenient",):
            result = analyze_pronunciation(
                phrase, audio, whisper_text, word_spans,
                wv_proc, wv_model,
                strictness=strictness,
            )
            print(f"\n  [{strictness.upper():8s}] score={result.pct}%  ({result.correct_count}/{result.total_count} words)")
            for wr in result.word_results:
                icon = "✅" if wr.correct else "❌"
                if wr.correct and not wr.user_phonemes:
                    icon = "⚠️ "
                u_ph = f"/{wr.user_phonemes}/" if wr.user_phonemes else "(no acoustic data)"
                r_ph = f"/{wr.ref_phonemes}/"
                print(f"    {icon} {wr.word:<20s}  said {u_ph}  ref {r_ph}")

        print()


if __name__ == "__main__":
    main()
