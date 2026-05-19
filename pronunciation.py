#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

try:
    import numpy as np
    import sounddevice as sd
    from scipy.io import wavfile
except ImportError as exc:
    sys.exit(f"Missing audio dep: {exc}\nRun: pip install sounddevice scipy numpy")

try:
    import requests as _requests
except ImportError:
    sys.exit("Missing: requests\nRun: pip install requests")

try:
    from colorama import Fore, Style, init as _colorama_init
    _colorama_init(autoreset=True)
except ImportError:
    class _FallbackColor:
        def __getattr__(self, _): return ""
    Fore = Style = _FallbackColor()

# gTTS is optional — used only when ElevenLabs key is absent
try:
    from gtts import gTTS as _gTTS
    _GTTS_OK = True
except ImportError:
    _GTTS_OK = False

SAMPLE_RATE  = 16000
PHRASES_DIR  = Path("phrases")
SESSIONS_DIR = Path("sessions")

ELEVENLABS_TTS_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
ELEVENLABS_MODEL   = "eleven_multilingual_v2"
ELEVENLABS_FORMAT  = "mp3_44100_128"
DEFAULT_VOICE_ID   = "CIvegTlxTePhYSiBL1r4"  # Pepito


# ─── helpers ─────────────────────────────────────────────────────────────────

def c(text: str, color: str = "", bold: bool = False, end: str = "\n") -> None:
    prefix = (Style.BRIGHT if bold else "") + color
    print(f"{prefix}{text}{Style.RESET_ALL}", end=end, flush=True)


def sep() -> None:
    c("─" * 62, Fore.CYAN)


def _load_dotenv(path: Path = Path(".env")) -> None:
    """Minimal .env loader — no dependency on python-dotenv."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


def _elevenlabs_key() -> str | None:
    _load_dotenv()
    return os.environ.get("ELEVENLABS_API_KEY") or None


def play_audio(path: Path) -> None:
    if sys.platform == "darwin":
        subprocess.run(["afplay", str(path)], check=True)
    elif sys.platform.startswith("linux"):
        if path.suffix.lower() == ".mp3":
            subprocess.run(["mpg123", "-q", str(path)], check=True)
        else:
            subprocess.run(["aplay", "-q", str(path)], check=True)
    else:
        os.startfile(str(path))


def record_audio(duration: int) -> np.ndarray:
    input(f"\n{Fore.CYAN}  Press Enter to record...{Style.RESET_ALL}")

    for i in (3, 2, 1):
        print(f"    {i}...", end="\r", flush=True)
        time.sleep(1)

    print(f"{Fore.RED}  Recording...              {Style.RESET_ALL}", end="\r", flush=True)

    audio = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
    )

    start = time.time()
    while time.time() - start < duration:
        remain = duration - (time.time() - start)
        print(f"{Fore.RED}  Recording... {remain:.1f}s left  {Style.RESET_ALL}", end="\r", flush=True)
        time.sleep(0.1)

    sd.wait()
    c("  Done!                        ", Fore.GREEN)
    return audio.flatten()


def session_name() -> str:
    ns = time.time_ns()
    dt = datetime.now()
    return dt.strftime("%Y%m%d_%H%M%S") + f"_{ns % 1_000_000_000:09d}"


# ─── TTS ─────────────────────────────────────────────────────────────────────

def _tts_elevenlabs(text: str, out_path: Path, voice_id: str, api_key: str) -> None:
    url = ELEVENLABS_TTS_URL.format(voice_id=voice_id)
    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg",
    }
    body = {
        "text": text,
        "model_id": ELEVENLABS_MODEL,
        "output_format": ELEVENLABS_FORMAT,
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75,
            "style": 0.2,
            "speed": 1.0,
            "use_speaker_boost": True,
        },
    }
    resp = _requests.post(url, headers=headers, json=body, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"ElevenLabs error {resp.status_code}: {resp.text[:200]}")
    out_path.write_bytes(resp.content)


def _tts_gtts(text: str, out_path: Path) -> None:
    if not _GTTS_OK:
        sys.exit("No TTS available. Set ELEVENLABS_API_KEY or install gtts.")
    _gTTS(text=text, lang="en", slow=False).save(str(out_path))


def generate_audio(text: str, out_path: Path, voice_id: str) -> str:
    """Generate TTS audio. Returns provider used."""
    api_key = _elevenlabs_key()
    if api_key:
        _tts_elevenlabs(text, out_path, voice_id, api_key)
        return "elevenlabs"
    _tts_gtts(text, out_path)
    return "gtts"


# ─── generate command ────────────────────────────────────────────────────────

def cmd_generate(args: argparse.Namespace) -> None:
    topic     = args.phrases
    voice_id  = args.voice
    topic_dir = PHRASES_DIR / topic

    phrases_file  = topic_dir / "phrases.txt"
    settings_file = topic_dir / "settings.json"
    voices_dir    = topic_dir / "voices"

    if not phrases_file.exists():
        c(f"Not found: {phrases_file}", Fore.RED)
        sys.exit(1)

    voices_dir.mkdir(parents=True, exist_ok=True)

    phrases = [
        ln.strip()
        for ln in phrases_file.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]

    if not phrases:
        c(f"No phrases in {phrases_file}", Fore.RED)
        sys.exit(1)

    api_key  = _elevenlabs_key()
    provider = "elevenlabs" if api_key else "gtts"

    # Load existing settings to find already-generated phrases
    existing: list[dict] = []
    if settings_file.exists():
        try:
            existing = json.loads(settings_file.read_text(encoding="utf-8"))
        except Exception:
            existing = []
    existing_texts = {item["text"] for item in existing}
    next_id = max((int(item["id"]) for item in existing), default=0) + 1

    new_phrases = [p for p in phrases if p not in existing_texts]

    print()
    c(f"  Topic: {topic}  [{provider}]", Fore.CYAN, bold=True)
    if provider == "elevenlabs":
        c(f"  Voice: {voice_id}", Fore.CYAN)
    c(f"  {len(existing_texts)} existing  |  {len(new_phrases)} new", Fore.CYAN)
    sep()

    if not new_phrases:
        c("  All phrases already generated. Nothing to do.", Fore.YELLOW)
        print()
        return

    settings = existing[:]
    for i, text in enumerate(new_phrases, next_id):
        audio_path = voices_dir / f"{i}.mp3"
        print(f"  [{i:02d}] {text} ", end="", flush=True)
        generate_audio(text, audio_path, voice_id)
        c(f"→ {audio_path.name}", Fore.GREEN)
        settings.append({"id": str(i), "text": text})

    settings_file.write_text(
        json.dumps(settings, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    sep()
    c(f"  settings.json updated — {len(settings)} total entries ({len(new_phrases)} new).", Fore.GREEN, bold=True)
    print()


# ─── training command ────────────────────────────────────────────────────────

def cmd_training(args: argparse.Namespace) -> None:
    topic     = args.topic
    topic_dir = PHRASES_DIR / topic

    settings_file = topic_dir / "settings.json"
    voices_dir    = topic_dir / "voices"

    if not settings_file.exists():
        c(f"settings.json not found for topic '{topic}'. Run 'generate' first.", Fore.RED)
        sys.exit(1)

    phrases: list[dict] = json.loads(settings_file.read_text(encoding="utf-8"))

    if not phrases:
        c("No phrases in settings.json", Fore.RED)
        sys.exit(1)

    if args.random:
        phrases = phrases[:]
        random.shuffle(phrases)

    if args.limit:
        phrases = phrases[: args.limit]

    duration = args.duration

    SESSIONS_DIR.mkdir(exist_ok=True)
    session_dir = SESSIONS_DIR / session_name()
    session_dir.mkdir(parents=True)

    print()
    c("═" * 62, Fore.CYAN)
    c(f"  Topic: {topic}  |  {len(phrases)} phrase(s)  |  {duration}s recording", Fore.CYAN, bold=True)
    c(f"  Session: sessions/{session_dir.name}/", Fore.CYAN)
    c("═" * 62, Fore.CYAN)

    completed = 0

    for idx, item in enumerate(phrases, 1):
        phrase_id  = item["id"]
        text       = item["text"]
        voice_file = voices_dir / f"{phrase_id}.mp3"

        if not voice_file.exists():
            c(f"\n  Audio missing: {voice_file} — skipping.", Fore.YELLOW)
            continue

        sep()
        c(f"  Phrase {idx} / {len(phrases)}", Fore.CYAN)
        print()
        c(f'  "{text}"', Fore.WHITE, bold=True)
        print()

        input(f"  {Fore.CYAN}Press Enter to listen...{Style.RESET_ALL}")

        # Listen loop
        while True:
            play_audio(voice_file)
            print()
            choice = input(
                f"  {Fore.CYAN}[R]{Style.RESET_ALL} Repeat  "
                f"{Fore.CYAN}[N]{Style.RESET_ALL} Next → "
            ).strip().lower()
            if choice != "r":
                break

        # Record loop
        user_audio: np.ndarray | None = None
        while True:
            try:
                user_audio = record_audio(duration)
            except KeyboardInterrupt:
                c("\n  Skipped recording.", Fore.YELLOW)
                user_audio = None
                break

            # Decision loop — stays here until R or N
            while True:
                print()
                choice = input(
                    f"  {Fore.CYAN}[L]{Style.RESET_ALL} Listen  "
                    f"{Fore.CYAN}[R]{Style.RESET_ALL} Repeat  "
                    f"{Fore.CYAN}[N]{Style.RESET_ALL} Next → "
                ).strip().lower()

                if choice == "l" and user_audio is not None:
                    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                    try:
                        wavfile.write(tmp.name, SAMPLE_RATE, (user_audio * 32767).astype(np.int16))
                        tmp.close()
                        play_audio(Path(tmp.name))
                    finally:
                        try:
                            os.unlink(tmp.name)
                        except OSError:
                            pass
                    continue  # show decision prompt again, no re-record

                break  # R or N exits decision loop

            if choice != "r":
                break  # N exits record loop, R re-records

        # Save session files
        item_dir = session_dir / f"{idx:02d}"
        item_dir.mkdir()
        (item_dir / "phrase.txt").write_text(text, encoding="utf-8")
        shutil.copy(str(voice_file), str(item_dir / "audio.mp3"))

        if user_audio is not None:
            wavfile.write(
                str(item_dir / "me.wav"),
                SAMPLE_RATE,
                (user_audio * 32767).astype(np.int16),
            )

        completed += 1

    print()
    c("═" * 62, Fore.CYAN)
    c(f"  Done! {completed}/{len(phrases)} phrase(s) completed.", Fore.GREEN, bold=True)
    c(f"  Saved: sessions/{session_dir.name}/", Fore.GREEN)
    c("═" * 62, Fore.CYAN)
    print()

    choice = input(
        f"  {Fore.CYAN}[F]{Style.RESET_ALL} Finish  "
        f"{Fore.CYAN}[L]{Style.RESET_ALL} Listen → "
    ).strip().lower()

    if choice == "l":
        print()
        subdirs = sorted(d for d in session_dir.iterdir() if d.is_dir())
        for sub in subdirs:
            phrase_txt = sub / "phrase.txt"
            ref_audio  = sub / "audio.mp3"
            my_audio   = sub / "me.wav"

            text_label = phrase_txt.read_text(encoding="utf-8").strip() if phrase_txt.exists() else sub.name
            sep()
            c(f'  "{text_label}"', Fore.WHITE, bold=True)

            if ref_audio.exists():
                c("  Reference:", Fore.CYAN)
                play_audio(ref_audio)

            if my_audio.exists():
                c("  You:", Fore.CYAN)
                play_audio(my_audio)

        sep()
        print()

    print()


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        prog="pronunciation",
        description="Phrase-based pronunciation trainer",
    )
    sub = p.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="Generate TTS audio from phrases.txt")
    gen.add_argument("--phrases", required=True, metavar="TOPIC",
                     help="Topic folder (e.g. IT)")
    gen.add_argument("--voice", default=DEFAULT_VOICE_ID, metavar="VOICE_ID",
                     help=f"ElevenLabs voice ID (default: {DEFAULT_VOICE_ID})")

    train = sub.add_parser("training", help="Practice pronunciation")
    train.add_argument("--topic", required=True, metavar="TOPIC",
                       help="Topic to practice (e.g. IT)")
    train.add_argument("--limit", type=int, default=None, metavar="N",
                       help="Max phrases to practice")
    train.add_argument("--duration", type=int, default=5, metavar="SEC",
                       help="Recording length in seconds (default: 5)")
    train.add_argument("--random", action="store_true",
                       help="Shuffle phrases")

    args = p.parse_args()

    if args.command == "generate":
        cmd_generate(args)
    elif args.command == "training":
        cmd_training(args)


if __name__ == "__main__":
    main()
