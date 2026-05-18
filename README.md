# Pronunciation Trainer

Fully local English pronunciation trainer CLI. No cloud APIs — everything runs on your machine.

**Pipeline:** Microphone → Whisper → Wav2Vec2 → eSpeak NG → Phoneme Diff → Ollama/Mistral → Feedback

---

## Requirements

- macOS or Linux
- [Anaconda or Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- [Ollama](https://ollama.com) (optional — only needed for AI coaching feedback)
- A working microphone

---

## Install

### 1. System dependency — eSpeak NG

**macOS:**
```bash
brew install espeak-ng
```

**Ubuntu / Debian:**
```bash
sudo apt install espeak-ng
```

**Fedora / RHEL:**
```bash
sudo dnf install espeak-ng
```

---

### 2. Conda environment

```bash
conda env create -f environment.yml
conda activate pronunciation
```

> First run will download Whisper (~244 MB) and Wav2Vec2 (~360 MB) models automatically.

---

### 3. Ollama + Mistral (optional — for AI coaching feedback)

**Install Ollama:**
```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh
```

**Pull the Mistral model:**
```bash
ollama pull mistral
```

**Start the Ollama server** (keep this running in a separate terminal):
```bash
ollama serve
```

If Ollama is not running, the app automatically falls back to raw phoneme diff mode with no coaching text.

---

## Run

### Built-in sentences (default)

Cycles through 20 built-in sentences focused on sounds that are hard for Spanish speakers (`/θ/`, `/ð/`, `/v/` vs `/b/`, `/r/`, vowel length):

```bash
python pronunciation_trainer.py
```

Filter by difficulty:
```bash
python pronunciation_trainer.py --difficulty beginner
python pronunciation_trainer.py --difficulty intermediate
python pronunciation_trainer.py --difficulty advanced
```

---

### Load from a text file

```bash
python pronunciation_trainer.py --file my_text.txt
```

The app splits the text into sentences automatically, then goes one by one. Handles abbreviations (`Mr.`, `Dr.`) and decimal numbers without false splits.

---

### Paste / type text interactively

```bash
python pronunciation_trainer.py --paste
```

Paste or type a paragraph in the terminal, then press **Enter twice** to start. The app splits it into sentences and processes each one.

---

## Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--file PATH` | — | Load sentences from a `.txt` file |
| `--paste` | — | Type or paste text interactively |
| `--difficulty beginner\|intermediate\|advanced` | all | Filter built-in sentences |
| `--duration SEC` | `5` | Recording length in seconds |
| `--no-llm` | off | Skip Ollama entirely — show raw phoneme diffs only |

---

## How it works

1. The app shows a sentence to read aloud
2. Press **Enter** when ready — a 3…2…1 countdown starts
3. Speak clearly into your microphone
4. The app analyzes your audio:
   - **Whisper** transcribes what you said
   - **Wav2Vec2** extracts the acoustic character sequence (no language model — reveals real mispronunciations)
   - **eSpeak NG** converts both your words and the reference to IPA phonemes
   - A phoneme diff is run word by word
5. Results are shown with ✅ / ❌ per word, IPA comparison, and a score
6. **Ollama/Mistral** generates a coaching tip for each wrong word
7. At the end of all sentences, a session summary shows your overall score, most missed phonemes, and minimal pair practice suggestions

**Example output:**
```
──────────────────────────────────────────────────────────────
  Sentence 2 of 8

  👉  "I think the northern road goes through the forest."

  Press Enter when ready...
    3...
  🔴 Recording... 4.1s left

  📝 Heard: "i tink the nordern road goes tru the forest"

  ✅  "i"         → /aɪ/ ✓
  ❌  "think"     → you said /tɪŋk/, correct is /θɪŋk/
  ✅  "the"       → /ðə/ ✓
  ❌  "northern"  → you said /nɔːɹdɛɹn/, correct is /nɔːɹðəɹn/
  ✅  "road"      → /ɹoʊd/ ✓
  ✅  "goes"      → /ɡoʊz/ ✓
  ❌  "through"   → you said /tɹuː/, correct is /θɹuː/
  ✅  "the"       → /ðə/ ✓
  ✅  "forest"    → /fɔːɹɪst/ ✓

  📊 Score: 6/9 sounds correct (67%)
     [█████████████░░░░░░░]

  💡 Coach:
     'think': The /θ/ sound needs the tongue between the teeth — not a /t/.
     'northern': The 'th' in 'northern' is voiced /ð/ — same position but add voice.
     'through': Start with /θ/, tongue between teeth, then pull back quickly.

──────────────────────────────────────────────────────────────
  [R] Retry  [N] Next  [Q] Quit  →
```

**Session summary:**
```
──────────────────────────────────────────────────────────────
  SESSION SUMMARY
──────────────────────────────────────────────────────────────
  Sentences completed:  8
  Overall score:        74%

  🔁 Most missed sounds:
     /θ/  →  missed 7×  (think, through, three, northern, breathe)
     /ð/  →  missed 4×  (northern, weather, whether, other)
     /v/  →  missed 2×  (very, village)

  💡 Practice these minimal pairs:
     /θ/: think/sink, three/tree, both/boat, thin/tin
     /ð/: this/dis, breathe/breed, other/udder
     /v/: very/berry, vine/bine, vote/boat
```

---

## Troubleshooting

### Ollama not running
```
⚠️  Ollama not running — switching to --no-llm mode.
```
Start Ollama with `ollama serve` in a separate terminal, then rerun. Or use `--no-llm` intentionally.

### Mistral model not pulled
```bash
ollama pull mistral
```

### No microphone detected
Check system audio settings. List available devices:
```bash
python -c "import sounddevice; print(sounddevice.query_devices())"
```

### eSpeak NG not found (phonemizer warning)
```
⚠️  phonemizer/espeak-ng not available — phoneme comparison disabled
```
Install eSpeak NG (see Install step 1 above). The app still works — it falls back to word-level comparison.

### Model download fails / slow
Whisper and Wav2Vec2 download on first run. If interrupted, rerun — HuggingFace caches partial downloads. Models are cached at `~/.cache/huggingface/` and `~/.cache/whisper/`.

### `conda env create` fails on Python 3.14
Some pip packages may not yet ship wheels for Python 3.14. If you see build errors, pin Python to `3.11` in `environment.yml` and recreate:
```bash
conda env remove -n pronunciation
conda env create -f environment.yml
```

### Audio sounds clipped / distorted
Increase recording duration with `--duration 8` to give yourself more time per sentence.
