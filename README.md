# pronunciation

Phrase-based English pronunciation trainer. Generate TTS audio from phrase lists, then practice with your mic.

---

## Requirements

- macOS or Linux
- Python 3.x with conda
- ElevenLabs API key (or `gtts` as fallback)
- A working microphone

---

## Install

```bash
conda env create -f environment.yml
conda activate pronunciation
pip install gtts  # only needed if no ElevenLabs key
```

Add your ElevenLabs key to `.env` in the project root:

```
ELEVENLABS_API_KEY=sk_...
```

---

## Usage

### 1. Add phrases

Create `phrases/<TOPIC>/phrases.txt` — one phrase per line:

```
Whose computer is that on the desk?
I'll reach out to the dev team about the bug.
Let's touch base right after the deployment.
```

### 2. Generate audio

```bash
./pronunciation generate --phrases IT
```

Generates `phrases/IT/voices/1.mp3`, `2.mp3`, … and writes `phrases/IT/settings.json`.

Use a custom ElevenLabs voice:

```bash
./pronunciation generate --phrases IT --voice <voice_id>
```

Default voice: **Pepito** (`CIvegTlxTePhYSiBL1r4`).

If no API key is set, falls back to gTTS automatically.

### 3. Train

```bash
./pronunciation training --topic IT
./pronunciation training --topic IT --limit 5 --random --duration 7
```

**Flow per phrase:**
1. Phrase text is shown
2. Press Enter → audio plays
3. `[R]` repeat audio / `[N]` next
4. Mic opens — speak the phrase
5. `[R]` re-record / `[N]` next phrase

---

## Flags

### `generate`

| Flag | Default | Description |
|------|---------|-------------|
| `--phrases TOPIC` | required | Topic folder under `phrases/` |
| `--voice VOICE_ID` | `CIvegTlxTePhYSiBL1r4` | ElevenLabs voice ID |

### `training`

| Flag | Default | Description |
|------|---------|-------------|
| `--topic TOPIC` | required | Topic to practice |
| `--limit N` | all | Max phrases per session |
| `--duration SEC` | `5` | Recording length in seconds |
| `--random` | off | Shuffle phrases |

---

## Session output

Each session is saved to `sessions/<YYYYMMDD_HHMMSS_nanoseconds>/`:

```
sessions/20260519_143022_123456789/
  01/
    phrase.txt   ← the text
    audio.mp3    ← reference TTS audio
    me.wav       ← your recording
  02/
    ...
```

---

## Folder structure

```
phrases/
  IT/
    phrases.txt
    settings.json
    voices/
      1.mp3
      2.mp3
sessions/
  20260519_143022_123456789/
    01/
      phrase.txt
      audio.mp3
      me.wav
```
