# AutoScore 🎙️🏏

**Bol ke score karo. Yaad ban ke ghar le jao.**

AutoScore is a voice-first scorekeeper for informal Indian gully cricket, powered by [Sarvam AI](https://www.sarvam.ai). Instead of tapping score buttons, just say what happened:

- *"Arjun ne covers ke upar four maara."*
- *"Wide ball, keeper ke paas se nikal gayi."*
- *"Sana caught at long on, kya pressure catch tha!"*

The app transcribes Hindi/English code-mixed speech, extracts structured cricket events, updates a live scorecard, generates dramatic Hinglish commentary, and creates a shareable WhatsApp-ready match recap.

---

## How it works

| Step | What happens | Sarvam API |
|------|-------------|------------|
| 🎤 Speak | Scorer records what happened in natural Hindi/English | — |
| 📝 Transcribe | Audio → text with cricket-aware keyterms | **Saaras v4** (`POST /speech-to-text`) |
| 🧠 Interpret | Transcript → structured JSON cricket event | **Sarvam-105B** (`POST /v1/chat/completions`) |
| 📊 Update | Score, players, wagon wheel update instantly | — |
| 💬 Memory | WhatsApp-ready match recap on demand | **Sarvam-105B** |

---

## Local setup

```bash
# 1. Clone and enter the project
cd Sarvam_hackathon

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your API key
#    Open .env and paste your key:
#    SARVAM_API_KEY=your_key_here

# 4. Run the app
streamlit run app.py
```

### Where to place `SARVAM_API_KEY`

**Option A — `.env` file** (local development):

```
SARVAM_API_KEY=your_key_here
```

**Option B — Streamlit secrets** (cloud deployment):

Create `.streamlit/secrets.toml`:

```toml
SARVAM_API_KEY = "your_key_here"
```

Both are in `.gitignore` — your key stays safe.

---

## 90-second live demo script

1. **Open the app** → Show the landing page with `AutoScore 🎙️🏏` header and `SARVAM-POWERED` badge.
2. **Demo mode** → Toggle OFF Live Sarvam mode. Click `🏏 Record this ball` three times — the pre-loaded calls will auto-cycle:
   - Four by Arjun → score jumps to `4/0`
   - Wide ball → score shows `5/0`
   - Sana caught → wicket falls, `5/1`
3. **Show tabs** → Flip through Match feed (commentary + transcripts), Scorecard (batting/bowling tables), and Wagon wheel (shot map).
4. **Match memory** → Go to Match memory tab, click `✨ Create match memory` → a WhatsApp-ready Hinglish recap appears.
5. **Live mode** (if Wi-Fi is stable) → Toggle ON Live Sarvam mode. Record a real voice call → show Saaras transcription and Sarvam-105B interpretation working end-to-end.
6. **Close** → "AutoScore preserves the personality of your local match — speak naturally, Sarvam handles the rest."

---

## Demo mode

Toggle `Live Sarvam mode` OFF to use a deterministic local parser. This ensures the demo works without Wi-Fi, mic permissions, or API credits. The mode badge is always visible so judges are never misled.

---

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io).
3. Select the repo and set `app.py` as the main file.
4. Add `SARVAM_API_KEY` in **Settings → Secrets**:
   ```toml
   SARVAM_API_KEY = "your_key_here"
   ```
5. Deploy.

---

## Stack

- **Python** + **Streamlit**
- **Sarvam AI** — Saaras v4 (STT) + Sarvam-105B (LLM)
- **Plotly** — Wagon wheel visualisation
- **Pandas** — Scorecard tables

No database, no auth, no extra infra. Just one file, one API key, one command.

---

Built for the Sarvam AI Hackathon 🇮🇳
