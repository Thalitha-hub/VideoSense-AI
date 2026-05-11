# 🎬 VideoSense AI
### Agentic YouTube Video Quality Assessment System

> Built during the **École d'Été en Intelligence Artificielle** · Yaoundé, Cameroon · March–April 2026  
> **Author:** Mopock Talla Ceverine Thalitha — Data Scientist

---

## 📌 What is VideoSense AI?

Have you ever clicked on a YouTube video and wasted 20 minutes on low-quality content?

**VideoSense AI** solves that problem. Instead of trusting easily manipulated view counts or misleading thumbnails, it reads what **real viewers say in the comments** — the one signal that cannot easily be faked — and gives the video a quality score from **0 to 10**, with a full explanation of *why*.

The system is **agentic**: five specialist AI agents each analyze a different dimension of comment quality. Their findings are combined by a **Judge Layer** into a final score, a verdict, and a confidence percentage — making it a fully **explainable AI (XAI)** system.

---

## 📊 Key Stats

| Metric | Value |
|---|---|
| Specialist AI Agents | 5 |
| Quality Score Range | 0 – 10 |
| Comments Analyzed per Video | Up to 250 |
| Embedding Dimensions | 384 |
| Verdict Categories | 3 |

---

## 🧠 How It Works — End-to-End Pipeline

| Step | Action | Output |
|---|---|---|
| ① Collect | Fetch comments via YouTube Data API v3 | Raw dataset (CSV/Parquet) |
| ② Clean | Remove spam, duplicates, short comments | Filtered quality comments |
| ③ Embed | Convert text → 384-dim vectors (all-MiniLM-L6-v2) | FAISS index per video |
| ④ Analyse | 5 specialist agents each query the FAISS index | JSON scores per dimension |
| ⑤ Judge | Weighted average + missing-score correction | Final score 0–10 + verdict |
| ⑥ Explain | Radar chart, bar chart, natural-language reasons | Human-readable report |
| ⑦ Demo | Gradio web interface at localhost:7860 | Interactive app for any user |

---

## 🤖 The Five Specialist Agents

| Agent | Weight | Role |
|---|---|---|
| 🟢 **Sentiment Agent** | ×0.20 | Measures overall emotional tone of comments |
| 🔴 **Noise Agent** | ×0.15 | Detects spam, bots, and promotional content |
| 🔵 **Discourse Agent** | ×0.25 | Measures intellectual depth — debates, questions, corrections |
| 🟡 **Info Quality Agent** | ×0.20 | Assesses facts, studies, expert knowledge & misinformation risk |
| 🟣 **Helpfulness Agent** | ×0.20 | Measures practical value — timestamps, solved problems, actionable tips |

> **Discourse carries the highest weight (25%)** because it best discriminates between genuinely educational videos and superficially popular ones.

---

## ⚖️ Scoring Formula

```
Final Score = (Sentiment × 0.20) + (Noise × 0.15) + (Discourse × 0.25)
            + (Info Quality × 0.20) + (Helpfulness × 0.20)
```

### Verdict Thresholds

| Score | Verdict | Meaning |
|---|---|---|
| ≥ 7.5 / 10 | ✅ Worth Watching | High quality — recommended |
| 5.0 – 7.4 | ⚠️ Watch with Caution | Mixed signals — verify yourself |
| < 5.0 | ❌ Consider Skipping | Low quality — likely not worth your time |

### 💡 Missing-Score Correction (Key Innovation)
A music video will naturally score near zero on Discourse — viewers don't cite academic papers in pop song comments. Without correction, this unfairly penalizes entertainment content.

**The fix:** any agent score ≤ 0.5 is treated as *Not Applicable*, removed from the calculation, and remaining weights are re-normalized to sum to 1.0.

---

## 🛠️ Tech Stack

| Technology | Role |
|---|---|
| YouTube Data API v3 | Data collection — comments & metadata |
| all-MiniLM-L6-v2 | Sentence embeddings → 384-dim vectors |
| FAISS | Vector store — per-video semantic search |
| Phi-3 (Ollama) | Local LLM for agent reasoning & JSON output |
| LangChain | Document management & FAISS interface |
| Gradio | Interactive web demo interface |
| Python / Jupyter | Core language & development environment |

---

## 📁 Project Structure

```
VideoSense-AI/
├── notebooks/
│   ├── 01_data_collection.ipynb       # YouTube API + cleaning
│   ├── 02_embeddings_faiss.ipynb      # Sentence embeddings + FAISS index
│   ├── 03_specialist_agents.ipynb     # Five AI agents
│   ├── 04_judge_layer.ipynb           # Scoring + explainability
│   └── 05_gradio_demo.ipynb           # Web interface
├── data/                              # Raw and cleaned comment datasets
├── README.md
└── requirements.txt
```

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/Thalitha-hub/VideoSense-AI.git
cd VideoSense-AI
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up your YouTube API Key
Create a `.env` file in the root folder:
```
YOUTUBE_API_KEY=your_api_key_here
```

### 4. Install and run Ollama with Phi-3
```bash
ollama pull phi3
ollama run phi3
```

### 5. Run the notebooks in order
Open Jupyter and run notebooks `01` through `05` in sequence.

### 6. Launch the Gradio Demo
Run notebook `05` and open your browser at:
```
http://127.0.0.1:7860
```

---

## 📈 Explainability Features

- **Radar Chart** — Five-axis chart showing all dimension scores simultaneously
- **Leaderboard** — Bar chart of all analyzed videos, color-coded by verdict
- **Natural Language Reasons** — Plain-English explanation of every verdict
- **Confidence %** — Explicit uncertainty quantification so you know how much to trust the score

---

## 🔒 Responsible AI & Ethics

- **Privacy:** Only publicly visible YouTube comments are analyzed — no PII collected or stored
- **Bias Awareness:** Missing-score correction prevents entertainment content from being unfairly penalized
- **Misinformation:** The misinformation risk score is explicitly labeled as an *estimate*, not ground truth
- **Transparency:** Every prediction includes a full explanation — satisfying the right-to-explanation principle
- **Environmental:** Local Ollama inference avoids cloud API calls, reducing carbon footprint

---

## 🔭 Future Directions

- 🌍 **Multilingual support** using LaBSE for non-English videos
- 👍 **User feedback loop** to rate verdict accuracy and create fine-tuning data
- 🦙 **Upgrade to Llama 3** for improved JSON reliability
- 🧩 **Browser extension** showing VideoSense scores directly on YouTube
- 🔌 **API endpoint** for third-party integration

---

## 👩🏾‍💻 Author

**Mopock Talla Ceverine Thalitha** — Data Scientist  
📧 ceverinethalitha@gmail.com  
🐙 [github.com/Thalitha-hub](https://github.com/Thalitha-hub)  
💼 [linkedin.com/in/talla-thalitha](https://linkedin.com/in/talla-thalitha)

---

> *VideoSense AI demonstrates that responsible, explainable AI systems can be built rapidly using open-source tools. Instead of trusting view counts or thumbnails, the system reads what real viewers said — and explains its reasoning transparently.*
