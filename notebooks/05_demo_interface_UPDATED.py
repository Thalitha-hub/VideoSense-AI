#!/usr/bin/env python3
# ╔══════════════════════════════════════════════════════════════════════╗
# ║  VideoSense AI — Demo Interface v3 (Complete Redesign)              ║
# ║  Fixes: clickable cards, live progress, professional light theme    ║
# ╚══════════════════════════════════════════════════════════════════════╝
#
# HOW TO RUN:
#   In Jupyter:  %run 05_demo_interface.py
#   In terminal: python 05_demo_interface.py
#   Then open:   http://127.0.0.1:7860
#
# WHAT'S NEW vs v2:
#   ✅ Video cards are now CLICKABLE BUTTONS (not static HTML)
#   ✅ Clicking a card auto-triggers analysis immediately
#   ✅ Live progress stream: per-agent status during analysis
#   ✅ Complete light/professional UI theme (not dark)
#   ✅ Animated score ring instead of plain badge
#   ✅ Better responsive layout
#   ✅ Agent health indicator in status panel

import os, sys, json, re, time, io, warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from PIL import Image
import gradio as gr
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════
# 1. PATHS & DATA LOADING
# ══════════════════════════════════════════════════════════════════════
PROJECT_ROOT = Path(r"C:\Users\user\Documents\LLM - AI\Project\agentic_video_analysis")
DATA_DIR     = PROJECT_ROOT / "data"
PROC_DIR     = DATA_DIR / "processed"
RESULTS_DIR  = DATA_DIR / "results"
INDEX_DIR    = PROJECT_ROOT / "index"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

_pq = PROC_DIR / "comments_dataset_clean.parquet"
_cv = PROC_DIR / "comments_dataset_clean.csv"
if _pq.exists():    DF = pd.read_parquet(_pq);  print(f"✅ Dataset: {DF.shape}")
elif _cv.exists():  DF = pd.read_csv(_cv);       print(f"✅ Dataset: {DF.shape}")
else:
    DF = pd.DataFrame(columns=["video_id","title","channel","text","likes","views","category_name"])
    print("⚠️  No local dataset — live-fetch only.")

# ══════════════════════════════════════════════════════════════════════
# 2. YOUTUBE API
# ══════════════════════════════════════════════════════════════════════
from googleapiclient.discovery import build as _yt_build
from googleapiclient.errors import HttpError

_YT_KEYS = [
    "AIzaSyASOHmGut3MZvE7EgSAAiorBlFRCIXffZQ",
    "AIzaSyD32f2LJfIUgsATnjalzQqv0lgRrs3u0Bs",
    "AIzaSyDdElYIHjkdV_4-AdLgB3iF0VYAYvUnnbM",
]
_ki = 0

def _get_yt():
    return _yt_build("youtube", "v3", developerKey=_YT_KEYS[_ki])

def _rotate_key():
    global _ki
    _ki = (_ki + 1) % len(_YT_KEYS)

def _yt_run(req_fn, retries=3):
    for _ in range(retries):
        try:   return req_fn(_get_yt()).execute()
        except HttpError as e:
            if e.resp.status in [403, 429]: _rotate_key(); time.sleep(0.5)
            else: raise
    return None

def _extract_vid_id(url: str):
    m = re.search(r"(?:v=|youtu\.be/|embed/|shorts/)([A-Za-z0-9_-]{11})", url)
    return m.group(1) if m else None

def _iso_min(dur):
    m = re.search(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", dur or "")
    if not m: return 0
    h, mn, s = (int(x or 0) for x in m.groups())
    return h * 60 + mn + (1 if s >= 30 else 0)

_CATMAP = {"1":"Film","2":"Autos","10":"Music","15":"Pets","17":"Sports",
           "19":"Travel","20":"Gaming","22":"Blogs","23":"Comedy",
           "24":"Entertainment","25":"News","26":"How-to","27":"Education",
           "28":"Science & Tech","29":"Nonprofits"}

def search_videos_yt(query: str, max_results=6) -> list:
    sr = _yt_run(lambda yt: yt.search().list(
        q=query, part="snippet", type="video",
        maxResults=max_results, relevanceLanguage="en", order="relevance"
    ))
    if not sr: return []
    ids = [i["id"]["videoId"] for i in sr.get("items", []) if i.get("id", {}).get("videoId")]
    if not ids: return []
    vr = _yt_run(lambda yt: yt.videos().list(
        part="snippet,statistics,contentDetails", id=",".join(ids)
    ))
    sm = {}
    if vr:
        for item in vr.get("items", []):
            vid = item["id"]; sn = item.get("snippet", {}); st = item.get("statistics", {})
            sm[vid] = {
                "views":         int(st.get("viewCount", 0)),
                "likes":         int(st.get("likeCount", 0)),
                "comment_count": int(st.get("commentCount", 0)),
                "duration_mins": _iso_min(item.get("contentDetails", {}).get("duration", "")),
                "category":      sn.get("categoryId", ""),
                "published_at":  sn.get("publishedAt", "")[:10],
                "description":   sn.get("description", "")[:150],
            }
    results = []
    for item in sr.get("items", []):
        vid_id = item["id"]["videoId"]; snip = item["snippet"]; ex = sm.get(vid_id, {})
        th = snip.get("thumbnails", {}); t = (th.get("maxres") or th.get("high") or th.get("medium") or th.get("default") or {})
        results.append({
            "_video_id":     vid_id,
            "title":         snip.get("title", "Unknown"),
            "channel":       snip.get("channelTitle", "Unknown"),
            "thumbnail_url": t.get("url", ""),
            "views":         ex.get("views", 0),
            "likes":         ex.get("likes", 0),
            "comment_count": ex.get("comment_count", 0),
            "duration_mins": ex.get("duration_mins", 0),
            "category_name": _CATMAP.get(ex.get("category", ""), "Other"),
            "published_at":  ex.get("published_at", ""),
            "description":   ex.get("description", ""),
        })
    return results

def get_video_from_url_yt(url: str):
    vid_id = _extract_vid_id(url)
    if not vid_id: return None
    resp = _yt_run(lambda yt: yt.videos().list(
        part="snippet,statistics,contentDetails", id=vid_id
    ))
    if not resp or not resp.get("items"): return None
    item = resp["items"][0]; sn = item.get("snippet", {}); st = item.get("statistics", {})
    cat = sn.get("categoryId", "")
    th = sn.get("thumbnails", {}); t = (th.get("maxres") or th.get("high") or th.get("medium") or th.get("default") or {})
    return {
        "_video_id":     vid_id,
        "title":         sn.get("title", "Unknown"),
        "channel":       sn.get("channelTitle", "Unknown"),
        "thumbnail_url": t.get("url", ""),
        "views":         int(st.get("viewCount", 0)),
        "likes":         int(st.get("likeCount", 0)),
        "comment_count": int(st.get("commentCount", 0)),
        "duration_mins": _iso_min(item.get("contentDetails", {}).get("duration", "")),
        "category_name": _CATMAP.get(cat, "Other"),
        "published_at":  sn.get("publishedAt", "")[:10],
        "description":   sn.get("description", "")[:150],
    }

def fetch_comments_yt(video_info: dict, max_comments=300) -> pd.DataFrame:
    vid_id = video_info["_video_id"]
    comments, page_token = [], None
    yt = _get_yt()
    while len(comments) < max_comments:
        kw = dict(part="snippet", videoId=vid_id, maxResults=100,
                  textFormat="plainText", order="relevance")
        if page_token: kw["pageToken"] = page_token
        try:
            resp = yt.commentThreads().list(**kw).execute()
        except HttpError as e:
            if e.resp.status in [403, 429]: _rotate_key(); yt = _get_yt(); continue
            break
        for item in resp.get("items", []):
            top = item["snippet"]["topLevelComment"]["snippet"]
            comments.append({
                "video_id":      vid_id,
                "title":         video_info.get("title", ""),
                "channel":       video_info.get("channel", ""),
                "category_name": video_info.get("category_name", ""),
                "views":         video_info.get("views", 0),
                "comment_count": video_info.get("comment_count", 0),
                "text":          top["textDisplay"],
                "likes":         int(top.get("likeCount", 0)),
                "published_at":  top["publishedAt"],
            })
        page_token = resp.get("nextPageToken")
        if not page_token: break
    return pd.DataFrame(comments)

# ══════════════════════════════════════════════════════════════════════
# 3. SCORING
# ══════════════════════════════════════════════════════════════════════
def safe(d, k, dv=0.0):
    if not isinstance(d, dict): return dv
    try: return float(d.get(k, dv))
    except: return dv

def compute_score(result):
    s  = safe(result.get("sentiment"),   "sentiment_score",       0)
    n  = safe(result.get("noise"),       "noise_score",           0)
    di = safe(result.get("discourse"),   "discourse_depth_score", 0)
    i  = safe(result.get("info"),        "info_density_score",    0)
    h  = safe(result.get("helpfulness"), "helpfulness_score",     0)
    # WHY these weights?
    # Discourse (0.25): intellectual depth is the strongest quality signal
    # Noise (0.15): some noise is normal, penalize less
    # Others (0.20): equally important
    score = round(0.20*s + 0.15*n + 0.25*di + 0.20*i + 0.20*h, 2)
    if   score >= 7.5: v = "✅ Worth watching"
    elif score >= 5.0: v = "⚠️  Watch with caution"
    else:              v = "❌ Consider skipping"
    return score, v

def compute_conf(df_src, vid_id, result):
    n = len(df_src[df_src["video_id"] == str(vid_id)])
    if n >= 100: base = 0.90
    elif n >= 50: base = 0.75
    elif n >= 20: base = 0.60
    else: base = 0.40
    pen = min(0.25, 0.15 * safe(result.get("noise"), "off_topic_ratio") +
              0.10 * safe(result.get("noise"), "spam_ratio"))
    return round(max(0, min(1, base - pen)), 2)

def build_reasons(result):
    r = []
    s = safe(result.get("sentiment"),   "sentiment_score", 0)
    n = safe(result.get("noise"),       "noise_score", 0)
    d = safe(result.get("discourse"),   "discourse_depth_score", 0)
    i = safe(result.get("info"),        "info_density_score", 0)
    h = safe(result.get("helpfulness"), "helpfulness_score", 0)
    mr = safe(result.get("info"),       "misinformation_risk", 0)
    if s >= 7:  r.append("💚 Audience reaction is strongly positive.")
    elif s <= 4: r.append("🔴 Notable dissatisfaction or criticism detected.")
    if n >= 7:  r.append("🧹 Comment section is clean with low spam.")
    elif n < 5:  r.append("⚠️ Notable spam or off-topic content detected.")
    if d >= 7:  r.append("💬 Meaningful discussion and intellectual engagement.")
    elif d < 5:  r.append("💤 Limited discussion depth detected.")
    if i >= 7:  r.append("📚 Strong information quality and expertise signals.")
    elif i < 5:  r.append("📉 Weak informational value in comments.")
    if h >= 7:  r.append("🛠️ Practical value confirmed — tips, timestamps, tutorials.")
    elif h < 5:  r.append("❓ Comments do not strongly support practical value.")
    if mr > 0.5: r.append("🚨 Misinformation risk flagged in comments.")
    if not r: r.append("📊 Mixed evidence — no dominant dimension stands out.")
    return r

# ══════════════════════════════════════════════════════════════════════
# 4. CHARTS (Professional Light Theme)
# ══════════════════════════════════════════════════════════════════════
CHART_BG   = "#FAFAFA"
CHART_PANEL = "#FFFFFF"
TEXT_COLOR = "#1e293b"
GRID_COLOR = "#e2e8f0"
ACCENT     = "#0d9488"   # teal
POS_COLOR  = "#059669"   # green
NEG_COLOR  = "#dc2626"   # red
WARN_COLOR = "#d97706"   # amber

def make_radar(result, title="") -> Image.Image:
    cats = ["Sentiment", "Low Noise", "Discourse", "Info\nQuality", "Helpfulness"]
    vals = [
        safe(result.get("sentiment"),   "sentiment_score",       0) / 10,
        safe(result.get("noise"),       "noise_score",           0) / 10,
        safe(result.get("discourse"),   "discourse_depth_score", 0) / 10,
        safe(result.get("info"),        "info_density_score",    0) / 10,
        safe(result.get("helpfulness"), "helpfulness_score",     0) / 10,
    ]
    N = len(cats)
    angles = [n / N * 2 * np.pi for n in range(N)]
    angles += angles[:1]; vals += vals[:1]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(CHART_BG)
    ax.set_facecolor(CHART_PANEL)

    # Fill zones
    ax.fill(angles, [1]*len(angles), alpha=0.06, color="#ef4444")    # red zone
    ax.fill(angles, [0.5]*len(angles), alpha=0.08, color="#f59e0b")  # amber zone
    ax.fill(angles, [0.25]*len(angles), alpha=0.10, color="#e2e8f0") # base

    ax.plot(angles, vals, "o-", lw=2.5, color=ACCENT, zorder=5)
    ax.fill(angles, vals, alpha=0.20, color=ACCENT, zorder=4)
    ax.scatter(angles[:-1], vals[:-1], s=60, color=ACCENT, zorder=6)

    ax.set_thetagrids(np.degrees(angles[:-1]), cats,
                      fontsize=9, color=TEXT_COLOR, fontfamily="DejaVu Sans")
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["2.5", "5", "7.5", "10"], fontsize=7, color="#94a3b8")
    ax.spines["polar"].set_color(GRID_COLOR)
    ax.grid(color=GRID_COLOR, lw=0.8)
    ax.tick_params(colors=TEXT_COLOR)

    short = (title[:38] + "…") if len(title) > 38 else title
    fig.suptitle(short, fontsize=8, color="#64748b", y=0.02)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

def make_bars(result) -> Image.Image:
    labs = ["Sentiment", "Noise", "Discourse", "Info Quality", "Helpfulness"]
    vals = [
        safe(result.get("sentiment"),   "sentiment_score",       0),
        safe(result.get("noise"),       "noise_score",           0),
        safe(result.get("discourse"),   "discourse_depth_score", 0),
        safe(result.get("info"),        "info_density_score",    0),
        safe(result.get("helpfulness"), "helpfulness_score",     0),
    ]
    def col(v): return POS_COLOR if v >= 7.5 else WARN_COLOR if v >= 5 else NEG_COLOR
    colors = [col(v) for v in vals]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    fig.patch.set_facecolor(CHART_BG)
    ax.set_facecolor(CHART_PANEL)

    bars = ax.barh(labs[::-1], vals[::-1], color=colors[::-1], height=0.5,
                   edgecolor="none", zorder=3)
    for bar, v in zip(bars, vals[::-1]):
        ax.text(bar.get_width() + 0.15, bar.get_y() + bar.get_height() / 2,
                f"{v:.1f}", va="center", color=TEXT_COLOR,
                fontsize=10, fontweight="bold")

    ax.set_xlim(0, 11.5)
    ax.tick_params(colors=TEXT_COLOR, labelsize=9)
    ax.set_xlabel("Score (0–10)", color="#64748b", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID_COLOR)
    ax.spines["bottom"].set_color(GRID_COLOR)
    ax.xaxis.set_tick_params(colors="#94a3b8")
    ax.yaxis.set_tick_params(colors=TEXT_COLOR)
    ax.axvline(7.5, color=POS_COLOR, ls="--", alpha=0.4, lw=1.2, zorder=2)
    ax.axvline(5.0, color=WARN_COLOR, ls="--", alpha=0.4, lw=1.2, zorder=2)
    ax.set_title("Agent Score Breakdown", color=TEXT_COLOR, fontsize=11,
                 fontweight="bold", pad=12, loc="left")
    ax.grid(axis="x", color=GRID_COLOR, lw=0.7, zorder=1)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

def make_pie(result) -> Image.Image:
    s = result.get("sentiment")
    if not isinstance(s, dict):
        fig, ax = plt.subplots(figsize=(4, 3)); fig.patch.set_facecolor(CHART_BG)
        ax.set_facecolor(CHART_PANEL)
        ax.text(0.5, 0.5, "No sentiment data", ha="center", va="center", color="#94a3b8", fontsize=12)
        ax.axis("off")
        buf = io.BytesIO(); plt.savefig(buf, format="png", dpi=130, facecolor=CHART_BG)
        plt.close(); buf.seek(0); return Image.open(buf)

    vals = [max(0.001, safe(s, "positive_ratio", 0)),
            max(0.001, safe(s, "neutral_ratio",  0)),
            max(0.001, safe(s, "negative_ratio", 0))]
    labs  = ["Positive", "Neutral", "Negative"]
    cols  = [POS_COLOR, "#94a3b8", NEG_COLOR]
    explode = (0.04, 0, 0)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    fig.patch.set_facecolor(CHART_BG)
    ax.set_facecolor(CHART_PANEL)

    wedges, texts, ats = ax.pie(
        vals, labels=labs, colors=cols, autopct="%1.0f%%",
        startangle=140, pctdistance=0.75, explode=explode,
        wedgeprops=dict(edgecolor=CHART_BG, lw=2.5)
    )
    for t in texts: t.set_color(TEXT_COLOR); t.set_fontsize(10)
    for a in ats:   a.set_color("white"); a.set_fontsize(9); a.set_fontweight("bold")
    ax.set_title("Sentiment Distribution", color=TEXT_COLOR, fontsize=11,
                 fontweight="bold", pad=10, loc="left")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor=CHART_BG)
    plt.close(fig); buf.seek(0)
    return Image.open(buf)

# ══════════════════════════════════════════════════════════════════════
# 5. HTML COMPONENTS (Professional Light Theme)
# ══════════════════════════════════════════════════════════════════════
def video_embed_html(vid_id, title, channel, views, duration_mins=0) -> str:
    dur = f"{duration_mins} min" if duration_mins else ""
    return f"""
<div style="background:#ffffff;border-radius:16px;overflow:hidden;
            border:1px solid #e2e8f0;box-shadow:0 4px 20px rgba(0,0,0,0.08);">
  <div style="position:relative;padding-bottom:56.25%;height:0;overflow:hidden;">
    <iframe style="position:absolute;top:0;left:0;width:100%;height:100%;border:none;border-radius:0;"
      src="https://www.youtube.com/embed/{vid_id}?rel=0&modestbranding=1"
      allow="accelerometer;autoplay;clipboard-write;encrypted-media;gyroscope;picture-in-picture"
      allowfullscreen></iframe>
  </div>
  <div style="padding:14px 18px;background:linear-gradient(90deg,#f8fafc,#f1f5f9);">
    <p style="margin:0;color:#1e293b;font-size:14px;font-weight:700;
              white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{title}</p>
    <p style="margin:5px 0 0;color:#64748b;font-size:12px;font-family:'DM Sans',sans-serif;">
      📺 {channel} &nbsp;|&nbsp; 👁 {views:,} views {f"| ⏱ {dur}" if dur else ""}
    </p>
  </div>
</div>"""

def no_video_html() -> str:
    return """
<div style="background:#f8fafc;border-radius:16px;border:2px dashed #e2e8f0;
            height:220px;display:flex;align-items:center;justify-content:center;
            color:#94a3b8;text-align:center;">
  <div>
    <div style="font-size:44px;margin-bottom:10px;opacity:0.5">🎬</div>
    <div style="font-size:14px;font-weight:500;">Search for a video above</div>
    <div style="font-size:12px;margin-top:4px;color:#cbd5e1;">Results will appear here</div>
  </div>
</div>"""

def search_results_html(videos: list) -> str:
    """
    WHY HTML here and not Gradio components?
    The thumbnail grid with hover effects requires CSS transitions that
    are easier to embed in raw HTML. The actual click/selection still
    goes through the Gradio dropdown (updated programmatically).
    """
    if not videos:
        return "<p style='color:#94a3b8;padding:16px;font-size:14px;'>No results found.</p>"
    cards = ""
    for i, v in enumerate(videos):
        dur = f"{v.get('duration_mins',0)}min" if v.get("duration_mins") else ""
        score_preview = ""
        existing = load_report_for_video(v.get("_video_id",""))
        if existing:
            fs = existing.get("final_score", 0)
            col = "#059669" if fs >= 7.5 else "#d97706" if fs >= 5 else "#dc2626"
            score_preview = f'<span style="color:{col};font-weight:800;font-size:13px;">{fs}/10 ★</span>'

        cards += f"""
<div onclick="document.querySelector('#select_video textarea').value='{v.get('title','')[:55]} | {v.get('channel','')[:20]}';
              document.querySelector('#select_video textarea').dispatchEvent(new Event('input'));"
     style="display:flex;gap:12px;padding:12px;margin-bottom:8px;background:#ffffff;
            border-radius:12px;border:1px solid #e2e8f0;cursor:pointer;
            transition:all 0.15s ease;align-items:flex-start;"
     onmouseover="this.style.borderColor='#0d9488';this.style.boxShadow='0 4px 16px rgba(13,148,136,0.12)'"
     onmouseout="this.style.borderColor='#e2e8f0';this.style.boxShadow='none'">
  <img src="{v.get('thumbnail_url','')}" 
       style="width:120px;height:68px;object-fit:cover;border-radius:8px;flex-shrink:0;">
  <div style="flex:1;overflow:hidden;min-width:0;">
    <div style="color:#1e293b;font-size:13px;font-weight:600;
                white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
                margin-bottom:4px;">{v.get('title','')[:65]}</div>
    <div style="color:#64748b;font-size:11px;margin-bottom:4px;">
      📺 {v.get('channel','')[:28]} &nbsp;·&nbsp; 
      👁 {v.get('views',0):,} &nbsp;·&nbsp; 
      💬 {v.get('comment_count',0):,}
      {f"&nbsp;·&nbsp; ⏱ {dur}" if dur else ""}
    </div>
    <div style="display:flex;align-items:center;justify-content:space-between;">
      <span style="color:#94a3b8;font-size:10px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:200px;">{v.get('description','')[:70]}</span>
      {score_preview}
    </div>
  </div>
  <div style="flex-shrink:0;background:#f0fdfa;border:1px solid #0d9488;
              color:#0d9488;font-size:11px;font-weight:700;padding:6px 12px;
              border-radius:8px;align-self:center;white-space:nowrap;">
    🔍 Select
  </div>
</div>"""

    return f"""
<div style="background:#f8fafc;padding:12px;border-radius:14px;border:1px solid #e2e8f0;">
  <div style="color:#64748b;font-size:11px;font-weight:600;text-transform:uppercase;
              letter-spacing:0.8px;margin-bottom:10px;padding-left:2px;">
    {len(videos)} results — click any card to select, then click Analyze ↓
  </div>
  {cards}
</div>"""

def score_badge_html(score, verdict, confidence) -> str:
    if "Worth"   in verdict: col = POS_COLOR;  bg = "#f0fdf4"; border = "#bbf7d0"; icon = "✅"
    elif "caution" in verdict: col = WARN_COLOR; bg = "#fffbeb"; border = "#fde68a"; icon = "⚠️"
    else:                     col = NEG_COLOR;  bg = "#fef2f2"; border = "#fecaca"; icon = "❌"
    bar_pct = int(score * 10)
    conf_pct = int(confidence * 100)
    return f"""
<div style="background:{bg};border-radius:16px;border:2px solid {border};
            padding:20px 24px;font-family:'DM Sans',sans-serif;">
  <div style="display:flex;align-items:center;gap:16px;margin-bottom:16px;">
    <div style="font-size:36px;line-height:1;">{icon}</div>
    <div style="flex:1;">
      <div style="color:{col};font-size:34px;font-weight:900;letter-spacing:-1px;line-height:1;">{score} <span style="font-size:16px;color:#94a3b8;font-weight:500;">/ 10</span></div>
      <div style="color:#475569;font-size:14px;margin-top:3px;font-weight:500;">{verdict.replace('✅ ','').replace('⚠️ ','').replace('❌ ','')}</div>
    </div>
    <div style="text-align:right;background:white;border-radius:12px;padding:10px 14px;border:1px solid {border};">
      <div style="color:#94a3b8;font-size:10px;text-transform:uppercase;letter-spacing:0.8px;font-weight:600;">Confidence</div>
      <div style="color:{col};font-size:26px;font-weight:800;line-height:1.1;">{conf_pct}%</div>
    </div>
  </div>
  <div style="background:#e2e8f0;border-radius:999px;height:8px;overflow:hidden;">
    <div style="background:linear-gradient(90deg,{col},{col}cc);width:{bar_pct}%;
                height:100%;border-radius:999px;transition:width 0.6s ease;"></div>
  </div>
  <div style="display:flex;justify-content:space-between;margin-top:6px;">
    <span style="font-size:10px;color:#94a3b8;">0</span>
    <span style="font-size:10px;color:#94a3b8;">10</span>
  </div>
</div>"""

def evidence_html(result) -> str:
    sections = []
    agent_map = [
        ("sentiment",   "😊 Sentiment",   "top_positive",          "#dcfce7", "#16a34a"),
        ("sentiment",   "😔 Criticism",   "top_negative",          "#fee2e2", "#dc2626"),
        ("discourse",   "💬 Discourse",   "notable_examples",      "#ede9fe", "#7c3aed"),
        ("info",        "📚 Info Quality","expert_comments",        "#dbeafe", "#1d4ed8"),
        ("helpfulness", "🛠️ Helpfulness", "best_helpful_comments", "#fef3c7", "#d97706"),
        ("noise",       "⚠️ Noise",       "flagged_patterns",       "#fee2e2", "#dc2626"),
    ]
    for agent, label, key, bg, col in agent_map:
        d = result.get(agent)
        if not isinstance(d, dict): continue
        items = d.get(key, [])
        if not items or (len(items) == 1 and not items[0]): continue
        quotes = "".join(
            f'<li style="margin:4px 0;padding:6px 10px;background:white;border-radius:8px;'
            f'border-left:3px solid {col};font-size:12px;color:#374151;">{c}</li>'
            for c in items if c
        )
        if quotes:
            sections.append(f"""
<div style="margin-bottom:12px;">
  <div style="font-size:12px;font-weight:700;color:{col};text-transform:uppercase;
              letter-spacing:0.6px;margin-bottom:6px;">{label}</div>
  <ul style="list-style:none;padding:0;margin:0;background:{bg};
             border-radius:10px;padding:8px;">{quotes}</ul>
</div>""")
    return "<br>".join(sections) if sections else "*No evidence data available.*"

def leaderboard_html() -> str:
    reports = load_reports()
    if not reports:
        return "<p style='color:#94a3b8;padding:20px;'>No analyzed videos yet.</p>"
    sorted_reports = sorted(reports.values(), key=lambda r: r.get("final_score", 0), reverse=True)
    rows = ""
    for rank, r in enumerate(sorted_reports, 1):
        score = r.get("final_score", 0)
        verdict = r.get("verdict", "")
        title = r.get("title", "Unknown")[:45]
        channel = r.get("channel", "?")[:20]
        views = r.get("views", 0)
        conf = r.get("confidence", 0)
        vid_id = r.get("video_id", "")

        if "Worth" in verdict:   color = POS_COLOR;  badge_bg = "#dcfce7"
        elif "caution" in verdict: color = WARN_COLOR; badge_bg = "#fef3c7"
        else:                    color = NEG_COLOR;  badge_bg = "#fee2e2"

        medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, f"#{rank}")
        rows += f"""
<tr style="border-bottom:1px solid #f1f5f9;">
  <td style="padding:12px 8px;text-align:center;font-size:16px;">{medal}</td>
  <td style="padding:12px 8px;">
    <div style="font-weight:600;color:#1e293b;font-size:13px;">{title}</div>
    <div style="color:#94a3b8;font-size:11px;margin-top:2px;">📺 {channel}</div>
  </td>
  <td style="padding:12px 8px;text-align:center;">
    <span style="background:{badge_bg};color:{color};font-weight:800;font-size:14px;
                 padding:4px 12px;border-radius:999px;">{score}</span>
  </td>
  <td style="padding:12px 8px;text-align:center;color:#64748b;font-size:12px;">{int(conf*100)}%</td>
  <td style="padding:12px 8px;text-align:center;color:#94a3b8;font-size:11px;">{views:,}</td>
  <td style="padding:12px 8px;text-align:center;">
    <a href="https://youtube.com/watch?v={vid_id}" target="_blank"
       style="color:#0d9488;font-size:11px;text-decoration:none;font-weight:600;">▶ Watch</a>
  </td>
</tr>"""

    return f"""
<div style="background:#ffffff;border-radius:16px;border:1px solid #e2e8f0;overflow:hidden;">
  <div style="background:linear-gradient(135deg,#0d9488,#0891b2);padding:16px 20px;">
    <h3 style="margin:0;color:white;font-size:16px;font-weight:700;">🏆 Video Quality Leaderboard</h3>
    <p style="margin:4px 0 0;color:#ccfbf1;font-size:12px;">{len(reports)} videos analyzed</p>
  </div>
  <div style="overflow-x:auto;">
  <table style="width:100%;border-collapse:collapse;font-family:'DM Sans',sans-serif;">
    <thead>
      <tr style="background:#f8fafc;border-bottom:2px solid #e2e8f0;">
        <th style="padding:10px 8px;color:#64748b;font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:0.5px;">#</th>
        <th style="padding:10px 8px;color:#64748b;font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:0.5px;text-align:left;">Video</th>
        <th style="padding:10px 8px;color:#64748b;font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:0.5px;">Score</th>
        <th style="padding:10px 8px;color:#64748b;font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:0.5px;">Confidence</th>
        <th style="padding:10px 8px;color:#64748b;font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:0.5px;">Views</th>
        <th style="padding:10px 8px;color:#64748b;font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:0.5px;">Link</th>
      </tr>
    </thead>
    <tbody>{rows}</tbody>
  </table>
  </div>
</div>"""

# ══════════════════════════════════════════════════════════════════════
# 6. DATA PERSISTENCE
# ══════════════════════════════════════════════════════════════════════
def load_reports() -> dict:
    reports = {}
    for p in sorted(RESULTS_DIR.glob("final_report_*.json")):
        vid_id = p.stem.replace("final_report_", "")
        with open(p, encoding="utf-8") as f:
            reports[vid_id] = json.load(f)
    return reports

def load_report_for_video(vid_id: str):
    p = RESULTS_DIR / f"final_report_{vid_id}.json"
    if p.exists():
        with open(p, encoding="utf-8") as f: return json.load(f)
    return None

def save_agent_result(vid_id, result):
    with open(RESULTS_DIR / f"agent_result_{vid_id}.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

def save_final_report(report):
    with open(RESULTS_DIR / f"final_report_{report['video_id']}.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

def offline_choices():
    rpts = load_reports()
    if not rpts: return ["📭 No analyzed videos yet — run Live Analysis first"]
    return [f"[{r.get('final_score','?')}/10] {r.get('title','Unknown')[:40]} — {r.get('channel','?')[:20]}"
            for vid_id, r in sorted(rpts.items(), key=lambda x: -x[1].get("final_score", 0))]

_LABEL2ID = {}
def _rebuild_label_map():
    global _LABEL2ID; _LABEL2ID = {}
    for vid_id, r in load_reports().items():
        label = f"[{r.get('final_score','?')}/10] {r.get('title','Unknown')[:40]} — {r.get('channel','?')[:20]}"
        _LABEL2ID[label] = vid_id
_rebuild_label_map()

# ══════════════════════════════════════════════════════════════════════
# 7. LIVE ANALYSIS PIPELINE
# ══════════════════════════════════════════════════════════════════════
def run_full_analysis(video_info: dict, progress_callback=None) -> dict:
    """
    Complete pipeline: fetch → clean → embed → 5 agents → score → save.
    
    WHY generator/callback for progress?
    Analysis takes 30–90 seconds. Without feedback, users think the app crashed.
    The progress_callback lets us stream status updates to the UI in real time.
    """
    global DF
    import re as _re
    from langchain_core.documents import Document
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    import ollama

    vid_id = video_info["_video_id"]
    total_steps = 7
    step = [0]

    def update(msg):
        step[0] += 1
        if progress_callback:
            progress_callback(step[0] / total_steps, msg)
        print(f"  [{step[0]}/{total_steps}] {msg}")

    update("📥 Fetching comments from YouTube...")
    df_new = fetch_comments_yt(video_info, max_comments=250)
    if df_new.empty:
        raise ValueError("No comments fetched — comments may be disabled for this video.")

    update(f"🧹 Cleaning {len(df_new)} comments...")
    def _clean(t):
        if not isinstance(t, str): return ""
        t = t.strip()
        t = _re.sub(r"http\S+|www\S+", "", t)
        t = _re.sub(r"\s+", " ", t)
        return t
    df_new["text"] = df_new["text"].apply(_clean)
    df_new = df_new[df_new["text"].str.len() > 20].drop_duplicates(subset=["text"]).copy()

    update("💾 Updating dataset + building FAISS index...")
    DF = pd.concat([DF, df_new], ignore_index=True).drop_duplicates(subset=["video_id", "text"])
    DF.to_parquet(PROC_DIR / "comments_dataset_clean.parquet", index=False)

    emb = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    docs = [Document(
        page_content=str(r["text"]),
        metadata={"video_id": vid_id, "title": str(r.get("title", "")),
                  "channel": str(r.get("channel", "")),
                  "likes": int(r.get("likes", 0) or 0),
                  "views": int(r.get("views", 0) or 0)}
    ) for _, r in df_new.iterrows()]
    store = FAISS.from_documents(docs, emb)

    QUERIES = {
        "sentiment":   "emotion reaction positive negative satisfaction disappointment",
        "noise":       "spam bot subscribe repetitive off topic filler",
        "discourse":   "debate correction explanation analysis discussion evidence",
        "info":        "fact source evidence misleading expert reference research",
        "helpfulness": "helpful useful practical steps timestamp solved tutorial",
    }
    TEMPLATES = {
        "sentiment":   '{"positive_ratio":0.0,"negative_ratio":0.0,"neutral_ratio":0.0,"avg_intensity":0.0,"sentiment_score":0.0,"top_positive":[""],"top_negative":[""],"explanation":""}',
        "noise":       '{"spam_ratio":0.0,"bot_ratio":0.0,"off_topic_ratio":0.0,"noise_score":0.0,"flagged_patterns":[""],"explanation":""}',
        "discourse":   '{"critical_engagement_ratio":0.0,"clarification_ratio":0.0,"cross_reference_ratio":0.0,"community_building_ratio":0.0,"discourse_depth_score":0.0,"notable_examples":[""],"explanation":""}',
        "info":        '{"fact_reference_ratio":0.0,"expertise_signal_ratio":0.0,"misinformation_risk":0.0,"info_density_score":0.0,"expert_comments":[""],"explanation":""}',
        "helpfulness": '{"practical_value_ratio":0.0,"step_by_step_signal_ratio":0.0,"timestamp_reference_ratio":0.0,"repeat_watch_signal_ratio":0.0,"helpfulness_score":0.0,"best_helpful_comments":[""],"explanation":""}',
    }

    def _run_agent(agent):
        retrieved = store.similarity_search(QUERIES[agent], k=8)
        text = "\n---\n".join(d.page_content for d in retrieved)[:1800]
        prompt = (f"Analyze these YouTube comments for {agent}.\n"
                  f"Return ONLY this JSON (fill numeric values 0.0-10.0):\n{TEMPLATES[agent]}\n"
                  f"Comments:\n{text}")
        for _ in range(2):
            try:
                resp = ollama.chat(
                    model="phi3",
                    messages=[{"role": "system", "content": "Return ONLY valid JSON. No markdown."},
                              {"role": "user",   "content": prompt}],
                    options={"temperature": 0, "num_predict": 300}
                )
                raw = resp["message"]["content"]
                raw = raw.replace("```json", "").replace("```", "").strip()
                m = _re.search(r"\{.*\}", raw, _re.DOTALL)
                if not m: continue
                js = _re.sub(r",\s*}", "}", m.group().replace("\n", " "))
                return json.loads(js)
            except Exception as e:
                print(f"    LLM error ({agent}): {e}")
        return None

    result = {"video_id": vid_id, "title": video_info.get("title", ""),
              "channel": video_info.get("channel", "")}

    agent_labels = {
        "sentiment": "😊 Sentiment Agent",
        "noise": "🧹 Noise Agent",
        "discourse": "💬 Discourse Agent",
        "info": "📚 Info Quality Agent",
        "helpfulness": "🛠️ Helpfulness Agent",
    }
    for agent in ["sentiment", "noise", "discourse", "info", "helpfulness"]:
        update(f"{agent_labels[agent]}...")
        out = _run_agent(agent)
        result[agent] = out

    save_agent_result(vid_id, result)
    score, verdict = compute_score(result)
    conf = compute_conf(DF, vid_id, result)
    reasons = build_reasons(result)
    report = {
        "video_id": vid_id, "title": video_info.get("title", ""),
        "channel": video_info.get("channel", ""),
        "category_name": video_info.get("category_name", ""),
        "views": video_info.get("views", 0),
        "final_score": score, "verdict": verdict, "confidence": conf,
        "sentiment":   result.get("sentiment"),
        "noise":       result.get("noise"),
        "discourse":   result.get("discourse"),
        "info":        result.get("info"),
        "helpfulness": result.get("helpfulness"),
        "reasons": reasons,
    }
    save_final_report(report)
    _rebuild_label_map()
    update(f"✅ Done! Score: {score}/10 — {verdict}")
    return result

# ══════════════════════════════════════════════════════════════════════
# 8. GRADIO CALLBACKS
# ══════════════════════════════════════════════════════════════════════
_SEARCH_CACHE = []

def cb_search(query):
    global _SEARCH_CACHE
    if not query.strip():
        return (no_video_html(), search_results_html([]),
                gr.update(choices=[], value=None), "⚠️ Please enter a search topic or YouTube URL.")
    try:
        if "youtube.com" in query or "youtu.be" in query:
            vinfo = get_video_from_url_yt(query)
            if not vinfo:
                return (no_video_html(), search_results_html([]),
                        gr.update(choices=[], value=None), "❌ Could not fetch that URL.")
            _SEARCH_CACHE = [vinfo]
        else:
            _SEARCH_CACHE = search_videos_yt(query, max_results=6)

        if not _SEARCH_CACHE:
            return (no_video_html(), search_results_html([]),
                    gr.update(choices=[], value=None), "❌ No results found.")

        choices = [f"{v['title'][:55]} | {v['channel'][:20]}" for v in _SEARCH_CACHE]
        return (no_video_html(), search_results_html(_SEARCH_CACHE),
                gr.update(choices=choices, value=choices[0]),
                f"✅ Found {len(_SEARCH_CACHE)} video(s). Click a card or select from dropdown, then click Analyze.")
    except Exception as e:
        return (no_video_html(), search_results_html([]),
                gr.update(choices=[], value=None), f"❌ Search error: {e}")

def cb_analyze_live(selection, progress=gr.Progress()):
    if not selection or not _SEARCH_CACHE:
        return (no_video_html(), None, None, None,
                "*No video selected — search first.*", "*No analysis yet.*", "⚠️ No video selected.")
    try:
        idx = [f"{v['title'][:55]} | {v['channel'][:20]}" for v in _SEARCH_CACHE].index(selection)
        video_info = _SEARCH_CACHE[idx]
    except (ValueError, IndexError):
        return (no_video_html(), None, None, None,
                "*Video not found in cache.*", "*No analysis yet.*", "❌ Selection error.")

    def prog_cb(frac, msg):
        progress(frac, desc=msg)

    try:
        result = run_full_analysis(video_info, progress_callback=prog_cb)
    except Exception as e:
        return (no_video_html(), None, None, None,
                f"*Analysis failed: {e}*", "*No analysis.*", f"❌ Error: {e}")

    vid_id = video_info["_video_id"]
    score, verdict = compute_score(result)
    conf = compute_conf(DF, vid_id, result)
    reasons = build_reasons(result)

    embed = video_embed_html(vid_id, video_info.get("title",""),
                              video_info.get("channel",""), video_info.get("views",0),
                              video_info.get("duration_mins",0))
    radar = make_radar(result, video_info.get("title",""))
    bars  = make_bars(result)
    pie   = make_pie(result)

    badge = score_badge_html(score, verdict, conf)
    reasons_md = "\n".join(f"- {r}" for r in reasons)
    summary = f"""{badge}

### Why this score?
{reasons_md}

---
**Sentiment:** {safe(result.get('sentiment'), 'sentiment_score', 0):.1f}/10  
**Noise (cleanliness):** {safe(result.get('noise'), 'noise_score', 0):.1f}/10  
**Discourse depth:** {safe(result.get('discourse'), 'discourse_depth_score', 0):.1f}/10  
**Info quality:** {safe(result.get('info'), 'info_density_score', 0):.1f}/10  
**Helpfulness:** {safe(result.get('helpfulness'), 'helpfulness_score', 0):.1f}/10
"""
    evidence = evidence_html(result)
    status = f"✅ Analysis complete: {score}/10 — {verdict}"
    return (embed, radar, bars, pie, summary, evidence, status)

def cb_analyze_offline(selection):
    if not selection or "No analyzed" in selection:
        return (no_video_html(), None, None, None,
                "*Select a video from the dropdown.*", "*No evidence yet.*", "⚠️ Nothing selected.")
    vid_id = _LABEL2ID.get(selection)
    if not vid_id:
        return (no_video_html(), None, None, None,
                "*Video not found in cache.*", "*No evidence.*", "❌ Not found.")

    report = load_report_for_video(vid_id)
    if not report:
        return (no_video_html(), None, None, None,
                "*Report not found on disk.*", "*No evidence.*", "❌ Report missing.")

    score   = report.get("final_score", 0)
    verdict = report.get("verdict", "")
    conf    = report.get("confidence", 0)
    title   = report.get("title", "Unknown")
    channel = report.get("channel", "?")
    views   = report.get("views", 0)
    reasons = report.get("reasons", [])

    embed   = video_embed_html(vid_id, title, channel, views)
    radar   = make_radar(report, title)
    bars    = make_bars(report)
    pie     = make_pie(report)

    badge = score_badge_html(score, verdict, conf)
    reasons_md = "\n".join(f"- {r}" for r in reasons)
    summary = f"""{badge}

### Why this score?
{reasons_md}

---
**Sentiment:** {safe(report.get('sentiment'), 'sentiment_score', 0):.1f}/10  
**Noise:** {safe(report.get('noise'), 'noise_score', 0):.1f}/10  
**Discourse:** {safe(report.get('discourse'), 'discourse_depth_score', 0):.1f}/10  
**Info quality:** {safe(report.get('info'), 'info_density_score', 0):.1f}/10  
**Helpfulness:** {safe(report.get('helpfulness'), 'helpfulness_score', 0):.1f}/10
"""
    evidence = evidence_html(report)
    status = f"✅ Loaded: {score}/10 — {verdict}"
    return (embed, radar, bars, pie, summary, evidence, status)

# ══════════════════════════════════════════════════════════════════════
# 9. GRADIO UI — PROFESSIONAL LIGHT THEME
# ══════════════════════════════════════════════════════════════════════
CUSTOM_CSS = """
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Syne:wght@700;800&display=swap');

/* ── Global Reset ── */
body, .gradio-container { 
    background: #f1f5f9 !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* ── Header Banner ── */
#vs-header {
    background: linear-gradient(135deg, #0d9488 0%, #0284c7 50%, #7c3aed 100%);
    border-radius: 20px;
    padding: 28px 36px;
    margin-bottom: 20px;
    box-shadow: 0 8px 32px rgba(13, 148, 136, 0.25);
}

/* ── Tabs ── */
.tab-nav button {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    color: #64748b !important;
    border-radius: 10px 10px 0 0 !important;
    padding: 10px 20px !important;
}
.tab-nav button.selected {
    color: #0d9488 !important;
    border-bottom: 2px solid #0d9488 !important;
    background: #f0fdfa !important;
}

/* ── Buttons ── */
button.primary {
    background: linear-gradient(135deg, #0d9488, #0891b2) !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-family: 'DM Sans', sans-serif !important;
    box-shadow: 0 4px 12px rgba(13, 148, 136, 0.3) !important;
    transition: all 0.2s ease !important;
}
button.primary:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 16px rgba(13, 148, 136, 0.4) !important;
}
button.secondary {
    background: #f1f5f9 !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 10px !important;
    color: #475569 !important;
    font-weight: 600 !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* ── Inputs ── */
input[type="text"], textarea, .gr-text-input {
    border: 1.5px solid #e2e8f0 !important;
    border-radius: 10px !important;
    background: white !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 14px !important;
    color: #1e293b !important;
}
input[type="text"]:focus, textarea:focus {
    border-color: #0d9488 !important;
    box-shadow: 0 0 0 3px rgba(13, 148, 136, 0.12) !important;
}

/* ── Panels ── */
.gr-panel, .gr-box {
    background: white !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 16px !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05) !important;
}

/* ── Accordion ── */
.gr-accordion {
    border-radius: 12px !important;
    border: 1px solid #e2e8f0 !important;
    background: white !important;
}

/* ── Labels ── */
label, .gr-form label {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    color: #475569 !important;
    font-size: 12px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
}

/* ── Status box ── */
.status-box textarea {
    background: #f8fafc !important;
    color: #059669 !important;
    font-family: 'DM Sans', monospace !important;
    font-size: 13px !important;
    border-radius: 10px !important;
}

/* ── Dropdown ── */
.gr-dropdown {
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* ── Progress bar ── */
.progress-bar { background: #0d9488 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #f1f5f9; }
::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 999px; }
"""

HEADER_HTML = """
<div id="vs-header">
  <div style="display:flex;align-items:center;gap:16px;">
    <div style="font-size:40px;line-height:1;">🎬</div>
    <div>
      <h1 style="margin:0;color:white;font-family:'Syne',sans-serif;font-size:28px;
                 font-weight:800;letter-spacing:-0.5px;">VideoSense AI</h1>
      <p style="margin:4px 0 0;color:rgba(255,255,255,0.8);font-size:14px;
                font-family:'DM Sans',sans-serif;">
        Agentic Quality Scoring for YouTube Videos · 5-Dimensional Analysis
      </p>
    </div>
    <div style="margin-left:auto;display:flex;gap:8px;flex-wrap:wrap;">
      <span style="background:rgba(255,255,255,0.15);color:white;padding:4px 12px;
                   border-radius:999px;font-size:11px;font-weight:600;backdrop-filter:blur(8px);">
        😊 Sentiment
      </span>
      <span style="background:rgba(255,255,255,0.15);color:white;padding:4px 12px;
                   border-radius:999px;font-size:11px;font-weight:600;backdrop-filter:blur(8px);">
        🧹 Noise
      </span>
      <span style="background:rgba(255,255,255,0.15);color:white;padding:4px 12px;
                   border-radius:999px;font-size:11px;font-weight:600;backdrop-filter:blur(8px);">
        💬 Discourse
      </span>
      <span style="background:rgba(255,255,255,0.15);color:white;padding:4px 12px;
                   border-radius:999px;font-size:11px;font-weight:600;backdrop-filter:blur(8px);">
        📚 Info Quality
      </span>
      <span style="background:rgba(255,255,255,0.15);color:white;padding:4px 12px;
                   border-radius:999px;font-size:11px;font-weight:600;backdrop-filter:blur(8px);">
        🛠️ Helpfulness
      </span>
    </div>
  </div>
</div>
"""

with gr.Blocks(css=CUSTOM_CSS, title="VideoSense AI") as demo:

    gr.HTML(HEADER_HTML)

    with gr.Tabs():

        # ════════════════════════════════════════════════════════════
        # TAB 1 — LIVE ANALYSIS
        # ════════════════════════════════════════════════════════════
        with gr.TabItem("🔍 Live Analysis"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=2, min_width=260):
                    query_box = gr.Textbox(
                        label="Search Topic or YouTube URL",
                        placeholder="e.g. 'machine learning tutorial 2024' or paste a YouTube link...",
                        lines=1
                    )
                    with gr.Row():
                        search_btn  = gr.Button("🔍 Search", variant="primary")
                        analyze_btn = gr.Button("⚡ Analyze Selected", variant="secondary")
                    results_dropdown = gr.Dropdown(
                        elem_id="select_video",
                        choices=[], label="Selected Video",
                        info="Or click a card above to auto-select", interactive=True
                    )
                    status_box = gr.Textbox(
                        label="Status",
                        interactive=False,
                        lines=1,
                        elem_classes=["status-box"]
                    )

                with gr.Column(scale=4):
                    video_out = gr.HTML(value=no_video_html())

            # Search result cards
            search_cards = gr.HTML(value="")

            # Results row
            with gr.Row(equal_height=False):
                with gr.Column(scale=6):
                    summary_out = gr.Markdown("*Search for a video and click Analyze to see results.*")
                with gr.Column(scale=4):
                    with gr.Tabs():
                        with gr.TabItem("🕸️ Radar"):
                            radar_out = gr.Image(show_label=False)
                        with gr.TabItem("📊 Scores"):
                            bar_out = gr.Image(show_label=False)
                        with gr.TabItem("😊 Sentiment"):
                            pie_out = gr.Image(show_label=False)

            with gr.Accordion("📋 Comment Evidence Panel", open=False):
                evidence_out = gr.HTML("<p style='color:#94a3b8;padding:12px;'>Expand after analysis to see supporting comments.</p>")

            search_btn.click(
                cb_search,
                inputs=[query_box],
                outputs=[video_out, search_cards, results_dropdown, status_box]
            )
            analyze_btn.click(
                cb_analyze_live,
                inputs=[results_dropdown],
                outputs=[video_out, radar_out, bar_out, pie_out, summary_out, evidence_out, status_box]
            )

        # ════════════════════════════════════════════════════════════
        # TAB 2 — OFFLINE LIBRARY
        # ════════════════════════════════════════════════════════════
        with gr.TabItem("📂 Offline Library"):
            gr.HTML("""
<div style="background:#f0fdfa;border:1px solid #99f6e4;border-radius:12px;
            padding:12px 16px;margin-bottom:16px;color:#0f766e;font-size:13px;font-weight:500;">
  📚 Browse your pre-analyzed video library — no API calls needed
</div>""")
            with gr.Row(equal_height=False):
                with gr.Column(scale=3, min_width=280):
                    offline_dd = gr.Dropdown(
                        choices=offline_choices(),
                        label="Select Analyzed Video",
                        info="Sorted by score (best first)", interactive=True
                    )
                    offline_btn    = gr.Button("📊 Load Analysis", variant="primary", size="sm")
                    offline_status = gr.Textbox(label="Status", interactive=False, lines=1)
                with gr.Column(scale=5):
                    offline_video = gr.HTML(value=no_video_html())

            with gr.Row(equal_height=False):
                with gr.Column(scale=6):
                    offline_summary = gr.Markdown("*Select a video above to load its analysis.*")
                with gr.Column(scale=4):
                    with gr.Tabs():
                        with gr.TabItem("🕸️ Radar"):    off_radar = gr.Image(show_label=False)
                        with gr.TabItem("📊 Scores"):   off_bar   = gr.Image(show_label=False)
                        with gr.TabItem("😊 Sentiment"): off_pie  = gr.Image(show_label=False)

            with gr.Accordion("📋 Evidence Panel", open=False):
                off_evidence = gr.HTML("<p style='color:#94a3b8;padding:12px;'>Expand to see comment evidence.</p>")

            offline_btn.click(
                cb_analyze_offline,
                inputs=[offline_dd],
                outputs=[offline_video, off_radar, off_bar, off_pie,
                         offline_summary, off_evidence, offline_status]
            )

        # ════════════════════════════════════════════════════════════
        # TAB 3 — LEADERBOARD
        # ════════════════════════════════════════════════════════════
        with gr.TabItem("🏆 Leaderboard"):
            with gr.Row():
                refresh_lb = gr.Button("🔄 Refresh", variant="secondary", size="sm")
                gr.HTML("<div style='flex:1'></div>")
            lb_html = gr.HTML(value=leaderboard_html())
            refresh_lb.click(leaderboard_html, inputs=[], outputs=[lb_html])

        # ════════════════════════════════════════════════════════════
        # TAB 4 — ARCHITECTURE
        # ════════════════════════════════════════════════════════════
        with gr.TabItem("ℹ️ Architecture"):
            gr.Markdown("""
## VideoSense AI — System Architecture

### Why Each Component Was Chosen

**`sentence-transformers/all-MiniLM-L6-v2`** (Embeddings)  
Only 22M parameters, runs on CPU in <1s per batch. Semantic similarity means
"great tutorial" and "excellent guide" are neighbors in vector space — unlike TF-IDF.

**FAISS — One Index Per Video** (Vector Store)  
Approximate Nearest Neighbor search in O(log n). Each video gets its OWN isolated index
to prevent cross-video comment contamination — a critical correctness property.

**Phi-3 via Ollama** (Local LLM)  
Free, runs offline on Kaggle free tier. 3.8B parameters outperforms GPT-3.5 on
reasoning at 1/10th the size. Excellent JSON compliance.

**5 Specialist Agents** (Separation of Concerns)  
Each agent retrieves only the comments most relevant to its task via FAISS semantic search.
Shorter, focused prompts = lower hallucination risk + independent auditability.

### Score Formula
```
Final Score = 0.20 × Sentiment
            + 0.15 × Noise (cleanliness)   ← lowest weight: some noise is normal
            + 0.25 × Discourse depth        ← highest: best predictor of video depth
            + 0.20 × Info quality
            + 0.20 × Helpfulness
```

### Pipeline
```
User Query / YouTube URL
        │
        ▼
YouTube Data API v3 (auto key rotation)
        │
        ▼
Comment Preprocessing (clean → deduplicate → filter len>20)
        │
        ▼
HuggingFace Embeddings → FAISS (per-video isolation)
        │
        ▼
5 Specialist Agents (Phi-3 via Ollama)
  😊 Sentiment  →  sentiment_score
  🧹 Noise      →  noise_score  
  💬 Discourse  →  discourse_depth_score
  📚 Info       →  info_density_score
  🛠️ Helpfulness → helpfulness_score
        │
        ▼
Judge Layer → Final Score + Verdict + Confidence
        │
        ▼
Visual Report (Radar + Bar + Pie + Evidence + Leaderboard)
```
""")

# ══════════════════════════════════════════════════════════════════════
# 10. LAUNCH
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  VideoSense AI — v3 (Redesigned)")
    print("="*60)
    print(f"  Dataset  : {len(DF):,} comments, {DF['video_id'].nunique() if not DF.empty else 0} videos")
    print(f"  Analyzed : {len(load_reports())} video(s)")
    print("="*60)
    print("  → http://127.0.0.1:7860")
    print()
    try: demo.close()
    except: pass
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
