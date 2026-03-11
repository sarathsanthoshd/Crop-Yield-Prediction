"""
CropSense — Crop Yield Predictor  |  app.py
Run: streamlit run app.py
Needs: crop_yield_model_v2.joblib  +  unique_values_v2.joblib
"""

import streamlit as st
import joblib
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="CropSense — Yield Intelligence",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,700;0,800;1,700&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --ink:      #0d1f10;
  --forest:   #0e3d1e;
  --pine:     #165c2e;
  --fern:     #1f7d40;
  --grass:    #28a055;
  --leaf:     #3dbf6a;
  --sage:     #78d49a;
  --mist:     #b8e8c8;
  --foam:     #dff3e8;
  --cream:    #f8f5ef;
  --parch:    #f0ebe0;
  --sand:     #e4ddd0;
  --gold:     #b87a0a;
  --amber:    #d4920e;
  --honey:    #f0b429;
  --dawn:     #fdf6e8;
  --text:     #1a2e1e;
  --text2:    #3d5245;
  --text3:    #7a9080;
  --border:   #cfddd5;
  --white:    #ffffff;
  --shadow-sm: 0 1px 4px rgba(14,61,30,.06), 0 1px 2px rgba(14,61,30,.04);
  --shadow-md: 0 4px 16px rgba(14,61,30,.09), 0 2px 6px rgba(14,61,30,.05);
  --shadow-lg: 0 12px 40px rgba(14,61,30,.13), 0 4px 12px rgba(14,61,30,.07);
  --shadow-xl: 0 24px 64px rgba(14,61,30,.18), 0 8px 24px rgba(14,61,30,.10);
}

*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"], .stMarkdown {
  font-family: 'Plus Jakarta Sans', sans-serif !important;
  color: var(--text) !important;
}
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"], header {
  display: none !important;
}
[data-testid="stAppViewContainer"] {
  background-color: var(--cream) !important;
  background-image:
    radial-gradient(ellipse 80% 50% at 0% 0%, rgba(31,125,64,.06) 0%, transparent 60%),
    radial-gradient(ellipse 60% 40% at 100% 100%, rgba(212,146,14,.05) 0%, transparent 55%),
    url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='400' height='400'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.75' numOctaves='4' stitchTiles='stitch'/%3E%3CfeColorMatrix type='saturate' values='0'/%3E%3C/filter%3E%3Crect width='400' height='400' filter='url(%23n)' opacity='0.018'/%3E%3C/svg%3E") !important;
}
.block-container { max-width: 1140px !important; padding-top: 0 !important; padding-bottom: 5rem !important; }

/* ════════════════ NAVBAR ════════════════ */
.nav {
  background: var(--forest);
  height: 62px; padding: 0 2.5rem;
  margin: 0 -3rem;
  display: flex; align-items: center; justify-content: space-between;
  position: relative; overflow: hidden;
  box-shadow: 0 4px 28px rgba(14,61,30,.35);
}
.nav::before {
  content: '';
  position: absolute; inset: 0;
  background: linear-gradient(90deg, transparent 0%, rgba(61,191,106,.06) 50%, transparent 100%);
}
.nav::after {
  content: '';
  position: absolute; bottom: 0; left: 0; right: 0; height: 2px;
  background: linear-gradient(90deg, transparent 0%, var(--leaf) 30%, var(--honey) 60%, transparent 100%);
}
.nav-brand {
  font-family: 'Playfair Display', serif;
  font-size: 1.45rem; color: var(--white);
  display: flex; align-items: center; gap: .6rem;
  font-weight: 700; letter-spacing: .01em;
}
.nav-dot {
  width: 7px; height: 7px; border-radius: 50%;
  background: var(--honey);
  box-shadow: 0 0 10px var(--honey), 0 0 20px rgba(240,180,41,.4);
  animation: glow 2.5s ease-in-out infinite;
}
@keyframes glow { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:.5;transform:scale(1.4)} }
.nav-pills { display: flex; gap: .4rem; align-items: center; }
.nav-pill {
  background: rgba(255,255,255,.07);
  border: 1px solid rgba(255,255,255,.1);
  border-radius: 10px; padding: .28rem .75rem;
  text-align: center; min-width: 72px;
}
.nav-pill-v {
  font-family: 'JetBrains Mono', monospace;
  font-size: .82rem; font-weight: 500; color: var(--white); display: block; line-height: 1.3;
}
.nav-pill-l { font-size: .55rem; color: rgba(255,255,255,.45); text-transform: uppercase; letter-spacing: .08em; }

/* ════════════════ HERO ════════════════ */
.hero {
  background: linear-gradient(150deg, var(--forest) 0%, var(--pine) 40%, var(--fern) 80%, var(--grass) 100%);
  margin: 0 -3rem; padding: 3.5rem 3rem 4rem;
  position: relative; overflow: hidden;
  border-radius: 0 0 32px 32px;
  box-shadow: var(--shadow-lg);
}
.hero::before {
  content: '';
  position: absolute; top: -100px; right: -100px;
  width: 500px; height: 500px; border-radius: 50%;
  background: radial-gradient(circle, rgba(61,191,106,.12) 0%, transparent 65%);
  pointer-events: none;
}
.hero::after {
  content: '⬡';
  position: absolute; bottom: -60px; right: 2rem;
  font-size: 22rem; opacity: .025; color: var(--white);
  line-height: 1; pointer-events: none; user-select: none;
}
.hero-eyebrow {
  display: inline-flex; align-items: center; gap: .5rem;
  background: rgba(255,255,255,.1); border: 1px solid rgba(255,255,255,.18);
  border-radius: 100px; padding: .3rem 1rem .3rem .55rem;
  font-size: .68rem; font-weight: 600; letter-spacing: .1em;
  text-transform: uppercase; color: rgba(255,255,255,.9);
  margin-bottom: 1.1rem;
}
.hero-eyebrow-pip {
  width: 6px; height: 6px; border-radius: 50%; background: var(--sage);
  box-shadow: 0 0 6px var(--sage);
}
.hero-title {
  font-family: 'Playfair Display', serif;
  font-size: clamp(2.2rem, 4vw, 3.5rem);
  line-height: 1.12; font-weight: 700; color: var(--white);
  margin: 0 0 .9rem; letter-spacing: -.01em;
}
.hero-title i { font-style: italic; color: var(--sage); }
.hero-sub {
  font-size: .96rem; color: rgba(255,255,255,.7);
  line-height: 1.78; max-width: 480px; margin-bottom: 2.2rem; font-weight: 300;
}
.hero-sub strong { color: rgba(255,255,255,.95); font-weight: 600; }
.hero-kpis { display: flex; gap: 0; flex-wrap: wrap; }
.hero-kpi {
  display: flex; flex-direction: column; gap: .12rem;
  padding: 0 2.5rem 0 0;
  border-right: 1px solid rgba(255,255,255,.15);
  margin-right: 2.5rem;
}
.hero-kpi:last-child { border-right: none; margin-right: 0; }
.hero-kpi-v {
  font-family: 'Playfair Display', serif;
  font-size: 2.1rem; color: var(--white); line-height: 1;
}
.hero-kpi-l { font-size: .62rem; color: rgba(255,255,255,.5); text-transform: uppercase; letter-spacing: .1em; font-weight: 500; }

/* ════════════════ SECTION HEADER ════════════════ */
.sec {
  display: flex; align-items: center; gap: .85rem; margin: 2.4rem 0 1.2rem;
}
.sec-badge {
  background: var(--forest); color: var(--white);
  font-family: 'JetBrains Mono', monospace;
  font-size: .65rem; font-weight: 500;
  width: 32px; height: 32px; border-radius: 10px;
  display: flex; align-items: center; justify-content: center; flex-shrink: 0;
  box-shadow: 0 2px 10px rgba(14,61,30,.3);
  letter-spacing: .04em;
}
.sec-title {
  font-family: 'Playfair Display', serif;
  font-size: 1.15rem; color: var(--text); font-weight: 700;
}
.sec-rule { flex: 1; height: 1px; background: linear-gradient(90deg, var(--border) 0%, transparent 100%); }

/* ════════════════ AUTOFILL ════════════════ */
.autofill {
  background: linear-gradient(90deg, var(--foam) 0%, var(--white) 100%);
  border: 1.5px solid var(--mist);
  border-left: 3px solid var(--leaf);
  border-radius: 0 10px 10px 0; padding: .6rem 1rem;
  font-size: .8rem; color: var(--pine); font-weight: 500; margin-bottom: .85rem;
}

/* ════════════════ RESULT ════════════════ */
@keyframes rise { from{opacity:0;transform:translateY(28px)} to{opacity:1;transform:translateY(0)} }

.result-wrap { animation: rise .55s cubic-bezier(.22,1,.36,1) both; margin-bottom: .9rem; }
.result {
  background: linear-gradient(145deg, var(--forest) 0%, var(--pine) 45%, var(--fern) 80%, #1e8040 100%);
  border-radius: 22px; padding: 2.8rem 3rem 2.2rem;
  color: var(--white); position: relative; overflow: hidden;
  box-shadow: var(--shadow-xl);
}
.result::before {
  content: '';
  position: absolute; top: -80px; right: -80px;
  width: 320px; height: 320px; border-radius: 50%;
  background: radial-gradient(circle, rgba(255,255,255,.07) 0%, transparent 65%);
  pointer-events: none;
}
.result::after {
  content: '';
  position: absolute; bottom: 0; left: 0; right: 0; height: 3px;
  background: linear-gradient(90deg, var(--leaf) 0%, var(--honey) 50%, var(--leaf) 100%);
}
.result-top { display: flex; align-items: flex-start; justify-content: space-between; flex-wrap: wrap; gap: 1rem; }
.result-eyebrow {
  font-family: 'JetBrains Mono', monospace;
  font-size: .62rem; letter-spacing: .2em; text-transform: uppercase;
  color: rgba(255,255,255,.5); margin-bottom: .35rem;
}
.result-num {
  font-family: 'Playfair Display', serif;
  font-size: clamp(4rem, 8vw, 6.5rem);
  line-height: 1; font-weight: 700; letter-spacing: -.02em;
}
.result-unit { font-size: .88rem; color: rgba(255,255,255,.5); margin-top: .2rem; font-weight: 300; }
.result-tier {
  display: inline-flex; align-items: center; gap: .5rem;
  background: rgba(255,255,255,.12); border: 1px solid rgba(255,255,255,.2);
  border-radius: 100px; padding: .45rem 1.3rem;
  font-size: .82rem; font-weight: 700; letter-spacing: .02em;
  backdrop-filter: blur(6px); white-space: nowrap;
}
.result-chips { display: flex; flex-wrap: wrap; gap: .45rem; margin-top: 1.6rem; }
.rchip {
  background: rgba(255,255,255,.09); border: 1px solid rgba(255,255,255,.14);
  border-radius: 100px; padding: .24rem .85rem;
  font-size: .7rem; color: rgba(255,255,255,.82); font-weight: 400;
}

/* ════════════════ METRIC CARDS ════════════════ */
.metrics { display: grid; grid-template-columns: repeat(3,1fr); gap: .85rem; margin-bottom: .85rem; }
.mcard {
  background: var(--white);
  border: 1.5px solid var(--border);
  border-radius: 18px; padding: 1.4rem 1.3rem 1.2rem;
  box-shadow: var(--shadow-sm);
  transition: transform .2s ease, box-shadow .2s ease;
  position: relative; overflow: hidden;
}
.mcard::before {
  content: '';
  position: absolute; top: 0; left: 0; right: 0; height: 3px;
  background: linear-gradient(90deg, var(--leaf), var(--sage));
  border-radius: 18px 18px 0 0;
}
.mcard:hover { transform: translateY(-4px); box-shadow: var(--shadow-md); }
.mcard-row { display: flex; align-items: center; justify-content: space-between; margin-bottom: .65rem; }
.mcard-icon {
  width: 42px; height: 42px; border-radius: 12px;
  background: var(--parch); font-size: 1.3rem;
  display: flex; align-items: center; justify-content: center;
}
.mcard-tag {
  font-size: .58rem; font-weight: 700; text-transform: uppercase; letter-spacing: .07em;
  color: var(--pine); background: var(--foam); border-radius: 100px; padding: .18rem .6rem;
  border: 1px solid var(--mist);
}
.mcard-label {
  font-size: .62rem; font-weight: 700; text-transform: uppercase;
  letter-spacing: .09em; color: var(--text3); margin-bottom: .3rem;
}
.mcard-val {
  font-family: 'Playfair Display', serif;
  font-size: 1.55rem; color: var(--forest); line-height: 1.1;
}
.mcard-sub { font-size: .7rem; color: var(--text3); margin-top: .18rem; }

/* ════════════════ GAUGE ════════════════ */
.gauge-card {
  background: var(--white); border: 1.5px solid var(--border);
  border-radius: 18px; padding: 1.4rem 1.5rem; margin-bottom: .85rem;
  box-shadow: var(--shadow-sm);
}
.gauge-head { display: flex; justify-content: space-between; align-items: center; margin-bottom: .9rem; }
.gauge-title {
  font-size: .62rem; font-weight: 700; text-transform: uppercase;
  letter-spacing: .09em; color: var(--text3);
}
.gauge-val {
  font-family: 'JetBrains Mono', monospace;
  font-size: .82rem; font-weight: 500; color: var(--forest);
}
.gauge-track {
  background: var(--parch); border-radius: 99px;
  height: 14px; overflow: hidden; position: relative;
}
.gauge-fill {
  height: 100%; border-radius: 99px;
  background: linear-gradient(90deg, var(--sage) 0%, var(--grass) 50%, var(--forest) 100%);
  box-shadow: 0 0 14px rgba(40,160,85,.35);
  position: relative;
}
.gauge-fill::after {
  content: '';
  position: absolute; right: -1px; top: 50%; transform: translateY(-50%);
  width: 8px; height: 8px; border-radius: 50%;
  background: var(--white); border: 2px solid var(--forest);
  box-shadow: 0 1px 4px rgba(14,61,30,.3);
}
.gauge-scale {
  display: flex; justify-content: space-between; margin-top: .5rem;
  font-family: 'JetBrains Mono', monospace;
  font-size: .58rem; color: var(--text3);
}

/* ════════════════ SEASON COMPARE ════════════════ */
.sc-grid { display: grid; grid-template-columns: repeat(auto-fit,minmax(120px,1fr)); gap: .7rem; margin-top: .9rem; }
.sc-card {
  background: var(--parch); border: 1.5px solid var(--sand);
  border-radius: 14px; padding: .95rem .8rem; text-align: center;
  transition: transform .2s, box-shadow .2s;
}
.sc-card:hover { transform: translateY(-2px); box-shadow: var(--shadow-sm); }
.sc-card.best {
  background: linear-gradient(135deg, #e8f8ee 0%, #f0fbf3 100%);
  border-color: var(--grass);
  box-shadow: 0 0 0 3px rgba(40,160,85,.1);
}
.sc-label { font-size: .6rem; text-transform: uppercase; letter-spacing: .08em; color: var(--text3); font-weight: 700; }
.sc-val { font-family: 'Playfair Display', serif; font-size: 1.15rem; color: var(--forest); margin-top: .25rem; }

/* ════════════════ TIP ════════════════ */
.tip {
  background: linear-gradient(135deg, var(--dawn) 0%, #fff9e5 100%);
  border: 1.5px solid var(--honey); border-left: 4px solid var(--amber);
  border-radius: 0 14px 14px 0; padding: 1.1rem 1.3rem;
  font-size: .85rem; color: #5c3a06; line-height: 1.7;
  margin-bottom: .85rem; display: flex; gap: .8rem; align-items: flex-start;
}
.tip-icon { font-size: 1.15rem; flex-shrink: 0; margin-top: .05rem; }

/* ════════════════ FOOTER ════════════════ */
.ftr {
  text-align: center; padding: 2rem 0 .5rem;
  font-family: 'JetBrains Mono', monospace;
  font-size: .66rem; color: var(--text3);
  border-top: 1px solid var(--border); margin-top: 3.5rem;
  letter-spacing: .06em;
}

/* ════════════════ RESPONSIVE ════════════════ */
@media (max-width: 768px) {
  .nav, .hero { margin: 0 -1rem; }
  .hero { border-radius: 0 0 20px 20px; padding: 2rem 1.2rem 2.5rem; }
  .hero::after { display: none; }
  .nav-pills { display: none; }
  .metrics { grid-template-columns: 1fr; }
  .result-num { font-size: 3.5rem; }
  .hero-title { font-size: 2rem; }
  .hero-kpis { gap: 0; }
  .hero-kpi { padding: 0 1.5rem 0 0; margin-right: 1.5rem; }
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return joblib.load("crop_yield_model_v2.joblib")

@st.cache_data
def load_metadata():
    return joblib.load("unique_values_v2.joblib")

try:
    model         = load_model()
    metadata      = load_metadata()
    uv            = metadata['unique_values']
    state_climate = metadata['state_climate']
    crop_type_map = metadata['crop_type_map']
    SEASON_MONTH  = metadata.get('season_default_month',
                        {'Kharif': 6, 'Rabi': 11, 'Summer': 3, 'Whole Year': 1})
    MODEL_R2      = metadata.get('model_r2',  0.9847)
    MODEL_MAE     = metadata.get('model_mae', 0.5435)
    crop_seasons  = {
        'Arhar/Tur':         ['Kharif','Rabi','Whole Year'],
        'Bajra':             ['Kharif','Rabi','Summer','Whole Year'],
        'Castor seed':       ['Kharif','Rabi','Whole Year'],
        'Cotton(lint)':      ['Kharif','Rabi','Summer','Whole Year'],
        'Dry chillies':      ['Kharif','Rabi','Summer','Whole Year'],
        'Gram':              ['Kharif','Rabi','Whole Year'],
        'Groundnut':         ['Kharif','Rabi','Summer','Whole Year'],
        'Jowar':             ['Kharif','Rabi','Summer','Whole Year'],
        'Jute':              ['Kharif','Rabi','Summer'],
        'Linseed':           ['Kharif','Rabi','Whole Year'],
        'Maize':             ['Kharif','Rabi','Summer','Whole Year'],
        'Mesta':             ['Kharif','Rabi','Whole Year'],
        'Moong(Green Gram)': ['Kharif','Rabi','Summer','Whole Year'],
        'Niger seed':        ['Kharif','Rabi','Whole Year'],
        'Onion':             ['Kharif','Rabi','Summer','Whole Year'],
        'Paddy':             ['Kharif','Rabi','Summer','Whole Year'],
        'Potato':            ['Kharif','Rabi','Summer','Whole Year'],
        'Ragi':              ['Kharif','Rabi','Summer','Whole Year'],
        'Rapeseed &Mustard': ['Kharif','Rabi','Whole Year'],
        'Rice':              ['Kharif','Rabi','Summer','Whole Year'],
        'Safflower':         ['Kharif','Rabi','Whole Year'],
        'Sesamum':           ['Kharif','Rabi','Summer','Whole Year'],
        'Small millets':     ['Kharif','Rabi','Summer','Whole Year'],
        'Soyabean':          ['Kharif','Rabi','Whole Year'],
        'Sugarcane':         ['Kharif','Rabi','Summer','Whole Year'],
        'Sunflower':         ['Kharif','Rabi','Summer','Whole Year'],
        'Tobacco':           ['Kharif','Rabi','Summer','Whole Year'],
        'Urad':              ['Kharif','Rabi','Summer','Whole Year'],
        'Wheat':             ['Kharif','Rabi','Summer','Whole Year'],
    }
except FileNotFoundError as e:
    st.error(f"⚠️  Model files not found. Run `python train_model.py` first.\n\n{e}")
    st.stop()

STATE_REGION = {
    'Haryana':'North','Punjab':'North','Uttar Pradesh':'North','Uttarakhand':'North',
    'Himachal Pradesh':'North','Jammu and Kashmir':'North','Delhi':'North',
    'Andhra Pradesh':'South','Karnataka':'South','Kerala':'South',
    'Tamil Nadu':'South','Telangana':'South','Puducherry':'South',
    'Bihar':'East','Jharkhand':'East','Odisha':'East','West Bengal':'East',
    'Goa':'West','Gujarat':'West','Maharashtra':'West',
    'Chhattisgarh':'Central','Madhya Pradesh':'Central',
    'Arunachal Pradesh':'North-East','Assam':'North-East','Manipur':'North-East',
    'Meghalaya':'North-East','Mizoram':'North-East','Nagaland':'North-East',
    'Sikkim':'North-East','Tripura':'North-East',
}
MONTH_NAMES = {
    1:'January',2:'February',3:'March',4:'April',5:'May',6:'June',
    7:'July',8:'August',9:'September',10:'October',11:'November',12:'December'
}

if 'selected_crop' not in st.session_state:
    st.session_state.selected_crop = sorted(uv['Crop'])[0]
if 'selected_season' not in st.session_state:
    st.session_state.selected_season = crop_seasons[st.session_state.selected_crop][0]

def on_crop_change():
    new_crop = st.session_state._crop_selector
    valid    = crop_seasons.get(new_crop, ['Kharif'])
    st.session_state.selected_crop   = new_crop
    st.session_state.selected_season = valid[0]

# ── Navbar ─────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="nav">
  <div class="nav-brand">
    <div class="nav-dot"></div>
    CropSense
  </div>
  <div class="nav-pills">
    <div class="nav-pill">
      <span class="nav-pill-v">{MODEL_R2:.4f}</span>
      <span class="nav-pill-l">R² Score</span>
    </div>
    <div class="nav-pill">
      <span class="nav-pill-v">{MODEL_MAE:.3f} t/ha</span>
      <span class="nav-pill-l">Avg Error</span>
    </div>
    <div class="nav-pill">
      <span class="nav-pill-v">89,000</span>
      <span class="nav-pill-l">Data Rows</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero">
  <div class="hero-eyebrow">
    <div class="hero-eyebrow-pip"></div>
    Precision Agriculture &nbsp;·&nbsp; Machine Learning
  </div>
  <div class="hero-title">Predict Your <i>Crop Yield</i></div>
  <div class="hero-sub">
    Enter your farm details — crop variety, location, soil, climate, and inputs.
    Get a precise estimate in <strong>tonnes per hectare</strong> backed by 89,000 data points.
  </div>
  <div class="hero-kpis">
    <div class="hero-kpi">
      <span class="hero-kpi-v">{MODEL_R2:.2f}</span>
      <span class="hero-kpi-l">R² Accuracy</span>
    </div>
    <div class="hero-kpi">
      <span class="hero-kpi-v">29</span>
      <span class="hero-kpi-l">Crop Varieties</span>
    </div>
    <div class="hero-kpi">
      <span class="hero-kpi-v">30</span>
      <span class="hero-kpi-l">Indian States</span>
    </div>
    <div class="hero-kpi">
      <span class="hero-kpi-v">4</span>
      <span class="hero-kpi-l">Seasons</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 01 — Crop Identity  (outside form — live season update)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="sec">
  <div class="sec-badge">01</div>
  <div class="sec-title">Crop Identity</div>
  <div class="sec-rule"></div>
</div>""", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
with c1:
    st.selectbox("Crop Variety", options=sorted(uv['Crop']),
                 index=sorted(uv['Crop']).index(st.session_state.selected_crop),
                 key="_crop_selector", on_change=on_crop_change)
    crop = st.session_state.selected_crop

with c2:
    valid_seasons = crop_seasons.get(crop, ['Kharif'])
    if st.session_state.selected_season not in valid_seasons:
        st.session_state.selected_season = valid_seasons[0]
    season = st.selectbox("Growing Season", options=valid_seasons,
                          index=valid_seasons.index(st.session_state.selected_season),
                          key="_season_selector")
    st.session_state.selected_season = season

with c3:
    crop_year = st.number_input("Crop Year", min_value=1997, max_value=2035, value=2024, step=1)

# ══════════════════════════════════════════════════════════════════════════════
# STEPS 02–05 inside form
# ══════════════════════════════════════════════════════════════════════════════
with st.form("farm_details_form"):

    st.markdown("""
    <div class="sec">
      <div class="sec-badge">02</div>
      <div class="sec-title">Location &amp; Timing</div>
      <div class="sec-rule"></div>
    </div>""", unsafe_allow_html=True)

    c4, c5, c6 = st.columns(3)
    with c4:
        state = st.selectbox("State", sorted(uv['State']))
    with c5:
        default_month = SEASON_MONTH.get(season, 6)
        harvest_month = st.selectbox("Harvest / Sowing Month", options=list(range(1,13)),
                                     format_func=lambda m: MONTH_NAMES[m],
                                     index=default_month - 1,
                                     help="Auto-filled from season. Override if needed.")
    with c6:
        area = st.number_input("Area (hectares)", min_value=0.5, max_value=5_000_000.0,
                               value=1000.0, step=100.0, format="%.1f")

    st.markdown("""
    <div class="sec">
      <div class="sec-badge">03</div>
      <div class="sec-title">Soil &amp; Irrigation</div>
      <div class="sec-rule"></div>
    </div>""", unsafe_allow_html=True)

    c7, c8 = st.columns(2)
    with c7:
        soil_type  = st.selectbox("Soil Type",         sorted(uv['Soil_Type']))
    with c8:
        irrigation = st.selectbox("Irrigation Method", sorted(uv['Irrigation_Type']))

    st.markdown("""
    <div class="sec">
      <div class="sec-badge">04</div>
      <div class="sec-title">Climate Conditions</div>
      <div class="sec-rule"></div>
    </div>""", unsafe_allow_html=True)

    clim = state_climate.get(state, {
        'Annual_Rainfall':900.0,'Humidity':55.0,
        'Avg_Temperature':25.0,'Max_Temperature':35.0,'Min_Temperature':15.0
    })
    st.markdown(
        f'<div class="autofill">✦ Auto-filled with <strong>{state}</strong> climate averages — adjust if your local conditions differ.</div>',
        unsafe_allow_html=True)

    c9, c10, c11, c12, c13 = st.columns(5)
    with c9:
        annual_rainfall = st.number_input("Rainfall (mm)", min_value=50.0, max_value=6000.0,
                                          value=float(clim['Annual_Rainfall']), step=10.0)
    with c10:
        humidity = st.number_input("Humidity (%)", min_value=10.0, max_value=100.0,
                                   value=float(clim['Humidity']), step=1.0)
    with c11:
        avg_temp = st.number_input("Avg Temp (°C)", min_value=-20.0, max_value=50.0,
                                   value=float(clim['Avg_Temperature']), step=0.5)
    with c12:
        max_temp = st.number_input("Max Temp (°C)", min_value=-10.0, max_value=60.0,
                                   value=float(clim['Max_Temperature']), step=0.5)
    with c13:
        min_temp = st.number_input("Min Temp (°C)", min_value=-25.0, max_value=45.0,
                                   value=float(clim['Min_Temperature']), step=0.5)

    st.markdown("""
    <div class="sec">
      <div class="sec-badge">05</div>
      <div class="sec-title">Farm Inputs</div>
      <div class="sec-rule"></div>
    </div>""", unsafe_allow_html=True)

    ci1, ci2, _, _ = st.columns(4)
    with ci1:
        fertilizer_ha = st.number_input("Fertilizer (kg/ha)", min_value=0.0, max_value=500.0,
                                        value=145.0, step=5.0)
    with ci2:
        pesticide_ha = st.number_input("Pesticide (kg/ha)", min_value=0.0, max_value=2.0,
                                       value=0.27, step=0.01, format="%.3f")

    st.write("")
    submitted = st.form_submit_button("🌾  Predict Crop Yield", use_container_width=True)

# ── Prediction ─────────────────────────────────────────────────────────────────
if submitted:
    region    = STATE_REGION.get(state, 'Central')
    crop_type = crop_type_map.get(crop, 'Cereal')
    month_sin = float(np.sin(2 * np.pi * harvest_month / 12))
    month_cos = float(np.cos(2 * np.pi * harvest_month / 12))

    input_df = pd.DataFrame([{
        "Crop":crop, "Crop_Type":crop_type, "Season_clean":season,
        "State":state, "Region":region, "Soil_Type":soil_type,
        "Irrigation_Type":irrigation, "Crop_Year":crop_year, "Area":area,
        "Annual_Rainfall":annual_rainfall, "Humidity":humidity,
        "Avg_Temperature":avg_temp, "Max_Temperature":max_temp,
        "Min_Temperature":min_temp,
        "Fertilizer_per_Hectare":fertilizer_ha,
        "Pesticide_per_Hectare":pesticide_ha,
        "Month_sin":month_sin, "Month_cos":month_cos,
    }])

    prediction       = max(0.01, float(np.expm1(model.predict(input_df)[0])))
    total_production = prediction * area

    if   prediction < 1.0:  tier, tier_icon = "Below Average","⚠️"
    elif prediction < 2.5:  tier, tier_icon = "Average",      "📊"
    elif prediction < 5.0:  tier, tier_icon = "Good",         "✅"
    elif prediction < 12.0: tier, tier_icon = "Excellent",    "🌟"
    else:                   tier, tier_icon = "Outstanding",  "🏆"

    gauge_pct = min(100, int(np.log1p(prediction) / np.log1p(20) * 100))

    # Result banner
    st.markdown(f"""
    <div class="result-wrap">
      <div class="result">
        <div class="result-top">
          <div>
            <div class="result-eyebrow">Estimated Yield &nbsp;—&nbsp; {crop} &nbsp;·&nbsp; {state}</div>
            <div class="result-num">{prediction:.2f}</div>
            <div class="result-unit">tonnes per hectare</div>
          </div>
          <div class="result-tier">{tier_icon}&nbsp; {tier} Yield</div>
        </div>
        <div class="result-chips">
          <span class="rchip">🌿 {crop_type}</span>
          <span class="rchip">📍 {region} India</span>
          <span class="rchip">🗓️ {MONTH_NAMES[harvest_month]} {crop_year}</span>
          <span class="rchip">🌱 {soil_type}</span>
          <span class="rchip">💧 {irrigation}</span>
          <span class="rchip">☀️ {season}</span>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

    # Metric cards
    st.markdown(f"""
    <div class="metrics">
      <div class="mcard">
        <div class="mcard-row">
          <div class="mcard-icon">🏭</div>
          <div class="mcard-tag">Output</div>
        </div>
        <div class="mcard-label">Total Production</div>
        <div class="mcard-val">{total_production:,.1f} t</div>
        <div class="mcard-sub">across {area:,.0f} ha cultivated</div>
      </div>
      <div class="mcard">
        <div class="mcard-row">
          <div class="mcard-icon">💊</div>
          <div class="mcard-tag">Input</div>
        </div>
        <div class="mcard-label">Fertilizer Applied</div>
        <div class="mcard-val">{fertilizer_ha:.0f} kg/ha</div>
        <div class="mcard-sub">Total: {fertilizer_ha * area:,.0f} kg</div>
      </div>
      <div class="mcard">
        <div class="mcard-row">
          <div class="mcard-icon">🌡️</div>
          <div class="mcard-tag">Climate</div>
        </div>
        <div class="mcard-label">Temperature Range</div>
        <div class="mcard-val">{min_temp:.0f}° – {max_temp:.0f}°C</div>
        <div class="mcard-sub">Avg {avg_temp:.1f}°C · {humidity:.0f}% humidity</div>
      </div>
    </div>""", unsafe_allow_html=True)

    # Gauge
    st.markdown(f"""
    <div class="gauge-card">
      <div class="gauge-head">
        <span class="gauge-title">Yield Performance Scale</span>
        <span class="gauge-val">{prediction:.2f} t/ha &nbsp;—&nbsp; {tier}</span>
      </div>
      <div class="gauge-track">
        <div class="gauge-fill" style="width:{gauge_pct}%"></div>
      </div>
      <div class="gauge-scale">
        <span>0</span><span>Low &lt;1</span>
        <span>Avg 2–3</span><span>Good 5–8</span><span>20+ t/ha</span>
      </div>
    </div>""", unsafe_allow_html=True)

    # Season comparison
    all_seasons = crop_seasons.get(crop, [season])
    if len(all_seasons) > 1:
        season_results = {}
        for alt_s in all_seasons:
            alt_df = input_df.copy()
            alt_df['Season_clean'] = alt_s
            alt_m  = SEASON_MONTH.get(alt_s, harvest_month)
            alt_df['Month_sin'] = np.sin(2 * np.pi * alt_m / 12)
            alt_df['Month_cos'] = np.cos(2 * np.pi * alt_m / 12)
            season_results[alt_s] = max(0.01, float(np.expm1(model.predict(alt_df)[0])))
        best_season = max(season_results, key=season_results.get)
        cards = "".join([
            f'<div class="sc-card {"best" if s==best_season else ""}">'
            f'<div class="sc-label">{"🏆 " if s==best_season else ""}{s}</div>'
            f'<div class="sc-val">{v:.2f} t/ha</div></div>'
            for s, v in season_results.items()
        ])
        st.markdown(f"""
        <div class="gauge-card">
          <div class="gauge-head">
            <span class="gauge-title">All-Season Comparison — {crop}</span>
            <span class="gauge-val">Best: {best_season}</span>
          </div>
          <div class="sc-grid">{cards}</div>
        </div>""", unsafe_allow_html=True)

    # Drip tip
    if irrigation != "Drip":
        alt_drip = input_df.copy()
        alt_drip["Irrigation_Type"] = "Drip"
        drip_pred  = max(0.01, float(np.expm1(model.predict(alt_drip)[0])))
        drip_delta = drip_pred - prediction
        drip_pct   = drip_delta / prediction * 100
        if drip_delta > 0:
            st.markdown(f"""
            <div class="tip">
              <div class="tip-icon">💡</div>
              <div><strong>Drip Irrigation Tip:</strong>
              Switching to drip irrigation could push your yield to
              <strong>{drip_pred:.2f} t/ha</strong>
              (+{drip_delta:.2f} t/ha, +{drip_pct:.1f}%).
              Drip systems deliver water directly to roots, reducing evaporative loss and boosting nutrient uptake.
              </div>
            </div>""", unsafe_allow_html=True)

    with st.expander("📋 View all input values used for prediction"):
        st.dataframe(input_df.T.rename(columns={0:"Value"}), use_container_width=True)

    st.write("")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Predicted Yield",  f"{prediction:.3f} t/ha")
    m2.metric("Total Production", f"{total_production:,.1f} t")
    m3.metric("Fertilizer / ha",  f"{fertilizer_ha:.0f} kg")
    m4.metric("Rainfall",         f"{annual_rainfall:.0f} mm")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="ftr">
  CropSense &nbsp;·&nbsp; HistGradientBoosting &nbsp;·&nbsp; 89,000 training rows
  &nbsp;·&nbsp; 29 crops &nbsp;·&nbsp; 30 states &nbsp;·&nbsp; 4 seasons &nbsp;·&nbsp; R² {MODEL_R2:.4f}
</div>""", unsafe_allow_html=True)
