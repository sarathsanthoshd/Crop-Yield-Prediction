"""
CropSense — Crop Yield Predictor  |  app.py
Run: streamlit run app.py
Needs: crop_yield_model_v2.joblib  +  unique_values_v2.joblib
"""

import streamlit as st
import joblib
import pandas as pd
import numpy as np

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CropSense — Yield Intelligence",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Syne:wght@700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Design tokens ── */
:root {
  --green-900: #0a2e1a;
  --green-800: #113d24;
  --green-700: #1a5c37;
  --green-600: #22784a;
  --green-500: #2d9e63;
  --green-400: #3dbf77;
  --green-300: #6dd49a;
  --green-100: #d6f5e3;
  --green-50:  #edfaf3;
  --amber-500: #d97706;
  --amber-400: #f59e0b;
  --amber-100: #fef3c7;
  --bg:        #f0f4f1;
  --surface:   #ffffff;
  --surface-2: #f7faf8;
  --border:    #d4e0d8;
  --border-2:  #e8f0eb;
  --text-1:    #0d1f15;
  --text-2:    #3d5247;
  --text-3:    #7a9486;
  --shadow-sm: 0 1px 3px rgba(10,46,26,.06), 0 1px 2px rgba(10,46,26,.04);
  --shadow-md: 0 4px 16px rgba(10,46,26,.08), 0 2px 6px rgba(10,46,26,.05);
  --shadow-lg: 0 12px 40px rgba(10,46,26,.12), 0 4px 12px rgba(10,46,26,.07);
  --shadow-xl: 0 24px 64px rgba(10,46,26,.16), 0 8px 24px rgba(10,46,26,.10);
  --r-sm: 8px; --r-md: 14px; --r-lg: 20px; --r-xl: 28px;
}

/* ── Reset & base ── */
*, *::before, *::after { box-sizing: border-box; }
html, body, [data-testid="stAppViewContainer"] {
  background: var(--bg) !important;
  font-family: 'Plus Jakarta Sans', sans-serif;
  color: var(--text-1);
  -webkit-font-smoothing: antialiased;
}
#MainMenu, footer, [data-testid="stToolbar"],
[data-testid="stDecoration"], header { display: none !important; }
.block-container { max-width: 1180px !important; padding: 0 1.5rem 4rem !important; }

/* subtle dot-grid background */
[data-testid="stAppViewContainer"]::before {
  content: '';
  position: fixed; inset: 0; pointer-events: none; z-index: 0;
  background-image:
    radial-gradient(circle, rgba(45,158,99,.07) 1px, transparent 1px);
  background-size: 28px 28px;
}
[data-testid="stAppViewContainer"]::after {
  content: '';
  position: fixed; inset: 0; pointer-events: none; z-index: 0;
  background:
    radial-gradient(ellipse 80% 50% at 0% 0%, rgba(45,158,99,.10) 0%, transparent 55%),
    radial-gradient(ellipse 60% 40% at 100% 100%, rgba(217,119,6,.07) 0%, transparent 50%);
}

/* ── Navbar ── */
.nav-bar {
  background: linear-gradient(90deg, var(--green-900) 0%, var(--green-800) 100%);
  padding: 0 2rem;
  margin: 0 -1.5rem 0;
  height: 60px;
  display: flex; align-items: center; justify-content: space-between;
  position: sticky; top: 0; z-index: 200;
  box-shadow: 0 2px 20px rgba(10,46,26,.25);
}
.nav-brand {
  display: flex; align-items: center; gap: .7rem;
  font-family: 'Syne', sans-serif;
  font-size: 1.3rem; font-weight: 800;
  color: #fff; letter-spacing: -.01em;
}
.nav-brand-icon {
  width: 32px; height: 32px; border-radius: 8px;
  background: linear-gradient(135deg, var(--green-500), var(--green-400));
  display: flex; align-items: center; justify-content: center;
  font-size: 1rem;
  box-shadow: 0 2px 8px rgba(45,158,99,.4);
}
.nav-pills { display: flex; gap: .5rem; align-items: center; }
.nav-pill {
  background: rgba(255,255,255,.08);
  border: 1px solid rgba(255,255,255,.12);
  border-radius: 100px;
  padding: .3rem .9rem;
  display: flex; flex-direction: column; align-items: center;
}
.nav-pill-val {
  font-family: 'JetBrains Mono', monospace;
  font-size: .82rem; font-weight: 500; color: #fff; line-height: 1.2;
}
.nav-pill-lbl {
  font-size: .58rem; text-transform: uppercase;
  letter-spacing: .08em; color: rgba(255,255,255,.55); line-height: 1;
}

/* ── Hero ── */
.hero {
  padding: 3rem 0 2rem;
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 2rem;
  align-items: center;
}
.hero-left { display: flex; flex-direction: column; gap: .8rem; }
.hero-eyebrow {
  display: inline-flex; align-items: center; gap: .5rem;
  background: var(--green-50); border: 1.5px solid var(--green-300);
  border-radius: 100px; padding: .35rem 1rem;
  font-size: .7rem; font-weight: 700;
  letter-spacing: .12em; text-transform: uppercase;
  color: var(--green-700);
  font-family: 'JetBrains Mono', monospace;
  width: fit-content;
}
.hero-title {
  font-family: 'Syne', sans-serif;
  font-size: clamp(2.4rem, 4.5vw, 3.6rem);
  font-weight: 800; line-height: 1.05;
  color: var(--green-900); margin: 0;
  letter-spacing: -.03em;
}
.hero-title span { color: var(--green-600); }
.hero-desc {
  font-size: .96rem; color: var(--text-2);
  line-height: 1.75; font-weight: 400;
  max-width: 440px;
}
.hero-stats {
  display: flex; gap: 1.5rem; margin-top: .5rem;
  flex-wrap: wrap;
}
.hero-stat {
  display: flex; flex-direction: column; gap: .1rem;
}
.hero-stat-val {
  font-family: 'Syne', sans-serif;
  font-size: 1.5rem; font-weight: 800;
  color: var(--green-700); line-height: 1;
}
.hero-stat-lbl {
  font-size: .68rem; color: var(--text-3);
  text-transform: uppercase; letter-spacing: .08em; font-weight: 500;
}
.hero-right {
  background: var(--surface);
  border-radius: var(--r-xl);
  border: 1.5px solid var(--border);
  padding: 1.8rem;
  box-shadow: var(--shadow-lg);
  display: flex; flex-direction: column; gap: 1rem;
}
.hero-card-title {
  font-family: 'Syne', sans-serif;
  font-size: .78rem; font-weight: 700;
  text-transform: uppercase; letter-spacing: .1em;
  color: var(--text-3);
}
.crop-icon-grid {
  display: grid; grid-template-columns: repeat(4, 1fr); gap: .5rem;
}
.crop-icon-item {
  background: var(--surface-2);
  border: 1.5px solid var(--border-2);
  border-radius: var(--r-md);
  padding: .7rem .4rem;
  text-align: center; cursor: default;
  transition: all .2s;
}
.crop-icon-item:hover {
  border-color: var(--green-400);
  background: var(--green-50);
  transform: translateY(-2px);
  box-shadow: var(--shadow-sm);
}
.crop-icon-emoji { font-size: 1.4rem; display: block; }
.crop-icon-name {
  font-size: .6rem; font-weight: 600;
  color: var(--text-3); margin-top: .3rem;
  text-transform: uppercase; letter-spacing: .04em;
}

/* ── Section header ── */
.section-header {
  display: flex; align-items: center; gap: 1rem;
  margin: 2.2rem 0 1.2rem;
}
.section-badge {
  background: var(--green-700);
  color: #fff;
  width: 26px; height: 26px; border-radius: 7px;
  font-family: 'JetBrains Mono', monospace;
  font-size: .7rem; font-weight: 500;
  display: flex; align-items: center; justify-content: center;
  flex-shrink: 0;
}
.section-title {
  font-family: 'Plus Jakarta Sans', sans-serif;
  font-size: 1rem; font-weight: 700;
  text-transform: none; letter-spacing: 0;
  color: var(--text-1);
}
.section-rule {
  flex: 1; height: 1px;
  background: linear-gradient(90deg, var(--border), transparent);
}

/* ── Form card wrapper ── */
.form-card {
  background: var(--surface);
  border: 1.5px solid var(--border);
  border-radius: var(--r-xl);
  padding: 1.8rem 2rem;
  box-shadow: var(--shadow-sm);
  margin-bottom: 1rem;
}

/* ── Widget overrides ── */
[data-testid="stSelectbox"] label,
[data-testid="stNumberInput"] label {
  font-size: .88rem !important;
  font-weight: 600 !important;
  color: var(--text-1) !important;
  text-transform: none !important;
  letter-spacing: 0 !important;
  font-family: 'Plus Jakarta Sans', sans-serif !important;
}
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
  border-radius: var(--r-sm) !important;
  border-color: var(--border) !important;
  background: var(--surface-2) !important;
  font-family: 'Plus Jakarta Sans', sans-serif !important;
  font-size: .88rem !important;
  color: var(--text-1) !important;
  transition: border-color .15s, box-shadow .15s, background .15s !important;
}
div[data-baseweb="select"] > div:hover,
div[data-baseweb="input"] > div:hover {
  border-color: var(--green-400) !important;
  background: var(--surface) !important;
}
div[data-baseweb="select"] > div:focus-within,
div[data-baseweb="input"] > div:focus-within {
  border-color: var(--green-500) !important;
  box-shadow: 0 0 0 3px rgba(45,158,99,.15) !important;
  background: var(--surface) !important;
}
input[type="number"] {
  font-family: 'JetBrains Mono', monospace !important;
  font-weight: 500 !important;
}

/* ── Submit button ── */
[data-testid="stFormSubmitButton"] > button,
[data-testid="stButton"] > button {
  background: linear-gradient(135deg, var(--green-700) 0%, var(--green-500) 100%) !important;
  color: #fff !important; border: none !important;
  border-radius: var(--r-lg) !important;
  padding: .9rem 2.5rem !important;
  font-family: 'Syne', sans-serif !important;
  font-size: .95rem !important; font-weight: 700 !important;
  letter-spacing: .04em !important; width: 100% !important;
  box-shadow: 0 4px 20px rgba(34,120,74,.35), 0 1px 4px rgba(34,120,74,.2) !important;
  transition: all .2s ease !important; cursor: pointer !important;
}
[data-testid="stFormSubmitButton"] > button:hover,
[data-testid="stButton"] > button:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 8px 28px rgba(34,120,74,.45) !important;
  background: linear-gradient(135deg, var(--green-800) 0%, var(--green-600) 100%) !important;
}

/* ── Autofill notice ── */
.autofill-bar {
  background: linear-gradient(90deg, var(--green-50), var(--surface));
  border: 1.5px solid var(--green-300);
  border-radius: var(--r-md);
  padding: .6rem 1rem;
  font-size: .78rem; color: var(--green-700);
  display: flex; align-items: center; gap: .5rem;
  margin-bottom: .8rem; font-weight: 500;
}




/* ── Result section ── */
.result-wrap { margin: 2rem 0; animation: riseIn .45s cubic-bezier(.22,1,.36,1) forwards; }
@keyframes riseIn { from{opacity:0;transform:translateY(24px)} to{opacity:1;transform:translateY(0)} }

.result-banner {
  border-radius: var(--r-xl);
  background: linear-gradient(135deg, var(--green-900) 0%, var(--green-700) 60%, var(--green-500) 100%);
  padding: 2.5rem 2.8rem;
  color: #fff; position: relative; overflow: hidden;
  box-shadow: var(--shadow-xl);
}
.result-banner::before {
  content: '';
  position: absolute; top: -80px; right: -80px;
  width: 320px; height: 320px; border-radius: 50%;
  background: radial-gradient(circle, rgba(255,255,255,.06) 0%, transparent 65%);
  pointer-events: none;
}
.result-banner::after {
  content: '⬡';
  position: absolute; bottom: -30px; left: -10px;
  font-size: 16rem; opacity: .04; color: #fff;
  line-height: 1; pointer-events: none;
}
.result-top { display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 1rem; }
.result-crop-label {
  font-family: 'JetBrains Mono', monospace;
  font-size: .65rem; letter-spacing: .2em;
  text-transform: uppercase; opacity: .65;
  margin-bottom: .4rem;
}
.result-big {
  font-family: 'Syne', sans-serif;
  font-size: clamp(3.5rem, 7vw, 6rem);
  font-weight: 800; line-height: 1; letter-spacing: -.03em;
}
.result-unit-label { font-size: .95rem; opacity: .6; font-weight: 400; margin-top: .2rem; }
.result-badge {
  display: inline-flex; align-items: center; gap: .5rem;
  background: rgba(255,255,255,.14);
  border: 1.5px solid rgba(255,255,255,.22);
  border-radius: 100px; padding: .45rem 1.2rem;
  font-size: .8rem; font-weight: 700; letter-spacing: .04em;
  backdrop-filter: blur(4px);
}
.result-chips { display: flex; flex-wrap: wrap; gap: .5rem; margin-top: 1.5rem; }
.result-chip {
  background: rgba(255,255,255,.10);
  border: 1px solid rgba(255,255,255,.16);
  border-radius: 100px; padding: .28rem .85rem;
  font-size: .72rem; font-weight: 500; color: rgba(255,255,255,.88);
}

/* ── Metric cards ── */
.metric-row {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: .8rem; margin-top: .8rem;
}
.metric-card {
  background: var(--surface);
  border: 1.5px solid var(--border);
  border-radius: var(--r-lg);
  padding: 1.2rem 1.4rem;
  box-shadow: var(--shadow-sm);
  display: flex; flex-direction: column; gap: .3rem;
  transition: box-shadow .2s, transform .2s;
}
.metric-card:hover {
  box-shadow: var(--shadow-md);
  transform: translateY(-2px);
}
.metric-icon { font-size: 1.4rem; }
.metric-label {
  font-size: .65rem; font-weight: 700;
  text-transform: uppercase; letter-spacing: .1em;
  color: var(--text-3);
  font-family: 'JetBrains Mono', monospace;
}
.metric-val {
  font-family: 'Syne', sans-serif;
  font-size: 1.2rem; font-weight: 800;
  color: var(--green-700); line-height: 1;
}
.metric-sub { font-size: .7rem; color: var(--text-3); }

/* ── Gauge ── */
.gauge-card {
  background: var(--surface);
  border: 1.5px solid var(--border);
  border-radius: var(--r-lg);
  padding: 1.4rem 1.6rem;
  margin-top: .8rem;
  box-shadow: var(--shadow-sm);
}
.gauge-head {
  display: flex; justify-content: space-between; align-items: center;
  margin-bottom: .9rem;
}
.gauge-head-title {
  font-size: .68rem; font-weight: 700;
  text-transform: uppercase; letter-spacing: .1em;
  color: var(--text-3);
  font-family: 'JetBrains Mono', monospace;
}
.gauge-head-val {
  font-family: 'Syne', sans-serif;
  font-size: .9rem; font-weight: 800;
  color: var(--green-700);
}
.gauge-track {
  background: var(--border-2);
  border-radius: 99px; height: 10px;
  overflow: hidden; position: relative;
}
.gauge-fill {
  height: 100%; border-radius: 99px;
  background: linear-gradient(90deg, var(--green-400), var(--green-700), var(--green-900));
  transition: width .9s cubic-bezier(.34,1.56,.64,1);
  position: relative;
}
.gauge-fill::after {
  content: '';
  position: absolute; right: 0; top: 50%;
  transform: translateY(-50%);
  width: 14px; height: 14px; border-radius: 50%;
  background: #fff;
  box-shadow: 0 0 0 3px var(--green-600), 0 2px 6px rgba(10,46,26,.3);
}
.gauge-scale {
  display: flex; justify-content: space-between;
  margin-top: .5rem;
  font-size: .6rem; color: var(--text-3);
  font-family: 'JetBrains Mono', monospace;
}

/* ── Season compare ── */
.compare-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(100px, 1fr));
  gap: .6rem; margin-top: .8rem;
}
.compare-card {
  background: var(--surface-2);
  border: 1.5px solid var(--border);
  border-radius: var(--r-md);
  padding: .9rem .8rem; text-align: center;
  transition: all .2s;
}
.compare-card.best {
  background: var(--green-50);
  border-color: var(--green-500);
  box-shadow: 0 0 0 3px rgba(45,158,99,.12);
}
.compare-card-label {
  font-size: .62rem; text-transform: uppercase;
  letter-spacing: .08em; color: var(--text-3);
  font-family: 'JetBrains Mono', monospace;
  font-weight: 600;
}
.compare-card-val {
  font-family: 'Syne', sans-serif;
  font-size: 1.05rem; font-weight: 800;
  color: var(--green-700); margin-top: .3rem;
}

/* ── Tip box ── */
.tip-card {
  background: linear-gradient(135deg, var(--amber-100), #fff8e8);
  border: 1.5px solid #fde68a;
  border-radius: var(--r-lg);
  padding: 1.1rem 1.4rem;
  font-size: .86rem; color: #78350f;
  line-height: 1.7; margin-top: .8rem;
  display: flex; gap: .8rem; align-items: flex-start;
}
.tip-icon { font-size: 1.2rem; flex-shrink: 0; margin-top: .1rem; }
.tip-card strong { color: #92400e; }

/* ── Expander ── */
[data-testid="stExpander"] {
  border: 1.5px solid var(--border) !important;
  border-radius: var(--r-lg) !important;
  background: var(--surface) !important;
  margin-top: .8rem !important;
  box-shadow: var(--shadow-sm) !important;
}

/* ── Streamlit metric cards override ── */
[data-testid="stMetric"] {
  background: var(--surface) !important;
  border: 1.5px solid var(--border) !important;
  border-radius: var(--r-lg) !important;
  padding: 1rem 1.2rem !important;
  box-shadow: var(--shadow-sm) !important;
}
[data-testid="stMetricLabel"] {
  font-size: .68rem !important; font-weight: 700 !important;
  text-transform: uppercase !important; letter-spacing: .08em !important;
  color: var(--text-3) !important;
}
[data-testid="stMetricValue"] {
  font-family: 'Syne', sans-serif !important;
  font-size: 1.3rem !important; font-weight: 800 !important;
  color: var(--green-700) !important;
}

/* ── Footer ── */
.app-footer {
  text-align: center; padding: 2rem 0 1rem;
  font-size: .7rem; color: var(--text-3);
  border-top: 1px solid var(--border-2);
  margin-top: 3rem;
  font-family: 'JetBrains Mono', monospace;
  letter-spacing: .06em;
}

/* ── Responsive ── */
@media (max-width: 768px) {
  .block-container { padding: 0 1rem 3rem !important; }
  .hero { grid-template-columns: 1fr; }
  .hero-right { display: none; }
  .metric-row { grid-template-columns: 1fr; }
  .nav-pills { display: none; }
  .result-big { font-size: 3rem !important; }
  .insights-grid { grid-template-columns: 1fr !important; }
  .compare-grid { grid-template-columns: repeat(2, 1fr); }
}
@media (max-width: 480px) {
  .result-banner { padding: 1.5rem 1.4rem; }
  .result-top { flex-direction: column; }
}
</style>
""", unsafe_allow_html=True)

# ── Load model & metadata ──────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("crop_yield_model_v2.joblib")

@st.cache_data
def load_metadata():
    return joblib.load("unique_values_v2.joblib")

try:
    model    = load_model()
    metadata = load_metadata()
    uv             = metadata['unique_values']
    state_climate  = metadata['state_climate']
    crop_seasons = {
        'Arhar/Tur':         ['Kharif', 'Rabi', 'Whole Year'],
        'Bajra':             ['Kharif', 'Rabi', 'Summer', 'Whole Year'],
        'Castor seed':       ['Kharif', 'Rabi', 'Whole Year'],
        'Cotton(lint)':      ['Kharif', 'Rabi', 'Summer', 'Whole Year'],
        'Dry chillies':      ['Kharif', 'Rabi', 'Summer', 'Whole Year'],
        'Gram':              ['Kharif', 'Rabi', 'Whole Year'],
        'Groundnut':         ['Kharif', 'Rabi', 'Summer', 'Whole Year'],
        'Jowar':             ['Kharif', 'Rabi', 'Summer', 'Whole Year'],
        'Jute':              ['Kharif', 'Rabi', 'Summer'],
        'Linseed':           ['Kharif', 'Rabi', 'Whole Year'],
        'Maize':             ['Kharif', 'Rabi', 'Summer', 'Whole Year'],
        'Mesta':             ['Kharif', 'Rabi', 'Whole Year'],
        'Moong(Green Gram)': ['Kharif', 'Rabi', 'Summer', 'Whole Year'],
        'Niger seed':        ['Kharif', 'Rabi', 'Whole Year'],
        'Onion':             ['Kharif', 'Rabi', 'Summer', 'Whole Year'],
        'Paddy':             ['Kharif', 'Rabi', 'Summer', 'Whole Year'],
        'Potato':            ['Kharif', 'Rabi', 'Summer', 'Whole Year'],
        'Ragi':              ['Kharif', 'Rabi', 'Summer', 'Whole Year'],
        'Rapeseed &Mustard': ['Kharif', 'Rabi', 'Whole Year'],
        'Rice':              ['Kharif', 'Rabi', 'Summer', 'Whole Year'],
        'Safflower':         ['Kharif', 'Rabi', 'Whole Year'],
        'Sesamum':           ['Kharif', 'Rabi', 'Summer', 'Whole Year'],
        'Small millets':     ['Kharif', 'Rabi', 'Summer', 'Whole Year'],
        'Soyabean':          ['Kharif', 'Rabi', 'Whole Year'],
        'Sugarcane':         ['Kharif', 'Rabi', 'Summer', 'Whole Year'],
        'Sunflower':         ['Kharif', 'Rabi', 'Summer', 'Whole Year'],
        'Tobacco':           ['Kharif', 'Rabi', 'Summer', 'Whole Year'],
        'Urad':              ['Kharif', 'Rabi', 'Summer', 'Whole Year'],
        'Wheat':             ['Kharif', 'Rabi', 'Summer', 'Whole Year'],
    }
    crop_type_map  = metadata['crop_type_map']
    SEASON_MONTH   = metadata.get('season_default_month',
                         {'Kharif':6,'Rabi':11,'Summer':3,'Whole Year':1})
    MODEL_R2  = metadata.get('model_r2',  0.9847)
    MODEL_MAE = metadata.get('model_mae', 0.5435)
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
MONTH_NAMES = {1:'January',2:'February',3:'March',4:'April',5:'May',6:'June',
               7:'July',8:'August',9:'September',10:'October',11:'November',12:'December'}

# ── Session state defaults ─────────────────────────────────────────────────────
# KEY FIX: store crop + season in session_state so season updates
# instantly when crop changes — WITHOUT needing a form submit.
if 'selected_crop' not in st.session_state:
    st.session_state.selected_crop = sorted(uv['Crop'])[0]
if 'selected_season' not in st.session_state:
    st.session_state.selected_season = crop_seasons[st.session_state.selected_crop][0]

def on_crop_change():
    """Reset season to first valid option when crop changes."""
    new_crop    = st.session_state._crop_selector
    valid       = crop_seasons.get(new_crop, uv['Season'])
    st.session_state.selected_crop   = new_crop
    st.session_state.selected_season = valid[0]   # ← reset to first valid season

# ── Nav bar ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="nav-bar">
  <div class="nav-brand">
    <div class="nav-brand-icon">🌾</div>
    CropSense
  </div>
  <div class="nav-pills">
    <div class="nav-pill">
      <span class="nav-pill-val">{MODEL_R2:.4f}</span>
      <span class="nav-pill-lbl">R² Score</span>
    </div>
    <div class="nav-pill">
      <span class="nav-pill-val">{MODEL_MAE:.3f} t/ha</span>
      <span class="nav-pill-lbl">Avg Error</span>
    </div>
    <div class="nav-pill">
      <span class="nav-pill-val">89K rows</span>
      <span class="nav-pill-lbl">Trained on</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero">
  <div class="hero-left">
    <div class="hero-eyebrow">🌱 Precision Agriculture · ML Powered</div>
    <h1 class="hero-title">Predict Your<br><span>Crop Yield</span></h1>
    <p class="hero-desc">
      Enter your farm's details — crop, location, soil type, climate, and inputs.
      Get an accurate yield estimate in <strong>tonnes per hectare</strong>.
    </p>
    <div class="hero-stats">
      <div class="hero-stat">
        <div class="hero-stat-val">{MODEL_R2:.2f}</div>
        <div class="hero-stat-lbl">R² Accuracy</div>
      </div>
      <div class="hero-stat">
        <div class="hero-stat-val">29</div>
        <div class="hero-stat-lbl">Crop Types</div>
      </div>
      <div class="hero-stat">
        <div class="hero-stat-val">30</div>
        <div class="hero-stat-lbl">Indian States</div>
      </div>
      <div class="hero-stat">
        <div class="hero-stat-val">89K</div>
        <div class="hero-stat-lbl">Training Rows</div>
      </div>
    </div>
  </div>
  <div class="hero-right">
    <div class="hero-card-title">Supported Crops</div>
    <div class="crop-icon-grid">
      <div class="crop-icon-item"><span class="crop-icon-emoji">🌾</span><div class="crop-icon-name">Rice</div></div>
      <div class="crop-icon-item"><span class="crop-icon-emoji">🌿</span><div class="crop-icon-name">Wheat</div></div>
      <div class="crop-icon-item"><span class="crop-icon-emoji">🌽</span><div class="crop-icon-name">Maize</div></div>
      <div class="crop-icon-item"><span class="crop-icon-emoji">🥔</span><div class="crop-icon-name">Potato</div></div>
      <div class="crop-icon-item"><span class="crop-icon-emoji">🧅</span><div class="crop-icon-name">Onion</div></div>
      <div class="crop-icon-item"><span class="crop-icon-emoji">🌱</span><div class="crop-icon-name">Soyabean</div></div>
      <div class="crop-icon-item"><span class="crop-icon-emoji">🌻</span><div class="crop-icon-name">Sunflower</div></div>
      <div class="crop-icon-item"><span class="crop-icon-emoji">🎋</span><div class="crop-icon-name">Sugarcane</div></div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 01 — Crop Identity   (OUTSIDE form so season updates live)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="section-header">
  <div class="section-badge">01</div>
  <div class="section-title">Crop Identity</div>
  <div class="section-rule"></div>
</div>""", unsafe_allow_html=True)

col_crop, col_season, col_year = st.columns(3)

with col_crop:
    # on_change callback fires BEFORE the widget value is committed to session_state
    # so we read from the key '_crop_selector'
    st.selectbox(
        "Crop Variety",
        options=sorted(uv['Crop']),
        index=sorted(uv['Crop']).index(st.session_state.selected_crop),
        key="_crop_selector",
        on_change=on_crop_change,
    )
    crop = st.session_state.selected_crop

with col_season:
    valid_seasons = crop_seasons.get(crop, uv['Season'])
    if st.session_state.selected_season not in valid_seasons:
        st.session_state.selected_season = valid_seasons[0]
    season = st.selectbox(
        "Growing Season",
        options=valid_seasons,
        index=valid_seasons.index(st.session_state.selected_season),
        key="_season_selector",
    )
    st.session_state.selected_season = season

with col_year:
    crop_year = st.number_input("Crop Year", min_value=1997, max_value=2035,
                                value=2024, step=1)

# ══════════════════════════════════════════════════════════════════════════════
# STEPS 02–05  — inside the form (only submitted once)
# ══════════════════════════════════════════════════════════════════════════════
with st.form("farm_details_form"):

    # ── STEP 02: Location & Timing ────────────────────────────────────────────
    st.markdown("""
    <div class="section-header">
      <div class="section-badge">02</div>
      <div class="section-title">Location &amp; Timing</div>
      <div class="section-rule"></div>
    </div>""", unsafe_allow_html=True)

    c4, c5, c6 = st.columns(3)
    with c4:
        state = st.selectbox("State", sorted(uv['State']))
    with c5:
        default_month = SEASON_MONTH.get(season, 6)
        harvest_month = st.selectbox(
            "Harvest / Sowing Month",
            options=list(range(1, 13)),
            format_func=lambda m: f"{m:02d}  {MONTH_NAMES[m]}",
            index=default_month - 1,
            help="Auto-filled from season. Override if needed.",
        )
    with c6:
        area = st.number_input("Area (hectares)", min_value=0.5,
                               max_value=5_000_000.0, value=1000.0,
                               step=100.0, format="%.1f")

    # ── STEP 03: Soil & Irrigation ────────────────────────────────────────────
    st.markdown("""
    <div class="section-header">
      <div class="section-badge">03</div>
      <div class="section-title">Soil &amp; Irrigation</div>
      <div class="section-rule"></div>
    </div>""", unsafe_allow_html=True)

    c7, c8 = st.columns(2)
    with c7:
        soil_type  = st.selectbox("Soil Type", sorted(uv['Soil_Type']))
    with c8:
        irrigation = st.selectbox("Irrigation Method", sorted(uv['Irrigation_Type']))

    # ── STEP 04: Climate ──────────────────────────────────────────────────────
    st.markdown("""
    <div class="section-header">
      <div class="section-badge">04</div>
      <div class="section-title">Climate Conditions</div>
      <div class="section-rule"></div>
    </div>""", unsafe_allow_html=True)

    clim = state_climate.get(state, {
        'Annual_Rainfall':900.0,'Humidity':55.0,
        'Avg_Temperature':25.0,'Max_Temperature':35.0,'Min_Temperature':15.0
    })
    st.markdown(f"""<div class="autofill-bar">
      ✦ Auto-filled with <strong>{state}</strong> averages — edit freely.
    </div>""", unsafe_allow_html=True)

    c9, c10, c11, c12, c13 = st.columns(5)
    with c9:
        annual_rainfall = st.number_input("Rainfall (mm)",  min_value=50.0, max_value=6000.0, value=float(clim['Annual_Rainfall']), step=10.0)
    with c10:
        humidity        = st.number_input("Humidity (%)",   min_value=10.0, max_value=100.0,  value=float(clim['Humidity']),        step=1.0)
    with c11:
        avg_temp        = st.number_input("Avg Temp (°C)",  min_value=-20.0,max_value=50.0,   value=float(clim['Avg_Temperature']), step=0.5)
    with c12:
        max_temp        = st.number_input("Max Temp (°C)",  min_value=-10.0,max_value=60.0,   value=float(clim['Max_Temperature']), step=0.5)
    with c13:
        min_temp        = st.number_input("Min Temp (°C)",  min_value=-25.0,max_value=45.0,   value=float(clim['Min_Temperature']), step=0.5)

    # ── STEP 05: Farm Inputs ──────────────────────────────────────────────────
    st.markdown("""
    <div class="section-header">
      <div class="section-badge">05</div>
      <div class="section-title">Farm Inputs</div>
      <div class="section-rule"></div>
    </div>""", unsafe_allow_html=True)

    ci1, ci2, _, _ = st.columns(4)
    with ci1:
        fertilizer_ha = st.number_input("Fertilizer (kg/ha)", min_value=0.0, max_value=500.0, value=145.0, step=5.0)
    with ci2:
        pesticide_ha  = st.number_input("Pesticide (kg/ha)",  min_value=0.0, max_value=2.0,   value=0.27,  step=0.01, format="%.3f")

    st.markdown("<br>", unsafe_allow_html=True)
    submitted = st.form_submit_button("🌱  Predict Crop Yield")

# ── Prediction ─────────────────────────────────────────────────────────────────
if submitted:
    region    = STATE_REGION.get(state, 'Central')
    crop_type = crop_type_map.get(crop, 'Cereal')
    month_sin = float(np.sin(2 * np.pi * harvest_month / 12))
    month_cos = float(np.cos(2 * np.pi * harvest_month / 12))

    input_df = pd.DataFrame([{
        "Crop":crop,"Crop_Type":crop_type,"Season_clean":season,
        "State":state,"Region":region,"Soil_Type":soil_type,"Irrigation_Type":irrigation,
        "Crop_Year":crop_year,"Area":area,
        "Annual_Rainfall":annual_rainfall,"Humidity":humidity,
        "Avg_Temperature":avg_temp,"Max_Temperature":max_temp,"Min_Temperature":min_temp,
        "Fertilizer_per_Hectare":fertilizer_ha,"Pesticide_per_Hectare":pesticide_ha,
        "Month_sin":month_sin,"Month_cos":month_cos,
    }])

    prediction       = max(0.01, float(np.expm1(model.predict(input_df)[0])))
    total_production = prediction * area

    if   prediction < 1.0:  tier, tier_icon = "Below Average","⚠️"
    elif prediction < 2.5:  tier, tier_icon = "Average",      "📊"
    elif prediction < 5.0:  tier, tier_icon = "Good",         "✅"
    elif prediction < 12.0: tier, tier_icon = "Excellent",    "🌟"
    else:                   tier, tier_icon = "Outstanding",  "🏆"

    gauge_pct = min(100, int(np.log1p(prediction) / np.log1p(20) * 100))

    # ── Result banner ────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="result-wrap">
      <div class="result-banner">
        <div class="result-top">
          <div>
            <div class="result-crop-label">Estimated Yield — {crop} · {state}</div>
            <div class="result-big">{prediction:.2f}</div>
            <div class="result-unit-label">tonnes per hectare</div>
          </div>
          <div class="result-badge">{tier_icon} {tier} Yield</div>
        </div>
        <div class="result-chips">
          <span class="result-chip">🌿 {crop_type}</span>
          <span class="result-chip">📍 {region} India</span>
          <span class="result-chip">🗓️ {MONTH_NAMES[harvest_month]} {crop_year}</span>
          <span class="result-chip">🌱 {soil_type}</span>
          <span class="result-chip">💧 {irrigation}</span>
          <span class="result-chip">☀️ {season}</span>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

    # ── Metric cards ─────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="metric-row">
      <div class="metric-card">
        <div class="metric-icon">🏭</div>
        <div class="metric-label">Total Production</div>
        <div class="metric-val">{total_production:,.1f} t</div>
        <div class="metric-sub">across {area:,.0f} ha</div>
      </div>
      <div class="metric-card">
        <div class="metric-icon">💊</div>
        <div class="metric-label">Fertilizer Applied</div>
        <div class="metric-val">{fertilizer_ha:.0f} kg/ha</div>
        <div class="metric-sub">Total: {fertilizer_ha*area:,.0f} kg</div>
      </div>
      <div class="metric-card">
        <div class="metric-icon">🌡️</div>
        <div class="metric-label">Temperature Range</div>
        <div class="metric-val">{min_temp:.0f}° – {max_temp:.0f}°C</div>
        <div class="metric-sub">Avg {avg_temp:.1f}°C · {humidity:.0f}% RH</div>
      </div>
    </div>""", unsafe_allow_html=True)

    # ── Yield gauge ───────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="gauge-card">
      <div class="gauge-head">
        <span class="gauge-head-title">Yield Performance Gauge</span>
        <span class="gauge-head-val">{prediction:.2f} t/ha — {tier}</span>
      </div>
      <div class="gauge-track">
        <div class="gauge-fill" style="width:{gauge_pct}%"></div>
      </div>
      <div class="gauge-scale">
        <span>0</span><span>Low &lt;1</span>
        <span>Avg 2–3</span><span>Good 5–8</span><span>20+ t/ha</span>
      </div>
    </div>""", unsafe_allow_html=True)

    # ── Season comparison (now meaningful since all seasons are present) ───────
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
        chips = "".join([
            f'<div class="compare-card {"best" if s==best_season else ""}">'
            f'<div class="compare-card-label">{"🏆 " if s==best_season else ""}{s}</div>'
            f'<div class="compare-card-val">{v:.2f} t/ha</div>'
            f'</div>'
            for s, v in season_results.items()
        ])
        st.markdown(f"""
        <div class="gauge-card" style="margin-top:.8rem">
          <div class="gauge-head">
            <span class="gauge-head-title">Season Comparison — {crop}</span>
            <span class="gauge-head-val">Best: {best_season}</span>
          </div>
          <div class="compare-grid">{chips}</div>
        </div>""", unsafe_allow_html=True)

    # ── Drip irrigation what-if ───────────────────────────────────────────────
    if irrigation != "Drip":
        alt_drip = input_df.copy()
        alt_drip["Irrigation_Type"] = "Drip"
        drip_pred  = max(0.01, float(np.expm1(model.predict(alt_drip)[0])))
        drip_delta = drip_pred - prediction
        drip_pct   = drip_delta / prediction * 100
        if drip_delta > 0:
            st.markdown(f"""
            <div class="tip-card">
              <div class="tip-icon">💡</div>
              <div><strong>Drip Irrigation Tip:</strong>
              Switching to drip could push your yield to <strong>{drip_pred:.2f} t/ha</strong>
              (+{drip_delta:.2f} t/ha, +{drip_pct:.1f}%).
              Drip systems deliver water directly to roots, reducing waste and boosting nutrient uptake.
              </div>
            </div>""", unsafe_allow_html=True)

    # ── Input summary ─────────────────────────────────────────────────────────
    with st.expander("📋 View all input values used for prediction"):
        st.dataframe(input_df.T.rename(columns={0:"Value"}), use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Predicted Yield",  f"{prediction:.3f} t/ha")
    m2.metric("Total Production", f"{total_production:,.1f} t")
    m3.metric("Fertilizer / ha",  f"{fertilizer_ha:.0f} kg")
    m4.metric("Rainfall",         f"{annual_rainfall:.0f} mm")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="app-footer">
  CropSense · HistGradientBoosting · 89K rows · 29 crops · 30 states · 4 seasons · R² {MODEL_R2:.4f}
</div>""", unsafe_allow_html=True)
