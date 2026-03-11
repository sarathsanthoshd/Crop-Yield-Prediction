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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"], .stMarkdown {
  font-family: 'Inter', sans-serif !important;
}
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"], header {
  display: none !important;
}
[data-testid="stAppViewContainer"] { background-color: #f4f7f5 !important; }
.block-container { max-width: 1100px !important; padding-top: 0 !important; padding-bottom: 3rem !important; }

.cs-nav {
  background: #1a5c37; padding: 0.85rem 1.5rem;
  margin: 0 -3rem 1.5rem -3rem;
  display: flex; align-items: center; justify-content: space-between;
  box-shadow: 0 2px 8px rgba(0,0,0,0.15);
}
.cs-nav-brand { font-size: 1.25rem; font-weight: 700; color: #ffffff; letter-spacing: -0.01em; }
.cs-nav-brand span { color: #6dd49a; }
.cs-nav-stats { display: flex; gap: 1.5rem; }
.cs-nav-stat { text-align: center; color: rgba(255,255,255,0.9); }
.cs-nav-stat-val { font-size: 0.95rem; font-weight: 600; display: block; }
.cs-nav-stat-lbl { font-size: 0.65rem; color: rgba(255,255,255,0.6); text-transform: uppercase; letter-spacing: 0.06em; }

.cs-hero {
  background: linear-gradient(135deg, #1a5c37 0%, #2d9e63 100%);
  border-radius: 16px; padding: 2.5rem; margin-bottom: 2rem; color: #ffffff;
}
.cs-hero-tag {
  display: inline-block; background: rgba(255,255,255,0.15);
  border: 1px solid rgba(255,255,255,0.25); border-radius: 100px;
  padding: 0.3rem 0.9rem; font-size: 0.72rem; font-weight: 600;
  letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 1rem;
}
.cs-hero-title { font-size: 2.2rem; font-weight: 700; line-height: 1.2; margin: 0 0 0.75rem 0; letter-spacing: -0.02em; }
.cs-hero-desc { font-size: 0.95rem; color: rgba(255,255,255,0.8); line-height: 1.7; max-width: 520px; margin-bottom: 1.5rem; }
.cs-hero-stats { display: flex; gap: 2rem; flex-wrap: wrap; }
.cs-hero-stat-val { font-size: 1.6rem; font-weight: 700; display: block; line-height: 1; }
.cs-hero-stat-lbl { font-size: 0.68rem; color: rgba(255,255,255,0.65); text-transform: uppercase; letter-spacing: 0.07em; }

.cs-section {
  display: flex; align-items: center; gap: 0.75rem; margin: 2rem 0 1rem 0;
}
.cs-section-num {
  background: #1a5c37; color: #ffffff; width: 28px; height: 28px;
  border-radius: 8px; font-size: 0.72rem; font-weight: 700;
  display: flex; align-items: center; justify-content: center; flex-shrink: 0;
}
.cs-section-title { font-size: 1rem; font-weight: 600; color: #1a1a1a; }
.cs-section-line { flex: 1; height: 1px; background: #d0ddd5; }

.cs-autofill {
  background: #edfaf3; border: 1px solid #6dd49a; border-radius: 8px;
  padding: 0.6rem 1rem; font-size: 0.82rem; color: #1a5c37;
  font-weight: 500; margin-bottom: 0.75rem;
}

.cs-result {
  background: linear-gradient(135deg, #0a2e1a 0%, #1a5c37 60%, #2d9e63 100%);
  border-radius: 16px; padding: 2rem 2.5rem; color: #ffffff; margin: 1.5rem 0 0.75rem 0;
}
.cs-result-label { font-size: 0.68rem; letter-spacing: 0.18em; text-transform: uppercase; opacity: 0.65; margin-bottom: 0.3rem; }
.cs-result-number { font-size: 4.5rem; font-weight: 700; line-height: 1; letter-spacing: -0.03em; }
.cs-result-unit { font-size: 0.9rem; opacity: 0.6; margin-bottom: 0.75rem; }
.cs-result-badge {
  display: inline-flex; align-items: center; gap: 0.4rem;
  background: rgba(255,255,255,0.15); border: 1px solid rgba(255,255,255,0.25);
  border-radius: 100px; padding: 0.35rem 1rem; font-size: 0.82rem; font-weight: 600; margin-bottom: 1.25rem;
}
.cs-result-chips { display: flex; flex-wrap: wrap; gap: 0.4rem; }
.cs-chip {
  background: rgba(255,255,255,0.12); border: 1px solid rgba(255,255,255,0.18);
  border-radius: 100px; padding: 0.22rem 0.75rem; font-size: 0.72rem; color: rgba(255,255,255,0.88);
}

.cs-metrics { display: grid; grid-template-columns: repeat(3,1fr); gap: 0.75rem; margin-bottom: 0.75rem; }
.cs-metric { background: #ffffff; border: 1px solid #d0ddd5; border-radius: 12px; padding: 1.1rem 1.2rem; }
.cs-metric-icon { font-size: 1.3rem; margin-bottom: 0.4rem; }
.cs-metric-label { font-size: 0.65rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: #7a9486; margin-bottom: 0.2rem; }
.cs-metric-val { font-size: 1.15rem; font-weight: 700; color: #1a5c37; }
.cs-metric-sub { font-size: 0.7rem; color: #7a9486; margin-top: 0.15rem; }

.cs-gauge { background: #ffffff; border: 1px solid #d0ddd5; border-radius: 12px; padding: 1.2rem 1.4rem; margin-bottom: 0.75rem; }
.cs-gauge-head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem; }
.cs-gauge-title { font-size: 0.68rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: #7a9486; }
.cs-gauge-val { font-size: 0.88rem; font-weight: 700; color: #1a5c37; }
.cs-gauge-track { background: #e8f0eb; border-radius: 99px; height: 10px; overflow: hidden; }
.cs-gauge-fill { height: 100%; border-radius: 99px; background: linear-gradient(90deg, #6dd49a, #2d9e63, #1a5c37); }
.cs-gauge-scale { display: flex; justify-content: space-between; margin-top: 0.4rem; font-size: 0.6rem; color: #7a9486; }

.cs-compare { display: grid; grid-template-columns: repeat(auto-fit,minmax(110px,1fr)); gap: 0.6rem; margin-top: 0.75rem; }
.cs-compare-card { background: #f4f7f5; border: 1.5px solid #d0ddd5; border-radius: 10px; padding: 0.8rem; text-align: center; }
.cs-compare-card.best { background: #edfaf3; border-color: #2d9e63; }
.cs-compare-label { font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.07em; color: #7a9486; font-weight: 600; }
.cs-compare-val { font-size: 1rem; font-weight: 700; color: #1a5c37; margin-top: 0.25rem; }

.cs-tip {
  background: #fffbeb; border: 1px solid #fcd34d; border-radius: 12px;
  padding: 1rem 1.2rem; font-size: 0.85rem; color: #78350f; line-height: 1.65; margin-bottom: 0.75rem;
}

.cs-footer {
  text-align: center; padding: 2rem 0 0.5rem; font-size: 0.7rem; color: #7a9486;
  border-top: 1px solid #d0ddd5; margin-top: 3rem; letter-spacing: 0.04em;
}

@media (max-width: 768px) {
  .cs-nav { margin: 0 -1rem 1rem -1rem; }
  .cs-nav-stats { display: none; }
  .cs-metrics { grid-template-columns: 1fr; }
  .cs-result-number { font-size: 3rem; }
  .cs-hero-title { font-size: 1.6rem; }
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
    model    = load_model()
    metadata = load_metadata()
    uv            = metadata['unique_values']
    state_climate = metadata['state_climate']
    crop_type_map = metadata['crop_type_map']
    SEASON_MONTH  = metadata.get('season_default_month',
                        {'Kharif': 6, 'Rabi': 11, 'Summer': 3, 'Whole Year': 1})
    MODEL_R2  = metadata.get('model_r2',  0.9847)
    MODEL_MAE = metadata.get('model_mae', 0.5435)
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
except FileNotFoundError as e:
    st.error(f"Model files not found. Run `python train_model.py` first.\n\n{e}")
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
    1:'January', 2:'February', 3:'March', 4:'April',
    5:'May', 6:'June', 7:'July', 8:'August',
    9:'September', 10:'October', 11:'November', 12:'December'
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

st.markdown(f"""
<div class="cs-nav">
  <div class="cs-nav-brand">🌾 Crop<span>Sense</span></div>
  <div class="cs-nav-stats">
    <div class="cs-nav-stat">
      <span class="cs-nav-stat-val">{MODEL_R2:.4f}</span>
      <span class="cs-nav-stat-lbl">R² Score</span>
    </div>
    <div class="cs-nav-stat">
      <span class="cs-nav-stat-val">{MODEL_MAE:.3f} t/ha</span>
      <span class="cs-nav-stat-lbl">Avg Error</span>
    </div>
    <div class="cs-nav-stat">
      <span class="cs-nav-stat-val">89K rows</span>
      <span class="cs-nav-stat-lbl">Trained on</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="cs-hero">
  <div class="cs-hero-tag">🌱 Precision Agriculture · ML Powered</div>
  <div class="cs-hero-title">Predict Your Crop Yield</div>
  <div class="cs-hero-desc">
    Enter your farm details — crop, location, soil type, climate, and inputs.
    Get an accurate yield estimate in tonnes per hectare.
  </div>
  <div class="cs-hero-stats">
    <div>
      <span class="cs-hero-stat-val">{MODEL_R2:.2f}</span>
      <span class="cs-hero-stat-lbl">R² Accuracy</span>
    </div>
    <div>
      <span class="cs-hero-stat-val">29</span>
      <span class="cs-hero-stat-lbl">Crop Types</span>
    </div>
    <div>
      <span class="cs-hero-stat-val">30</span>
      <span class="cs-hero-stat-lbl">Indian States</span>
    </div>
    <div>
      <span class="cs-hero-stat-val">4</span>
      <span class="cs-hero-stat-lbl">Seasons</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="cs-section">
  <div class="cs-section-num">01</div>
  <div class="cs-section-title">Crop Identity</div>
  <div class="cs-section-line"></div>
</div>""", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
with c1:
    st.selectbox(
        "Crop Variety",
        options=sorted(uv['Crop']),
        index=sorted(uv['Crop']).index(st.session_state.selected_crop),
        key="_crop_selector",
        on_change=on_crop_change,
    )
    crop = st.session_state.selected_crop

with c2:
    valid_seasons = crop_seasons.get(crop, ['Kharif'])
    if st.session_state.selected_season not in valid_seasons:
        st.session_state.selected_season = valid_seasons[0]
    season = st.selectbox(
        "Growing Season",
        options=valid_seasons,
        index=valid_seasons.index(st.session_state.selected_season),
        key="_season_selector",
    )
    st.session_state.selected_season = season

with c3:
    crop_year = st.number_input("Crop Year", min_value=1997, max_value=2035, value=2024, step=1)

with st.form("farm_details_form"):

    st.markdown("""
    <div class="cs-section">
      <div class="cs-section-num">02</div>
      <div class="cs-section-title">Location &amp; Timing</div>
      <div class="cs-section-line"></div>
    </div>""", unsafe_allow_html=True)

    c4, c5, c6 = st.columns(3)
    with c4:
        state = st.selectbox("State", sorted(uv['State']))
    with c5:
        default_month = SEASON_MONTH.get(season, 6)
        harvest_month = st.selectbox(
            "Harvest / Sowing Month",
            options=list(range(1, 13)),
            format_func=lambda m: MONTH_NAMES[m],
            index=default_month - 1,
            help="Auto-filled from season. Override if needed.",
        )
    with c6:
        area = st.number_input("Area (hectares)", min_value=0.5, max_value=5_000_000.0,
                               value=1000.0, step=100.0, format="%.1f")

    st.markdown("""
    <div class="cs-section">
      <div class="cs-section-num">03</div>
      <div class="cs-section-title">Soil &amp; Irrigation</div>
      <div class="cs-section-line"></div>
    </div>""", unsafe_allow_html=True)

    c7, c8 = st.columns(2)
    with c7:
        soil_type  = st.selectbox("Soil Type", sorted(uv['Soil_Type']))
    with c8:
        irrigation = st.selectbox("Irrigation Method", sorted(uv['Irrigation_Type']))

    st.markdown("""
    <div class="cs-section">
      <div class="cs-section-num">04</div>
      <div class="cs-section-title">Climate Conditions</div>
      <div class="cs-section-line"></div>
    </div>""", unsafe_allow_html=True)

    clim = state_climate.get(state, {
        'Annual_Rainfall': 900.0, 'Humidity': 55.0,
        'Avg_Temperature': 25.0, 'Max_Temperature': 35.0, 'Min_Temperature': 15.0
    })
    st.markdown(f'<div class="cs-autofill">✦ Auto-filled with <strong>{state}</strong> averages — edit freely.</div>',
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
    <div class="cs-section">
      <div class="cs-section-num">05</div>
      <div class="cs-section-title">Farm Inputs</div>
      <div class="cs-section-line"></div>
    </div>""", unsafe_allow_html=True)

    ci1, ci2, _, _ = st.columns(4)
    with ci1:
        fertilizer_ha = st.number_input("Fertilizer (kg/ha)", min_value=0.0, max_value=500.0,
                                        value=145.0, step=5.0)
    with ci2:
        pesticide_ha = st.number_input("Pesticide (kg/ha)", min_value=0.0, max_value=2.0,
                                       value=0.27, step=0.01, format="%.3f")

    st.write("")
    submitted = st.form_submit_button("🌱  Predict Crop Yield", use_container_width=True)

if submitted:
    region    = STATE_REGION.get(state, 'Central')
    crop_type = crop_type_map.get(crop, 'Cereal')
    month_sin = float(np.sin(2 * np.pi * harvest_month / 12))
    month_cos = float(np.cos(2 * np.pi * harvest_month / 12))

    input_df = pd.DataFrame([{
        "Crop": crop, "Crop_Type": crop_type, "Season_clean": season,
        "State": state, "Region": region, "Soil_Type": soil_type,
        "Irrigation_Type": irrigation, "Crop_Year": crop_year, "Area": area,
        "Annual_Rainfall": annual_rainfall, "Humidity": humidity,
        "Avg_Temperature": avg_temp, "Max_Temperature": max_temp,
        "Min_Temperature": min_temp,
        "Fertilizer_per_Hectare": fertilizer_ha,
        "Pesticide_per_Hectare": pesticide_ha,
        "Month_sin": month_sin, "Month_cos": month_cos,
    }])

    prediction       = max(0.01, float(np.expm1(model.predict(input_df)[0])))
    total_production = prediction * area

    if   prediction < 1.0:  tier, tier_icon = "Below Average", "⚠️"
    elif prediction < 2.5:  tier, tier_icon = "Average",       "📊"
    elif prediction < 5.0:  tier, tier_icon = "Good",          "✅"
    elif prediction < 12.0: tier, tier_icon = "Excellent",     "🌟"
    else:                   tier, tier_icon = "Outstanding",   "🏆"

    gauge_pct = min(100, int(np.log1p(prediction) / np.log1p(20) * 100))

    st.markdown(f"""
    <div class="cs-result">
      <div class="cs-result-label">Estimated Yield — {crop} · {state}</div>
      <div class="cs-result-number">{prediction:.2f}</div>
      <div class="cs-result-unit">tonnes per hectare</div>
      <div class="cs-result-badge">{tier_icon} {tier} Yield</div>
      <div class="cs-result-chips">
        <span class="cs-chip">🌿 {crop_type}</span>
        <span class="cs-chip">📍 {region} India</span>
        <span class="cs-chip">🗓️ {MONTH_NAMES[harvest_month]} {crop_year}</span>
        <span class="cs-chip">🌱 {soil_type}</span>
        <span class="cs-chip">💧 {irrigation}</span>
        <span class="cs-chip">☀️ {season}</span>
      </div>
    </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="cs-metrics">
      <div class="cs-metric">
        <div class="cs-metric-icon">🏭</div>
        <div class="cs-metric-label">Total Production</div>
        <div class="cs-metric-val">{total_production:,.1f} t</div>
        <div class="cs-metric-sub">across {area:,.0f} ha</div>
      </div>
      <div class="cs-metric">
        <div class="cs-metric-icon">💊</div>
        <div class="cs-metric-label">Fertilizer Applied</div>
        <div class="cs-metric-val">{fertilizer_ha:.0f} kg/ha</div>
        <div class="cs-metric-sub">Total: {fertilizer_ha * area:,.0f} kg</div>
      </div>
      <div class="cs-metric">
        <div class="cs-metric-icon">🌡️</div>
        <div class="cs-metric-label">Temperature Range</div>
        <div class="cs-metric-val">{min_temp:.0f}° – {max_temp:.0f}°C</div>
        <div class="cs-metric-sub">Avg {avg_temp:.1f}°C · {humidity:.0f}% RH</div>
      </div>
    </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="cs-gauge">
      <div class="cs-gauge-head">
        <span class="cs-gauge-title">Yield Performance Gauge</span>
        <span class="cs-gauge-val">{prediction:.2f} t/ha — {tier}</span>
      </div>
      <div class="cs-gauge-track">
        <div class="cs-gauge-fill" style="width:{gauge_pct}%"></div>
      </div>
      <div class="cs-gauge-scale">
        <span>0</span><span>Low &lt;1</span>
        <span>Avg 2-3</span><span>Good 5-8</span><span>20+ t/ha</span>
      </div>
    </div>""", unsafe_allow_html=True)

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
            f'<div class="cs-compare-card {"best" if s == best_season else ""}">'
            f'<div class="cs-compare-label">{"🏆 " if s == best_season else ""}{s}</div>'
            f'<div class="cs-compare-val">{v:.2f} t/ha</div></div>'
            for s, v in season_results.items()
        ])
        st.markdown(f"""
        <div class="cs-gauge">
          <div class="cs-gauge-head">
            <span class="cs-gauge-title">Season Comparison — {crop}</span>
            <span class="cs-gauge-val">Best: {best_season}</span>
          </div>
          <div class="cs-compare">{cards}</div>
        </div>""", unsafe_allow_html=True)

    if irrigation != "Drip":
        alt_drip = input_df.copy()
        alt_drip["Irrigation_Type"] = "Drip"
        drip_pred  = max(0.01, float(np.expm1(model.predict(alt_drip)[0])))
        drip_delta = drip_pred - prediction
        drip_pct   = drip_delta / prediction * 100
        if drip_delta > 0:
            st.markdown(f"""
            <div class="cs-tip">
              💡 <strong>Drip Irrigation Tip:</strong>
              Switching to drip could push yield to <strong>{drip_pred:.2f} t/ha</strong>
              (+{drip_delta:.2f} t/ha, +{drip_pct:.1f}%).
              Drip delivers water directly to roots, reducing waste and boosting uptake.
            </div>""", unsafe_allow_html=True)

    with st.expander("📋 View all input values"):
        st.dataframe(input_df.T.rename(columns={0: "Value"}), use_container_width=True)

    st.write("")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Predicted Yield",  f"{prediction:.3f} t/ha")
    m2.metric("Total Production", f"{total_production:,.1f} t")
    m3.metric("Fertilizer / ha",  f"{fertilizer_ha:.0f} kg")
    m4.metric("Rainfall",         f"{annual_rainfall:.0f} mm")

st.markdown(f"""
<div class="cs-footer">
  CropSense · HistGradientBoosting · 89K rows · 29 crops · 30 states · 4 seasons · R² {MODEL_R2:.4f}
</div>""", unsafe_allow_html=True)
