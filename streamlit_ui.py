"""
🏦 ChurnLens — Bank Customer Churn Prediction UI
--------------------------------------------------
Interactive Streamlit app that lets you tweak every feature column of
Churn_Modelling.xls (via synced slider + number input) and instantly see
whether the customer is predicted to churn.

Model: replicates the pipeline trained in bankcustomerchurn.ipynb
    · Geography  -> {France: 2, Germany: 1, Spain: 0}
    · Gender     -> {Male: 1, Female: 0}
    · identity columns (RowNumber, CustomerId, Surname) dropped
    · StandardScaler on the features
    · GradientBoostingClassifier(n_estimators=200, learning_rate=0.5,
                                 random_state=41, max_features=9)

    Churn   (Exited = 1)  ->  😟  NOT HAPPY
    No churn (Exited = 0) ->  😄  HAPPY

Run it:
    pip install streamlit scikit-learn pandas openpyxl
    streamlit run streamlit_ui.py
"""

import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
DATA_FILE = "Churn_Modelling.csv"    # (file is CSV-formatted despite the name)
MODEL_FILE = "churn_gbr.pkl"         # cached (model, scaler) from the notebook pipeline
RANDOM_STATE = 42                    # fixed so the app's holdout score is reproducible
TEST_SIZE = 0.25                     # same split ratio as the notebook

# Exact feature order after the notebook's preprocessing:
#   x = data.iloc[:, :-1]  with  RowNumber / CustomerId / Surname  dropped
FEATURE_NAMES = [
    "CreditScore", "Geography", "Gender", "Age", "Tenure", "Balance",
    "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary",
]
GEO_MAP = {"France": 2, "Germany": 1, "Spain": 0}
GENDER_MAP = {"Male": 1, "Female": 0}

st.set_page_config(
    page_title="ChurnLens · Bank Churn Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# Theming — green shade of white + hover animations
# ----------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
/* ---------- page background: soft green-tinted white ---------- */
.stApp {
    background: linear-gradient(135deg, #f2faf2 0%, #ffffff 45%, #e9f6ea 100%);
    color: #1b3a1e;
}
h1, h2, h3, h4 { color: #1b5e20 !important; }

/* ---------- hero banner ---------- */
.hero {
    background: linear-gradient(120deg, #1b5e20 0%, #2e7d32 55%, #43a047 100%);
    border-radius: 22px;
    padding: 1.6rem 2rem;
    color: #f1f8e9;
    box-shadow: 0 8px 24px rgba(27, 94, 32, .25);
    transition: transform .3s ease, box-shadow .3s ease;
}
.hero:hover { transform: translateY(-3px); box-shadow: 0 14px 32px rgba(27, 94, 32, .35); }
.hero h1 { color: #ffffff !important; margin-bottom: .2rem; }
.hero p  { color: #dcedc8; margin: 0; font-size: 1.02rem; }

/* ---------- cards with hover lift ---------- */
.card {
    background: rgba(255, 255, 255, .88);
    border: 1px solid #d7ecd8;
    border-radius: 18px;
    padding: 1.1rem 1.3rem;
    box-shadow: 0 2px 10px rgba(46, 125, 50, .07);
    transition: transform .25s ease, box-shadow .25s ease, border-color .25s ease;
    animation: fadeInUp .5s ease both;
}
.card:hover {
    transform: translateY(-6px) scale(1.01);
    box-shadow: 0 12px 28px rgba(46, 125, 50, .18);
    border-color: #a5d6a7;
}
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(14px); }
    to   { opacity: 1; transform: none; }
}

/* ---------- drop zone ---------- */
.drop-zone .stFileUploaderDropzone,
section[data-testid="stFileUploader"] div[data-testid="stFileUploaderDropzone"] {
    background: rgba(232, 245, 233, .75);
    border: 2px dashed #66bb6a !important;
    border-radius: 18px;
    transition: all .3s ease;
}
.drop-zone .stFileUploaderDropzone:hover,
section[data-testid="stFileUploader"] div[data-testid="stFileUploaderDropzone"]:hover {
    background: #dcedc8;
    border-color: #1b5e20 !important;
    transform: scale(1.015);
    box-shadow: 0 10px 26px rgba(46, 125, 50, .22);
}

/* ---------- green buttons with hover pop ---------- */
div.stButton > button, div.stDownloadButton > button {
    background: linear-gradient(135deg, #43a047, #1b5e20);
    color: #ffffff;
    border: none;
    border-radius: 12px;
    font-weight: 600;
    padding: .55rem 1.3rem;
    box-shadow: 0 3px 10px rgba(46, 125, 50, .25);
    transition: all .25s ease;
}
div.stButton > button:hover, div.stDownloadButton > button:hover {
    transform: translateY(-2px) scale(1.04);
    box-shadow: 0 8px 20px rgba(46, 125, 50, .38);
    background: linear-gradient(135deg, #4caf50, #2e7d32);
}
div.stButton > button:active { transform: scale(.97); }

/* ---------- sidebar ---------- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #edf7ee 0%, #f8fdf8 100%);
    border-right: 1px solid #d7ecd8;
}

/* ---------- green sliders ---------- */
.stSlider > div { padding-top: .2rem; }
div[data-baseweb="slider"] > div:first-child {
    background: linear-gradient(90deg, #a5d6a7, #2e7d32) !important;
}
div[data-baseweb="slider"] [role="slider"] {
    background-color: #2e7d32 !important;
    border: 3px solid #ffffff !important;
    box-shadow: 0 2px 8px rgba(27, 94, 32, .45);
    transition: transform .2s ease, box-shadow .2s ease;
}
div[data-baseweb="slider"] [role="slider"]:hover {
    transform: scale(1.25);
    box-shadow: 0 4px 14px rgba(27, 94, 32, .55);
}

/* ---------- number inputs & selects ---------- */
div[data-baseweb="input"], div[data-baseweb="select"] > div {
    border-radius: 10px !important;
    transition: box-shadow .2s ease, border-color .2s ease;
}
div[data-baseweb="input"]:hover, div[data-baseweb="select"] > div:hover {
    border-color: #66bb6a !important;
    box-shadow: 0 0 0 3px rgba(102, 187, 106, .18);
}

/* ---------- feature chips ---------- */
.chip {
    display: inline-block;
    padding: .32rem .8rem;
    margin: .18rem .22rem;
    border-radius: 999px;
    background: #e8f5e9;
    border: 1px solid #c8e6c9;
    font-size: .84rem;
    color: #000000;
    transition: all .2s ease;
    cursor: default;
}
.chip:hover { background: #a5d6a7; transform: scale(1.08); }
.chip-warn  { background: #fff8e1; border-color: #ffe082; color: #8d6e00; }
.chip-warn:hover { background: #ffecb3; }

/* ---------- verdict emoji ---------- */
@keyframes pulse {
    0% { transform: scale(1); }
    50% { transform: scale(1.18); }
    100% { transform: scale(1); }
}
.big-emoji {
    font-size: 5.2rem;
    line-height: 1.1;
    display: inline-block;
    animation: pulse 2.2s ease-in-out infinite;
}

/* ---------- donut gauge ---------- */
.gauge {
    width: 190px; height: 190px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    margin: .4rem auto;
    box-shadow: 0 6px 20px rgba(46, 125, 50, .15);
    transition: transform .35s ease, box-shadow .35s ease;
}
.gauge:hover { transform: scale(1.05) rotate(2deg); box-shadow: 0 10px 28px rgba(46, 125, 50, .25); }
.gauge-inner {
    width: 150px; height: 150px;
    border-radius: 50%;
    background: #ffffff;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
}
.gauge-val  { font-size: 1.55rem; font-weight: 800; }
.gauge-lbl  { font-size: .72rem; color: #6b8f6e; text-transform: uppercase; letter-spacing: .06em; }

/* ---------- toggles ---------- */
label[data-baseweb="checkbox"] { transition: transform .2s ease; }
label[data-baseweb="checkbox"]:hover { transform: translateX(4px); }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Data + notebook pipeline
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="📖 Loading customer data…")
def load_dataset(path: str) -> pd.DataFrame:
    """Read the raw dataset (it is comma-separated even though named .xls)."""
    try:
        df = pd.read_csv(path)
    except Exception:
        df = pd.read_excel(path)  # genuine .xls fallback (needs xlrd)
    df.columns = [c.strip() for c in df.columns]
    return df


def notebook_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the exact encoding from bankcustomerchurn.ipynb (cell 8).

    Geography: France->2, Germany->1, Spain->0   Gender: Male->1, Female->0
    """
    out = df.copy()
    geo = out["Geography"].map(GEO_MAP)
    gen = out["Gender"].map(GENDER_MAP)
    if geo.isna().any() or gen.isna().any():
        bad_geo = sorted(set(out.loc[geo.isna(), "Geography"].astype(str)))
        bad_gen = sorted(set(out.loc[gen.isna(), "Gender"].astype(str)))
        raise ValueError(
            f"Unrecognised categories — Geography: {bad_geo} (expected {list(GEO_MAP)}), "
            f"Gender: {bad_gen} (expected {list(GENDER_MAP)})"
        )
    out["Geography"] = geo.astype(int)
    out["Gender"] = gen.astype(int)
    return out


@st.cache_resource(show_spinner="🤖 Loading the notebook's Gradient Boosting model…")
def get_model_and_scaler(df: pd.DataFrame):
    """Reproduce the trained pipeline from bankcustomerchurn.ipynb.

    Loads churn_gbr.pkl when present; otherwise retrains the exact notebook
    model once (GradientBoostingClassifier + StandardScaler, 65/35 split) and
    caches it to disk for instant startup next time.
    """
    if os.path.exists(MODEL_FILE):
        try:
            model, scaler = joblib.load(MODEL_FILE)
            if getattr(model, "n_features_in_", len(FEATURE_NAMES)) == len(FEATURE_NAMES):
                return model, scaler, None
        except Exception:
            pass  # stale/incompatible artifact -> retrain

    # --- exact notebook pipeline -------------------------------------------
    data = notebook_preprocess(df)
    x = data[FEATURE_NAMES]
    y = data["Exited"].astype(int)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)   # fit on train, like the notebook
    x_test_scaled = scaler.transform(x_test)

    model = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.5, random_state=41, max_features=9
    )
    model.fit(x_train_scaled, y_train)

    # holdout evaluation, mirroring notebook cells 32–33
    holdout_acc = accuracy_score(y_test, model.predict(x_test_scaled))

    try:
        joblib.dump((model, scaler), MODEL_FILE)
    except Exception:
        pass  # cache is best-effort only
    return model, scaler, holdout_acc


def predict_churn_probability(model, scaler, raw_df: pd.DataFrame) -> np.ndarray:
    """Vectorised churn probability for preprocessed-or-raw feature rows."""
    feats = notebook_preprocess(raw_df)[FEATURE_NAMES]
    return model.predict_proba(scaler.transform(feats))[:, 1]


# ----------------------------------------------------------------------------
# Reusable widgets
# ----------------------------------------------------------------------------
def _slider_changed(key):
    st.session_state[f"{key}_n"] = st.session_state[f"{key}_s"]


def _number_changed(key):
    st.session_state[f"{key}_s"] = st.session_state[f"{key}_n"]


def synced_slider_number(label, key, min_v, max_v, default, step, fmt="%d", help=""):
    """A slider and a number input box kept perfectly in sync (two-way)."""
    if f"{key}_s" not in st.session_state:
        st.session_state[f"{key}_s"] = default
    if f"{key}_n" not in st.session_state:
        st.session_state[f"{key}_n"] = default

    c1, c2 = st.columns([0.62, 0.38], gap="small")
    with c1:
        st.slider(
            label, min_value=min_v, max_value=max_v, step=step,
            key=f"{key}_s", on_change=_slider_changed, args=(key,), help=help,
        )
    with c2:
        val = st.number_input(
            "value", min_value=float(min_v), max_value=float(max_v),
            step=float(step), key=f"{key}_n", on_change=_number_changed,
            args=(key,), label_visibility="collapsed", format=fmt,
        )
    return float(val)


def fit_step(v, min_v, max_v, step):
    """Clamp v to [min, max] and snap to the slider grid."""
    v = min(max(v, min_v), max_v)
    return min_v + round((v - min_v) / step) * step


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🏦 ChurnLens")
    st.caption("Tweak the customer profile and watch the churn verdict update live.")
    st.divider()
    st.markdown("#### 🧾 Identity")
    row_number = st.number_input("RowNumber", min_value=0, value=1, step=1, key="row_number")
    customer_id = st.number_input("CustomerId", min_value=0, value=15634602, step=1, key="customer_id")
    surname = st.text_input("Surname", value="Hargrave", key="surname",
                            help="Identity fields are collected but are dropped by the model.")
    st.divider()
    st.markdown("#### ⚙️ Controls")
    random_btn = st.button("🎲 Random customer", use_container_width=True)
    reset_btn = st.button("🔁 Reset to defaults", use_container_width=True)
    st.divider()
    st.markdown("#### 🧠 Model")
    st.caption("Gradient Boosting — replicated from **bankcustomerchurn.ipynb** "
               "(Geography/Gender label-encoded, StandardScaler, 65/35 split). "
               "😟 churn · 😄 stays")

# ----------------------------------------------------------------------------
# Main layout
# ----------------------------------------------------------------------------
st.markdown("""
<div class="hero">
    <h1>🏦 ChurnLens · Customer Churn Predictor</h1>
    <p>Slide the profile, watch the verdict flip live —
    <b>😟 churn (not happy)</b> vs <b>😄 stay (happy)</b></p>
</div>
""", unsafe_allow_html=True)

st.write("")

# ---- inputs -----------------------------------------------------------------
col_inputs, col_result = st.columns([0.60, 0.40], gap="large")

with col_inputs:
    st.markdown('<div class="card"><h4>👤 Customer Profile</h4>', unsafe_allow_html=True)

    # --- row 1 ---
    a, b, c = st.columns(3)
    with a:
        credit_score = synced_slider_number("💳 Credit Score", "credit", 350, 850, 650, 1)
    with b:
        age = synced_slider_number("🎂 Age", "age", 18, 95, 35, 1)
    with c:
        tenure = synced_slider_number("⏳ Tenure (yrs)", "tenure", 0, 10, 5, 1)

    # --- row 2 ---
    d, e, f = st.columns(3)
    with d:
        geography = st.selectbox("🌍 Geography", ["France", "Germany", "Spain"], key="geography")
    with e:
        gender = st.selectbox("🚻 Gender", ["Male", "Female"], key="gender")
    with f:
        num_products = synced_slider_number("📦 Products", "products", 1, 4, 2, 1,
                                            help="Number of bank products held")

    # --- row 3 ---
    g, h = st.columns(2)
    with g:
        balance = synced_slider_number("💰 Balance ($)", "balance", 0, 260000, 50000, 500, fmt="%.2f")
    with h:
        salary = synced_slider_number("💵 Estimated Salary ($)", "salary", 0, 250000, 80000, 1000, fmt="%.2f")

    # --- row 4: toggles ---
    i, j = st.columns(2)
    with i:
        has_cr_card = st.toggle("💳 Has Credit Card", value=True, key="has_cr_card")
    with j:
        is_active = st.toggle("✅ Active Member", value=True, key="is_active")

    st.markdown("</div>", unsafe_allow_html=True)

    # --- action buttons ---
    b1, b2, _ = st.columns([0.4, 0.4, 0.2])
    with b1:
        predict_btn = st.button("🔮 Predict churn", use_container_width=True)
    with b2:
        snapshot_btn = st.button("📋 Show customer snapshot", use_container_width=True)

inputs = {
    "RowNumber": int(row_number),
    "CustomerId": int(customer_id),
    "Surname": surname,
    "CreditScore": int(credit_score),
    "Geography": geography,
    "Gender": gender,
    "Age": int(age),
    "Tenure": int(tenure),
    "Balance": balance,
    "NumOfProducts": int(num_products),
    "HasCrCard": int(has_cr_card),
    "IsActiveMember": int(is_active),
    "EstimatedSalary": salary,
}

# ---- random / reset actions --------------------------------------------------
df = load_dataset(DATA_FILE)

if random_btn:
    row = df.sample(1, random_state=None).iloc[0]
    st.session_state["row_number"] = int(row["RowNumber"])
    st.session_state["customer_id"] = int(row["CustomerId"])
    st.session_state["surname"] = str(row["Surname"])
    st.session_state["credit_s"] = fit_step(int(row["CreditScore"]), 350, 850, 1)
    st.session_state["credit_n"] = st.session_state["credit_s"]
    st.session_state["age_s"] = fit_step(int(row["Age"]), 18, 95, 1)
    st.session_state["age_n"] = st.session_state["age_s"]
    st.session_state["tenure_s"] = fit_step(int(row["Tenure"]), 0, 10, 1)
    st.session_state["tenure_n"] = st.session_state["tenure_s"]
    st.session_state["geography"] = row["Geography"]
    st.session_state["gender"] = row["Gender"]
    st.session_state["products_s"] = fit_step(int(row["NumOfProducts"]), 1, 4, 1)
    st.session_state["products_n"] = st.session_state["products_s"]
    st.session_state["balance_s"] = fit_step(float(row["Balance"]), 0, 260000, 500)
    st.session_state["balance_n"] = st.session_state["balance_s"]
    st.session_state["salary_s"] = fit_step(float(row["EstimatedSalary"]), 0, 250000, 1000)
    st.session_state["salary_n"] = st.session_state["salary_s"]
    st.session_state["has_cr_card"] = bool(row["HasCrCard"])
    st.session_state["is_active"] = bool(row["IsActiveMember"])
    st.rerun()

if reset_btn:
    for k in list(st.session_state.keys()):
        if k.endswith("_s") or k.endswith("_n") or k in (
            "geography", "gender", "has_cr_card", "is_active",
            "surname", "row_number", "customer_id",
        ):
            del st.session_state[k]
    st.rerun()

# ---- model + prediction ------------------------------------------------------
model, scaler, holdout_acc = get_model_and_scaler(df)

feature_row = pd.DataFrame([{
    "CreditScore": inputs["CreditScore"],
    "Geography": inputs["Geography"],      # notebook encoding applied at inference
    "Gender": inputs["Gender"],
    "Age": inputs["Age"],
    "Tenure": inputs["Tenure"],
    "Balance": inputs["Balance"],
    "NumOfProducts": inputs["NumOfProducts"],
    "HasCrCard": inputs["HasCrCard"],
    "IsActiveMember": inputs["IsActiveMember"],
    "EstimatedSalary": inputs["EstimatedSalary"],
}])

if predict_btn:
    with st.spinner("🔮 Crunching the gradient-boosted trees…"):
        prob = float(predict_churn_probability(model, scaler, feature_row)[0])
    st.session_state["last_prob"] = prob
prob = float(st.session_state.get(
    "last_prob",
    predict_churn_probability(model, scaler, feature_row)[0],
))

# ---- result panel ------------------------------------------------------------
with col_result:
    churn = prob >= 0.5
    emoji = "😟" if churn else "😄"           # churn -> not happy | stay -> happy
    verdict = "NOT HAPPY — likely to CHURN" if churn else "HAPPY — likely to STAY"
    color = "#c62828" if churn else "#2e7d32"
    gauge_color = "#e53935" if churn else "#43a047"
    risk = ("🔴 Very High" if prob >= 0.75 else
            "🟠 High" if prob >= 0.5 else
            "🟡 Moderate" if prob >= 0.3 else "🟢 Low")

    deg = round(prob * 360, 1)
    st.markdown(f"""
    <div class="card">
      <div style="display:flex; gap:1.2rem; align-items:center; flex-wrap:wrap;">
        <div class="gauge" style="background: conic-gradient({gauge_color} {deg}deg, #e8f5e9 {deg}deg);">
          <div class="gauge-inner">
            <div class="gauge-val" style="color:{color};">{prob * 100:.1f}%</div>
            <div class="gauge-lbl">churn probability</div>
          </div>
        </div>
        <div style="flex:1; min-width:180px; text-align:center;">
          <span class="big-emoji">{emoji}</span>
          <h3 style="color:{color}; margin:.3rem 0 0;">{verdict}</h3>
          <span class="chip">Risk level: {risk}</span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.progress(float(prob), text=f"Churn probability: {prob:.0%} · Retention: {1 - prob:.0%}")
    if holdout_acc is not None:
        st.caption(f"🤖 Gradient Boosting (notebook pipeline) · holdout accuracy **{holdout_acc:.1%}** "
                   f"· trained on {len(df):,} customers")
    else:
        st.caption("🤖 Gradient Boosting (notebook pipeline) · loaded from cached churn_gbr.pkl")

    # simple, explainable tips
    tips = []
    if not is_active:
        tips.append("🟠 Customer is **inactive** — engagement campaigns work best here.")
    if num_products >= 3:
        tips.append("🟠 Holding **3–4 products** historically raises churn — check satisfaction.")
    if balance > 100000 and num_products <= 1:
        tips.append("🟡 High balance with **1 product** — upsell to deepen the relationship.")
    if age >= 55:
        tips.append("🟡 **Older customers** churn more — offer personalised service.")
    if not tips:
        tips.append("🟢 Healthy profile — no red flags detected.")
    st.markdown('<div class="card"><h4>💡 Retention Tips</h4>'
                + "".join(f"<p style='margin:.35rem 0'>{t}</p>" for t in tips)
                + "</div>", unsafe_allow_html=True)

# ---- snapshot chips ----------------------------------------------------------
if snapshot_btn:
    st.markdown("##### 📋 Customer snapshot")
    chips = [
        f"🆔 {inputs['CustomerId']} · {inputs['Surname']}",
        f"💳 Score {inputs['CreditScore']}", f"🌍 {inputs['Geography']}",
        f"🚻 {inputs['Gender']}", f"🎂 {inputs['Age']} yrs",
        f"⏳ {inputs['Tenure']} yrs", f"💰 ${inputs['Balance']:,.0f}",
        f"📦 {inputs['NumOfProducts']} products",
        f"💳 Card: {'Yes' if inputs['HasCrCard'] else 'No'}",
        f"✅ Active: {'Yes' if inputs['IsActiveMember'] else 'No'}",
        f"💵 ${inputs['EstimatedSalary']:,.0f}",
    ]
    warn = not churn
    cls = "chip chip-warn" if warn else "chip"
    st.markdown("".join(f"<span class='{cls}'>{c}</span>" for c in chips),
                unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Insights tabs
# ----------------------------------------------------------------------------
tab_data, tab_viz = st.tabs(["🗂️ Raw data", "📊 Dataset insights"])

with tab_data:
    st.dataframe(df.head(200), use_container_width=True, height=360)
    st.caption(f"Showing first 200 of {len(df):,} rows")

with tab_viz:
    k1, k2, k3, k4 = st.columns(4)
    churn_rate = df["Exited"].mean()
    k1.metric("👥 Customers", f"{len(df):,}")
    k2.metric("😟 Churn rate", f"{churn_rate:.1%}")
    k3.metric("🌍 Countries", df["Geography"].nunique())
    k4.metric("🎂 Avg. age", f"{df['Age'].mean():.1f}")

    v1, v2 = st.columns(2)
    with v1:
        st.markdown("##### 😟 Churn rate by geography")
        st.bar_chart(df.groupby("Geography")["Exited"].mean().rename("churn rate"))
    with v2:
        st.markdown("##### 📦 Churn rate by number of products")
        st.bar_chart(df.groupby("NumOfProducts")["Exited"].mean().rename("churn rate"))

    st.markdown("##### 🎂 Churn rate by age group")
    bins = [18, 30, 40, 50, 60, 100]
    labels = ["18–29", "30–39", "40–49", "50–59", "60+"]
    age_groups = pd.cut(df["Age"], bins=bins, labels=labels, right=False)
    st.bar_chart(df.groupby(age_groups, observed=True)["Exited"].mean().rename("churn rate"))

# ----------------------------------------------------------------------------
# 📁 Batch predictions — drag & drop your data (bottom of the page)
# ----------------------------------------------------------------------------
st.write("")
st.divider()
st.markdown("""
<div class="card drop-zone">
  <h4>📁 Batch Predictions — drop your files here</h4>
  <p style="color:#4e6b52; margin-top:.2rem;">
    Upload <b>.csv</b>, <b>.xlsx</b> or <b>.xls</b> files containing the customer columns.
    Every row is scored by the notebook's Gradient Boosting model — and if your file
    includes the <b>Exited</b> column, accuracy is evaluated too.
  </p>
</div>
""", unsafe_allow_html=True)

uploaded_files = st.file_uploader(
    "Drop CSV or XLSX files here",
    type=["csv", "xlsx", "xls"],
    accept_multiple_files=True,
    label_visibility="collapsed",
)


def read_upload(upload) -> pd.DataFrame:
    """Read an uploaded csv/xlsx/xls file into a DataFrame."""
    name = upload.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(upload)
    return pd.read_excel(upload)


def prepare_batch(raw: pd.DataFrame, filename: str):
    """Normalise an uploaded table to model-ready feature rows.

    Returns (features_df, y_true_or_None). Accepts both raw files
    (France/Male…) and already-encoded files (2/1/0 and 1/0).
    """
    data = raw.copy()
    data.columns = [str(c).strip() for c in data.columns]

    missing = [c for c in FEATURE_NAMES if c not in data.columns]
    if missing:
        raise ValueError(f"missing required columns: {', '.join(missing)}")

    # already label-encoded? (numeric Geography/Gender) — keep as-is
    if pd.api.types.is_numeric_dtype(data["Geography"]) and \
       pd.api.types.is_numeric_dtype(data["Gender"]):
        geo, gen = data["Geography"], data["Gender"]
        if not geo.isin([0, 1, 2]).all() or not gen.isin([0, 1]).all():
            raise ValueError("encoded Geography must be 0/1/2 and Gender 0/1")
        data["Geography"], data["Gender"] = geo.astype(int), gen.astype(int)
    else:
        data = notebook_preprocess(data)

    y_true = None
    if "Exited" in data.columns:
        y_true = data["Exited"].astype(int)

    return data[FEATURE_NAMES], y_true


if uploaded_files:
    for upload in uploaded_files:
        fname = upload.name
        with st.expander(f"📄 {fname}", expanded=len(uploaded_files) == 1):
            try:
                raw = read_upload(upload)
                features, y_true = prepare_batch(raw, fname)

                probs = predict_churn_probability(model, scaler, features)
                preds = (probs >= 0.5).astype(int)

                out = raw.copy()
                out["ChurnProbability"] = np.round(probs, 4)
                out["Prediction"] = np.where(preds == 1, "😟 Not Happy (Churn)",
                                             "😄 Happy (Stay)")

                m1, m2, m3 = st.columns(3)
                m1.metric("Rows scored", f"{len(out):,}")
                m2.metric("😟 Predicted churners", f"{int(preds.sum()):,}",
                          f"{preds.mean():.1%} of batch")
                m3.metric("😄 Predicted stayers", f"{int((1 - preds).sum()):,}",
                          f"{1 - preds.mean():.1%} of batch")

                if y_true is not None:
                    acc = accuracy_score(y_true, preds)
                    cm = confusion_matrix(y_true, preds, labels=[0, 1])
                    e1, e2 = st.columns([0.35, 0.65])
                    with e1:
                        st.metric("🎯 Accuracy vs Exited", f"{acc:.1%}")
                    with e2:
                        cm_df = pd.DataFrame(
                            cm,
                            index=["Actual 😄 Stay (0)", "Actual 😟 Churn (1)"],
                            columns=["Pred 😄 Stay (0)", "Pred 😟 Churn (1)"],
                        )
                        st.table(cm_df)

                st.dataframe(out.head(50), use_container_width=True)
                if len(out) > 50:
                    st.caption(f"Showing first 50 of {len(out):,} scored rows")

                st.download_button(
                    "⬇️ Download scored results (CSV)",
                    data=out.to_csv(index=False).encode("utf-8"),
                    file_name=f"scored_{os.path.splitext(fname)[0]}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            except Exception as exc:
                st.error(f"❌ Could not process **{fname}** — {exc}")
else:
    st.info("👋 No files yet — drag a **.csv** or **.xlsx** file onto the box above "
            "to score a whole batch of customers at once.")
