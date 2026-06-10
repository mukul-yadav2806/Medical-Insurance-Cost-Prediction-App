import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="MedCost AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
#  GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Syne:wght@700;800&display=swap');

/* ── Root & background ── */
html, body, [data-testid="stAppViewContainer"] {
    background: #0a0e1a !important;
    color: #e8eaf0;
    font-family: 'Inter', sans-serif;
}
[data-testid="stSidebar"] {
    background: #0d1120 !important;
    border-right: 1px solid #1e2538;
}
[data-testid="stHeader"] { background: transparent !important; }

/* ── Typography ── */
h1, h2, h3 { font-family: 'Syne', sans-serif !important; }

/* ── Metric cards ── */
.metric-card {
    background: linear-gradient(135deg, #111827 0%, #1a2035 100%);
    border: 1px solid #1e2d4a;
    border-radius: 16px;
    padding: 20px 24px;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #3b82f6, #06b6d4);
}
.metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #3b82f6, #06b6d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.metric-label {
    font-size: 0.75rem;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 4px;
}

/* ── Prediction result ── */
.prediction-box {
    background: linear-gradient(135deg, #0f2027, #1a3a4a, #0d2137);
    border: 1px solid #0ea5e9;
    border-radius: 20px;
    padding: 32px;
    text-align: center;
    margin: 16px 0;
    box-shadow: 0 0 40px rgba(14, 165, 233, 0.15);
}
.prediction-amount {
    font-family: 'Syne', sans-serif;
    font-size: 3.5rem;
    font-weight: 800;
    background: linear-gradient(135deg, #0ea5e9, #38bdf8, #7dd3fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.prediction-label {
    font-size: 0.85rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    margin-bottom: 8px;
}

/* ── Risk badge ── */
.risk-low  { background: #052e16; border: 1px solid #16a34a; color: #4ade80; padding: 6px 16px; border-radius: 20px; font-size: 0.8rem; font-weight: 600; display: inline-block; }
.risk-med  { background: #1c1003; border: 1px solid #d97706; color: #fbbf24; padding: 6px 16px; border-radius: 20px; font-size: 0.8rem; font-weight: 600; display: inline-block; }
.risk-high { background: #1c0505; border: 1px solid #dc2626; color: #f87171; padding: 6px 16px; border-radius: 20px; font-size: 0.8rem; font-weight: 600; display: inline-block; }

/* ── Section headers ── */
.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: #e2e8f0;
    margin-bottom: 4px;
}
.section-sub {
    font-size: 0.8rem;
    color: #475569;
    margin-bottom: 20px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* ── Sidebar inputs ── */
[data-testid="stSlider"] > div > div > div > div {
    background: #3b82f6 !important;
}
.stSelectbox > div > div {
    background: #111827 !important;
    border: 1px solid #1e2d4a !important;
    color: #e8eaf0 !important;
    border-radius: 10px !important;
}

/* ── Divider ── */
.fancy-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #1e2d4a, transparent);
    margin: 24px 0;
}

/* ── Info chip ── */
.info-chip {
    display: inline-block;
    background: #0f172a;
    border: 1px solid #1e2d4a;
    border-radius: 8px;
    padding: 4px 12px;
    font-size: 0.78rem;
    color: #64748b;
    margin: 2px;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0a0e1a; }
::-webkit-scrollbar-thumb { background: #1e2d4a; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  DATA & MODEL LOADING
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("C:\\Users\\USER\\Desktop\\insurance.csv")
    return df

@st.cache_resource
def train_models(df):
    le_sex    = LabelEncoder()
    le_smoker = LabelEncoder()
    le_region = LabelEncoder()

    X = df.copy()
    X['sex']    = le_sex.fit_transform(X['sex'])
    X['smoker'] = le_smoker.fit_transform(X['smoker'])
    X['region'] = le_region.fit_transform(X['region'])

    features = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']
    X_feat = X[features]
    y = df['charges']

    X_train, X_test, y_train, y_test = train_test_split(X_feat, y, test_size=0.2, random_state=42)

    models = {
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42),
        'Random Forest':     RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42),
        'Ridge Regression':  Ridge(alpha=1.0),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2  = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse= np.sqrt(mean_squared_error(y_test, y_pred))
        results[name] = {'model': model, 'r2': r2, 'mae': mae, 'rmse': rmse,
                         'y_test': y_test, 'y_pred': y_pred}

    encoders = {'sex': le_sex, 'smoker': le_smoker, 'region': le_region}
    return results, encoders, features

df = load_data()
model_results, encoders, features = train_models(df)
best_model_name = max(model_results, key=lambda k: model_results[k]['r2'])
best_model = model_results[best_model_name]['model']

# ─────────────────────────────────────────────
#  SIDEBAR — PATIENT INPUTS
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 20px 0 10px 0;'>
        <div style='font-family:Syne,sans-serif; font-size:1.5rem; font-weight:800; 
                    background:linear-gradient(135deg,#3b82f6,#06b6d4);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
            MedCost AI
        </div>
        <div style='font-size:0.72rem; color:#4b5563; letter-spacing:0.1em; text-transform:uppercase; margin-top:2px;'>
            Insurance Charge Predictor
        </div>
    </div>
    <div class='fancy-divider'></div>
    """, unsafe_allow_html=True)

    st.markdown("**👤 Demographics**")
    age = st.slider("Age", 18, 64, 32)
    sex = st.selectbox("Biological Sex", ["male", "female"])
    region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

    st.markdown("<div class='fancy-divider'></div>", unsafe_allow_html=True)
    st.markdown("**🩺 Health Profile**")
    bmi = st.slider("BMI", 10.0, 55.0, 26.5, 0.1,
                    help="Body Mass Index — weight(kg)/height(m)²")

    bmi_cat = ("Underweight" if bmi < 18.5 else
               "Normal"      if bmi < 25 else
               "Overweight"  if bmi < 30 else "Obese")
    bmi_col = ("#60a5fa" if bmi < 18.5 else "#4ade80" if bmi < 25 else "#fbbf24" if bmi < 30 else "#f87171")
    st.markdown(f"<span style='font-size:0.78rem; color:{bmi_col}; font-weight:600;'>● {bmi_cat}</span>", unsafe_allow_html=True)

    children = st.slider("Dependents / Children", 0, 5, 1)
    smoker   = st.selectbox("Smoking Status", ["no", "yes"])

    st.markdown("<div class='fancy-divider'></div>", unsafe_allow_html=True)
    model_choice = st.selectbox("🤖 ML Model",
                                list(model_results.keys()),
                                index=list(model_results.keys()).index(best_model_name))
    st.markdown(f"<span class='info-chip'>Best: {best_model_name}</span>", unsafe_allow_html=True)

    predict_btn = st.button("⚡ Predict Cost", use_container_width=True, type="primary")

# ─────────────────────────────────────────────
#  PREDICTION LOGIC
# ─────────────────────────────────────────────
def make_prediction(age, sex, bmi, children, smoker, region, model_name):
    row = {
        'age':      age,
        'sex':      encoders['sex'].transform([sex])[0],
        'bmi':      bmi,
        'children': children,
        'smoker':   encoders['smoker'].transform([smoker])[0],
        'region':   encoders['region'].transform([region])[0],
    }
    X = pd.DataFrame([row])[features]
    pred = model_results[model_name]['model'].predict(X)[0]
    return pred

if 'prediction' not in st.session_state:
    st.session_state.prediction = None

if predict_btn:
    st.session_state.prediction = make_prediction(age, sex, bmi, children, smoker, region, model_choice)
    st.session_state.inputs = dict(age=age, sex=sex, bmi=bmi, children=children, smoker=smoker, region=region)

# ─────────────────────────────────────────────
#  MAIN LAYOUT — TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🎯 Prediction", "📊 Data Insights", "🤖 Model Performance"])

# ══════════════════════════════════════════════
#  TAB 1 — PREDICTION
# ══════════════════════════════════════════════
with tab1:
    # Header
    st.markdown("""
    <div style='padding: 8px 0 24px 0;'>
        <div class='section-title'>Medical Insurance Cost Predictor</div>
        <div class='section-sub'>Gradient-boosted ML · 1,338 patient records · 6 features</div>
    </div>
    """, unsafe_allow_html=True)

    # Dataset summary metrics
    c1, c2, c3, c4 = st.columns(4)
    metrics = [
        (f"${df['charges'].mean():,.0f}", "Avg Annual Charge"),
        (f"${df['charges'].median():,.0f}", "Median Charge"),
        (f"{df['smoker'].value_counts(normalize=True)['yes']*100:.1f}%", "Smoker Rate"),
        (f"{df['bmi'].mean():.1f}", "Avg BMI"),
    ]
    for col, (val, label) in zip([c1, c2, c3, c4], metrics):
        with col:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{val}</div>
                <div class='metric-label'>{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)

    left, right = st.columns([1.1, 0.9])

    with left:
        if st.session_state.prediction is not None:
            pred = st.session_state.prediction
            annual = pred
            monthly = pred / 12

            # Risk tier
            percentile = (df['charges'] < pred).mean() * 100
            if percentile < 40:
                risk_html = "<span class='risk-low'>🟢 LOW RISK</span>"
                risk_note = "Below median cost profile"
            elif percentile < 70:
                risk_html = "<span class='risk-med'>🟡 MODERATE RISK</span>"
                risk_note = "Above-average cost profile"
            else:
                risk_html = "<span class='risk-high'>🔴 HIGH RISK</span>"
                risk_note = "High-cost profile — consider wellness programs"

            st.markdown(f"""
            <div class='prediction-box'>
                <div class='prediction-label'>Estimated Annual Insurance Cost</div>
                <div class='prediction-amount'>${annual:,.2f}</div>
                <div style='color:#64748b; font-size:0.85rem; margin:8px 0 16px;'>
                    ≈ <strong style='color:#94a3b8;'>${monthly:,.2f}</strong> per month
                </div>
                {risk_html}
                <div style='font-size:0.75rem; color:#4b5563; margin-top:8px;'>{risk_note}</div>
                <div style='font-size:0.72rem; color:#374151; margin-top:12px;'>
                    {percentile:.0f}th percentile · Model: {model_choice}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Feature contribution visualization
            inp = st.session_state.inputs
            contrib_data = {
                'Age':      age * 250,
                'BMI':      max(0, (bmi - 18.5) * 180),
                'Smoking':  23000 if smoker == 'yes' else 0,
                'Children': children * 400,
                'Sex':      200 if sex == 'male' else 0,
                'Region':   300 if region in ['northeast','southeast'] else 0,
            }
            total_c = sum(contrib_data.values())
            contrib_pct = {k: v/total_c*100 for k, v in contrib_data.items() if v > 0}

            colors = ['#3b82f6','#06b6d4','#f43f5e','#8b5cf6','#10b981','#f59e0b']
            fig_donut = go.Figure(go.Pie(
                labels=list(contrib_pct.keys()),
                values=list(contrib_pct.values()),
                hole=0.65,
                marker_colors=colors[:len(contrib_pct)],
                textinfo='label+percent',
                textfont=dict(color='#e2e8f0', size=11),
                hovertemplate='%{label}: %{percent}<extra></extra>'
            ))
            fig_donut.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                showlegend=False, height=240, margin=dict(t=10,b=10,l=10,r=10),
                annotations=[dict(text='Cost<br>Drivers', x=0.5, y=0.5,
                                  font=dict(size=13, color='#94a3b8', family='Inter'), showarrow=False)]
            )
            st.plotly_chart(fig_donut, use_container_width=True)
        else:
            st.markdown("""
            <div style='background:#0d1120; border:1px dashed #1e2d4a; border-radius:16px;
                        padding:48px; text-align:center; color:#374151;'>
                <div style='font-size:2.5rem; margin-bottom:12px;'>⚡</div>
                <div style='font-family:Syne,sans-serif; font-size:1.1rem; color:#4b5563;'>
                    Configure profile & click Predict
                </div>
                <div style='font-size:0.78rem; margin-top:8px;'>
                    Use the sidebar inputs on the left
                </div>
            </div>
            """, unsafe_allow_html=True)

    with right:
        st.markdown("<div class='section-title' style='font-size:1rem;'>Comparable Profiles</div>", unsafe_allow_html=True)

        # Find similar patients
        age_range = df[(df['age'].between(age-5, age+5)) &
                       (df['smoker'] == smoker)].copy()
        if len(age_range) < 5:
            age_range = df[df['smoker'] == smoker].copy()

        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(
            x=age_range['charges'],
            nbinsx=30,
            marker=dict(color='#1e3a5f', line=dict(color='#3b82f6', width=0.5)),
            name='Similar Profiles'
        ))
        if st.session_state.prediction:
            fig_dist.add_vline(x=st.session_state.prediction,
                               line=dict(color='#0ea5e9', width=2, dash='dash'),
                               annotation_text="Your estimate",
                               annotation_font=dict(color='#0ea5e9', size=11))
        fig_dist.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(color='#4b5563', title='Annual Charges ($)', gridcolor='#111827'),
            yaxis=dict(color='#4b5563', title='Count', gridcolor='#111827'),
            height=200, margin=dict(t=10,b=30,l=40,r=10),
            showlegend=False, font=dict(color='#6b7280')
        )
        st.plotly_chart(fig_dist, use_container_width=True)

        # BMI vs Charges scatter (smoker colored)
        st.markdown("<div class='section-title' style='font-size:1rem; margin-top:8px;'>BMI vs Charges</div>", unsafe_allow_html=True)
        fig_scatter = px.scatter(df, x='bmi', y='charges', color='smoker',
                                 color_discrete_map={'yes':'#f43f5e','no':'#3b82f6'},
                                 opacity=0.5, height=200)
        if st.session_state.prediction:
            fig_scatter.add_trace(go.Scatter(
                x=[bmi], y=[st.session_state.prediction],
                mode='markers',
                marker=dict(color='#fbbf24', size=14, symbol='star',
                            line=dict(color='white', width=1.5)),
                name='You', showlegend=True
            ))
        fig_scatter.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(color='#4b5563', gridcolor='#111827'),
            yaxis=dict(color='#4b5563', gridcolor='#111827'),
            margin=dict(t=10,b=30,l=50,r=10),
            legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#9ca3af', size=10)),
            font=dict(color='#6b7280')
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

# ══════════════════════════════════════════════
#  TAB 2 — DATA INSIGHTS
# ══════════════════════════════════════════════
with tab2:
    st.markdown("""
    <div class='section-title'>Dataset Insights</div>
    <div class='section-sub'>1,338 records · Kaggle Medical Cost Personal Dataset</div>
    """, unsafe_allow_html=True)

    row1 = st.columns(2)

    # Charges distribution
    with row1[0]:
        fig = px.histogram(df, x='charges', nbins=50,
                           title='Charges Distribution',
                           color_discrete_sequence=['#3b82f6'])
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          xaxis=dict(color='#4b5563', gridcolor='#0f172a'),
                          yaxis=dict(color='#4b5563', gridcolor='#0f172a'),
                          title_font=dict(color='#e2e8f0', family='Syne'),
                          margin=dict(t=40,b=20,l=20,r=10), height=280,
                          font=dict(color='#6b7280'))
        st.plotly_chart(fig, use_container_width=True)

    # Smoker vs non-smoker box
    with row1[1]:
        fig = px.box(df, x='smoker', y='charges', color='smoker',
                     color_discrete_map={'yes':'#f43f5e','no':'#3b82f6'},
                     title='Smoker vs Non-Smoker Charges')
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          xaxis=dict(color='#4b5563', gridcolor='#0f172a'),
                          yaxis=dict(color='#4b5563', gridcolor='#0f172a'),
                          title_font=dict(color='#e2e8f0', family='Syne'),
                          showlegend=False, margin=dict(t=40,b=20,l=20,r=10), height=280,
                          font=dict(color='#6b7280'))
        st.plotly_chart(fig, use_container_width=True)

    row2 = st.columns(2)

    # Age vs charges
    with row2[0]:
        fig = px.scatter(df, x='age', y='charges', color='smoker', trendline='ols',
                         color_discrete_map={'yes':'#f43f5e','no':'#3b82f6'},
                         title='Age vs Charges', opacity=0.6)
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          xaxis=dict(color='#4b5563', gridcolor='#0f172a'),
                          yaxis=dict(color='#4b5563', gridcolor='#0f172a'),
                          title_font=dict(color='#e2e8f0', family='Syne'),
                          legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#9ca3af')),
                          margin=dict(t=40,b=20,l=20,r=10), height=280,
                          font=dict(color='#6b7280'))
        st.plotly_chart(fig, use_container_width=True)

    # Avg charges by region
    with row2[1]:
        reg_avg = df.groupby('region')['charges'].mean().reset_index().sort_values('charges')
        fig = px.bar(reg_avg, x='charges', y='region', orientation='h',
                     title='Average Charges by Region',
                     color='charges', color_continuous_scale='Blues')
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          xaxis=dict(color='#4b5563', gridcolor='#0f172a'),
                          yaxis=dict(color='#4b5563'),
                          title_font=dict(color='#e2e8f0', family='Syne'),
                          coloraxis_showscale=False,
                          margin=dict(t=40,b=20,l=20,r=10), height=280,
                          font=dict(color='#6b7280'))
        st.plotly_chart(fig, use_container_width=True)

    # Heatmap — correlation
    st.markdown("<div class='section-title' style='font-size:1rem; margin-top:8px;'>Feature Correlation</div>", unsafe_allow_html=True)
    df_enc = df.copy()
    df_enc['sex']    = (df_enc['sex']=='male').astype(int)
    df_enc['smoker'] = (df_enc['smoker']=='yes').astype(int)
    df_enc['region'] = df_enc['region'].map({'northeast':0,'northwest':1,'southeast':2,'southwest':3})
    corr = df_enc.corr()

    fig_hm = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.columns,
        colorscale='RdBu', zmid=0,
        text=np.round(corr.values, 2), texttemplate='%{text}',
        textfont=dict(size=11, color='white')
    ))
    fig_hm.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                         height=340, margin=dict(t=10,b=10,l=80,r=10),
                         xaxis=dict(color='#9ca3af'), yaxis=dict(color='#9ca3af'),
                         font=dict(color='#6b7280'))
    st.plotly_chart(fig_hm, use_container_width=True)

    # Raw data preview
    with st.expander("📋 Raw Dataset Preview"):
        st.dataframe(df.head(50).style.background_gradient(subset=['charges'],
                     cmap='Blues'), use_container_width=True)

# ══════════════════════════════════════════════
#  TAB 3 — MODEL PERFORMANCE
# ══════════════════════════════════════════════
with tab3:
    st.markdown("""
    <div class='section-title'>Model Performance</div>
    <div class='section-sub'>80/20 train-test split · 3 algorithms compared</div>
    """, unsafe_allow_html=True)

    # Model comparison table
    perf_rows = []
    for name, res in model_results.items():
        best_tag = " ★" if name == best_model_name else ""
        perf_rows.append({
            'Model': name + best_tag,
            'R² Score': f"{res['r2']:.4f}",
            'MAE ($)': f"{res['mae']:,.2f}",
            'RMSE ($)': f"{res['rmse']:,.2f}",
        })
    perf_df = pd.DataFrame(perf_rows)

    m1, m2, m3 = st.columns(3)
    best = model_results[best_model_name]
    for col, (val, label) in zip([m1, m2, m3], [
        (f"{best['r2']:.4f}", f"{best_model_name} · R²"),
        (f"${best['mae']:,.0f}", "Mean Absolute Error"),
        (f"${best['rmse']:,.0f}", "Root Mean Sq Error"),
    ]):
        with col:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-value'>{val}</div>
                <div class='metric-label'>{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)

    left, right = st.columns(2)

    with left:
        # Actual vs Predicted
        res = model_results[model_choice]
        fig_ap = go.Figure()
        fig_ap.add_trace(go.Scatter(
            x=res['y_test'], y=res['y_pred'],
            mode='markers',
            marker=dict(color='#3b82f6', opacity=0.5, size=5),
            name='Predictions'
        ))
        mn, mx = df['charges'].min(), df['charges'].max()
        fig_ap.add_trace(go.Scatter(x=[mn,mx], y=[mn,mx],
                                    line=dict(color='#f43f5e', width=1.5, dash='dash'),
                                    name='Perfect'))
        fig_ap.update_layout(
            title=f'Actual vs Predicted — {model_choice}',
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(color='#4b5563', title='Actual ($)', gridcolor='#0f172a'),
            yaxis=dict(color='#4b5563', title='Predicted ($)', gridcolor='#0f172a'),
            title_font=dict(color='#e2e8f0', family='Syne'),
            legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#9ca3af')),
            height=340, margin=dict(t=40,b=30,l=50,r=10),
            font=dict(color='#6b7280')
        )
        st.plotly_chart(fig_ap, use_container_width=True)

    with right:
        # R² Bar comparison
        names  = list(model_results.keys())
        r2vals = [model_results[n]['r2'] for n in names]
        bar_colors = ['#0ea5e9' if n == best_model_name else '#1e3a5f' for n in names]

        fig_bar = go.Figure(go.Bar(
            x=names, y=r2vals,
            marker_color=bar_colors,
            text=[f"{v:.4f}" for v in r2vals],
            textposition='outside',
            textfont=dict(color='#94a3b8')
        ))
        fig_bar.update_layout(
            title='R² Score Comparison',
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(color='#4b5563', gridcolor='#0f172a'),
            yaxis=dict(color='#4b5563', gridcolor='#0f172a', range=[0,1]),
            title_font=dict(color='#e2e8f0', family='Syne'),
            height=340, margin=dict(t=40,b=30,l=50,r=10),
            font=dict(color='#6b7280')
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # Feature importance for best model
    if hasattr(best_model, 'feature_importances_'):
        fi = pd.DataFrame({'Feature': features, 'Importance': best_model.feature_importances_})
        fi = fi.sort_values('Importance')
        fig_fi = px.bar(fi, x='Importance', y='Feature', orientation='h',
                        title=f'Feature Importance — {best_model_name}',
                        color='Importance', color_continuous_scale='Blues')
        fig_fi.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                             xaxis=dict(color='#4b5563', gridcolor='#0f172a'),
                             yaxis=dict(color='#4b5563'),
                             title_font=dict(color='#e2e8f0', family='Syne'),
                             coloraxis_showscale=False,
                             height=300, margin=dict(t=40,b=20,l=20,r=10),
                             font=dict(color='#6b7280'))
        st.plotly_chart(fig_fi, use_container_width=True)

    # Residuals
    res = model_results[model_choice]
    residuals = np.array(res['y_test']) - np.array(res['y_pred'])
    fig_res = px.histogram(x=residuals, nbins=50,
                           title='Residual Distribution',
                           color_discrete_sequence=['#8b5cf6'])
    fig_res.add_vline(x=0, line=dict(color='#f43f5e', width=1.5, dash='dash'))
    fig_res.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                          xaxis=dict(color='#4b5563', title='Residual ($)', gridcolor='#0f172a'),
                          yaxis=dict(color='#4b5563', gridcolor='#0f172a'),
                          title_font=dict(color='#e2e8f0', family='Syne'),
                          height=260, margin=dict(t=40,b=20,l=20,r=10),
                          font=dict(color='#6b7280'), showlegend=False)
    st.plotly_chart(fig_res, use_container_width=True)

    # Performance table
    st.markdown("<div class='section-title' style='font-size:1rem;'>Full Model Metrics</div>", unsafe_allow_html=True)
    st.dataframe(perf_df, use_container_width=True, hide_index=True)
