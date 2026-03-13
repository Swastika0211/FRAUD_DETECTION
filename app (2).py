import streamlit as st

st.set_page_config(
    page_title="FraudShield AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
[data-testid="stSidebar"] { background: #0a0e1a !important; border-right: 1px solid #1e2d4a; }
[data-testid="stSidebar"] * { color: #c8d6f0 !important; }
.stApp { background: #06090f; }
.main .block-container { padding: 2rem 2.5rem; max-width: 1400px; }
.metric-card { background: #0d1526; border: 1px solid #1a2a4a; border-radius: 12px; padding: 1.2rem 1.5rem; text-align: center; }
.metric-label { font-size: 12px; color: #5a7ab0; text-transform: uppercase; letter-spacing: 1.5px; font-family: 'Space Mono', monospace; }
.metric-value { font-size: 2rem; font-weight: 600; color: #e8f0ff; margin: 4px 0; font-family: 'Space Mono', monospace; }
.metric-sub { font-size: 12px; color: #3a5a8a; }
.section-header { font-family: 'Space Mono', monospace; font-size: 11px; color: #3a6aaa; text-transform: uppercase; letter-spacing: 3px; margin-bottom: 1rem; padding-bottom: 6px; border-bottom: 1px solid #1a2a4a; }
.risk-low    { background:#0a2a1a; color:#4ade80; border:1px solid #16a34a; border-radius:8px; padding:6px 14px; font-weight:600; display:inline-block; }
.risk-medium { background:#2a1e0a; color:#fbbf24; border:1px solid #d97706; border-radius:8px; padding:6px 14px; font-weight:600; display:inline-block; }
.risk-high   { background:#2a0a0a; color:#f87171; border:1px solid #dc2626; border-radius:8px; padding:6px 14px; font-weight:600; display:inline-block; }
.stButton > button { background: #1a3a6a; color: #e8f0ff; border: 1px solid #2a5aaa; border-radius: 8px; font-family: 'Space Mono', monospace; font-size: 13px; padding: 10px 24px; transition: all 0.2s; }
.stButton > button:hover { background: #2a5aaa; border-color: #4a8ada; }
.alert-success { background:#0a2a1a; border-left:3px solid #22c55e; padding:12px 16px; border-radius:0 8px 8px 0; color:#86efac; margin:8px 0; }
.alert-warning { background:#2a1e00; border-left:3px solid #f59e0b; padding:12px 16px; border-radius:0 8px 8px 0; color:#fcd34d; margin:8px 0; }
.alert-danger  { background:#2a0808; border-left:3px solid #ef4444; padding:12px 16px; border-radius:0 8px 8px 0; color:#fca5a5; margin:8px 0; }
.score-track { background: #0d1526; border-radius: 99px; height: 14px; width: 100%; border: 1px solid #1a2a4a; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
import warnings, os
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix, roc_curve)

RANDOM_STATE = 42
CATEGORICAL_COLS = ['UserLocation', 'DeviceType', 'PaymentMethod']
TARGET = 'IsFraud'
PAYMENT_RISK = {'Crypto':3,'Credit Card':2,'PayPal':1,'Debit Card':1,'Bank Transfer':0}

def dark_layout(**kwargs):
    """Return a dark-theme layout dict safe for fig.update_layout()."""
    base = dict(
        plot_bgcolor='#0d1526',
        paper_bgcolor='#0d1526',
        font=dict(color='#8ab4d8'),
        xaxis=dict(gridcolor='#1a2a4a', zerolinecolor='#1a2a4a'),
        yaxis=dict(gridcolor='#1a2a4a', zerolinecolor='#1a2a4a'),
    )
    base.update(kwargs)
    return base

# Keep DARK as alias for charts that don't use axes (pie, gauge, etc.)
DARK = dict(plot_bgcolor='#0d1526', paper_bgcolor='#0d1526',
            font=dict(color='#8ab4d8'))

DATA_PATH = os.path.join(os.path.dirname(__file__), "fraud_transactions.csv")

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_data
def engineer_features(df):
    df = df.copy()
    df['AmountLog']          = np.log1p(df['TransactionAmount'])
    df['IsNightTransaction'] = ((df['TransactionHour'] < 6) | (df['TransactionHour'] >= 22)).astype(int)
    df['IsWeekend']          = (df['DayOfWeek'] >= 5).astype(int)
    df['IsNewAccount']       = (df['AccountAge'] < 90).astype(int)
    df['HighFrequency']      = (df['TransactionFrequency'] > 8).astype(int)
    df['HighAmount']         = (df['TransactionAmount'] > df['TransactionAmount'].quantile(0.90)).astype(int)
    df['RiskIndicator']      = (df['IsInternational'] + df['PreviousFraudHistory']*2 +
                                df['IsNightTransaction'] + df['IsNewAccount'] + df['HighFrequency'])
    df['PaymentRisk']        = df['PaymentMethod'].map(PAYMENT_RISK).fillna(1)
    return df

def smote_oversample(X, y, k=5, seed=42):
    rng = np.random.default_rng(seed)
    minority_idx = np.where(y == 1)[0]
    X_min = X[minority_idx]
    n_to_gen = (y==0).sum() - (y==1).sum()
    if n_to_gen <= 0:
        return X, y
    synthetic = []
    for _ in range(n_to_gen):
        i = rng.integers(0, len(X_min))
        sample = X_min[i]
        dists = np.sum((X_min - sample)**2, axis=1)
        dists[i] = np.inf
        nn = X_min[rng.choice(np.argsort(dists)[:k])]
        synthetic.append(sample + rng.random() * (nn - sample))
    return np.vstack([X, np.array(synthetic)]), np.concatenate([y, np.ones(n_to_gen, dtype=int)])

def train_all_models(progress_cb=None):
    df = engineer_features(load_data())
    le_dict = {}
    df_m = df.copy()
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        df_m[col] = le.fit_transform(df_m[col].astype(str))
        le_dict[col] = le
    features = [c for c in df_m.columns if c not in [TARGET,'TransactionID']]
    X = df_m[features].values
    y = df_m[TARGET].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                         random_state=RANDOM_STATE, stratify=y)
    if progress_cb: progress_cb(0.15, "Applying SMOTE...")
    X_tr_sm, y_tr_sm = smote_oversample(X_train, y_train, seed=RANDOM_STATE)
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr_sm)
    X_te_sc = scaler.transform(X_test)

    models_def = {
        'Logistic Regression': (LogisticRegression(max_iter=1000, C=1.0, class_weight='balanced',
                                                    random_state=RANDOM_STATE), True),
        'Random Forest':       (RandomForestClassifier(n_estimators=150, max_depth=15,
                                                        min_samples_leaf=4, class_weight='balanced',
                                                        random_state=RANDOM_STATE, n_jobs=-1), False),
        'Gradient Boosting':   (GradientBoostingClassifier(n_estimators=150, max_depth=5,
                                                             learning_rate=0.1,
                                                             random_state=RANDOM_STATE), False),
    }
    trained, results = {}, {}
    for i, (name, (model, use_sc)) in enumerate(models_def.items()):
        if progress_cb: progress_cb(0.30 + i*0.22, f"Training {name}...")
        Xtr = X_tr_sc if use_sc else X_tr_sm
        Xte = X_te_sc if use_sc else X_test
        model.fit(Xtr, y_tr_sm)
        y_pred  = model.predict(Xte)
        y_proba = model.predict_proba(Xte)[:,1]
        trained[name] = model
        results[name] = {
            'Accuracy' : round(accuracy_score(y_test, y_pred), 4),
            'Precision': round(precision_score(y_test, y_pred, zero_division=0), 4),
            'Recall'   : round(recall_score(y_test, y_pred), 4),
            'F1 Score' : round(f1_score(y_test, y_pred), 4),
            'ROC-AUC'  : round(roc_auc_score(y_test, y_proba), 4),
            'y_pred': y_pred, 'y_proba': y_proba,
            'cm': confusion_matrix(y_test, y_pred),
            'roc': roc_curve(y_test, y_proba),
        }
    if progress_cb: progress_cb(0.95, "Finalizing...")
    best_name = max(results, key=lambda n: results[n]['ROC-AUC'])
    rf = trained['Random Forest']
    return {'models': trained, 'results': results, 'best_name': best_name,
            'scaler': scaler, 'le_dict': le_dict, 'features': features,
            'y_test': y_test, 'importances': dict(zip(features, rf.feature_importances_))}

def predict_single(inp, state):
    features, le_dict = state['features'], state['le_dict']
    model   = state['models'][state['best_name']]
    scaler  = state['scaler']
    row = dict(inp)
    for col in CATEGORICAL_COLS:
        le = le_dict[col]
        try:    row[col] = int(le.transform([row[col]])[0])
        except: row[col] = 0
    row['AmountLog']          = np.log1p(row['TransactionAmount'])
    row['IsNightTransaction'] = int(row['TransactionHour'] < 6 or row['TransactionHour'] >= 22)
    row['IsWeekend']          = int(row['DayOfWeek'] >= 5)
    row['IsNewAccount']       = int(row['AccountAge'] < 90)
    row['HighFrequency']      = int(row['TransactionFrequency'] > 8)
    row['HighAmount']         = int(row['TransactionAmount'] > 1500)
    row['RiskIndicator']      = (row['IsInternational'] + row['PreviousFraudHistory']*2 +
                                 row['IsNightTransaction'] + row['IsNewAccount'] + row['HighFrequency'])
    row['PaymentRisk']        = PAYMENT_RISK.get(inp['PaymentMethod'], 1)
    X_row = np.array([[row[f] for f in features]])
    if state['best_name'] == 'Logistic Regression':
        X_row = scaler.transform(X_row)
    prob  = float(np.clip(model.predict_proba(X_row)[0][1], 0, 1))
    score = int(prob * 100)
    if score <= 30:
        cat, color, rec, alert = "Low Risk",    "#22c55e", "Transaction appears normal. Approve and monitor routinely.", "success"
    elif score <= 70:
        cat, color, rec, alert = "Medium Risk", "#f59e0b", "Flagged for review. Consider step-up authentication.", "warning"
    else:
        cat, color, rec, alert = "High Risk",   "#ef4444", "HIGH FRAUD PROBABILITY — Block and notify customer immediately.", "danger"
    return {'probability':prob, 'score':score, 'category':cat,
            'color':color, 'rec':rec, 'alert':alert}

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='padding:20px 0 28px;text-align:center;'>
        <div style='font-family:Space Mono,monospace;font-size:22px;color:#4a9aff;font-weight:700;letter-spacing:2px;'>
            🛡️ FraudShield
        </div>
        <div style='font-size:11px;color:#3a5a8a;letter-spacing:3px;text-transform:uppercase;margin-top:4px;'>
            AI Detection System
        </div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio("Navigate",
        ["🏠  Dashboard","🤖  Train Models","🔍  Predict Transaction","📊  Analytics","📋  Data Explorer"],
        label_visibility="collapsed")

    st.markdown("---")
    model_trained = "models" in st.session_state
    st.markdown(f"""
    <div style='font-size:11px;color:#3a5a8a;padding:8px 0;'>
        <div style='font-family:Space Mono,monospace;color:#2a4a7a;margin-bottom:8px;'>SYSTEM STATUS</div>
        <div>● Dataset loaded</div>
    </div>
    <div style='font-size:11px;color:{"#22c55e" if model_trained else "#f59e0b"};'>
        ● {"Models ready" if model_trained else "Models not trained"}
    </div>
    """, unsafe_allow_html=True)

page_key = page.split("  ")[1]

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
if page_key == "Dashboard":
    df = load_data()
    st.markdown("""
    <div style='margin-bottom:2rem;'>
        <div style='font-family:Space Mono,monospace;font-size:28px;color:#e8f0ff;font-weight:700;'>Transaction Overview</div>
        <div style='color:#3a6aaa;font-size:14px;margin-top:4px;'>Real-time fraud intelligence dashboard</div>
    </div>""", unsafe_allow_html=True)

    total = len(df); fraud_count = df['IsFraud'].sum(); legit_count = total - fraud_count
    fraud_rate = fraud_count/total*100
    avg_amount = df['TransactionAmount'].mean()
    fraud_amount = df[df['IsFraud']==1]['TransactionAmount'].mean()

    c1,c2,c3,c4,c5 = st.columns(5)
    for col,label,value,sub in [
        (c1,"TOTAL TXN",f"{total:,}","all transactions"),
        (c2,"FRAUD CASES",f"{fraud_count:,}","flagged"),
        (c3,"LEGIT CASES",f"{legit_count:,}","approved"),
        (c4,"FRAUD RATE",f"{fraud_rate:.2f}%","of total volume"),
        (c5,"AVG FRAUD AMT",f"${fraud_amount:,.0f}",f"vs ${avg_amount:,.0f} avg"),
    ]:
        col.markdown(f"<div class='metric-card'><div class='metric-label'>{label}</div><div class='metric-value'>{value}</div><div class='metric-sub'>{sub}</div></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1.6])
    with col1:
        st.markdown("<div class='section-header'>Class Distribution</div>", unsafe_allow_html=True)
        fig = go.Figure(go.Pie(labels=['Legitimate','Fraud'], values=[legit_count,fraud_count],
                               hole=0.65, marker_colors=['#1a6a3a','#8b1a1a'],
                               textinfo='percent', textfont_size=13))
        fig.add_annotation(text=f"{fraud_rate:.1f}%", x=0.5, y=0.55, font_size=24,
                           font=dict(color='#ef4444', family='Space Mono'), showarrow=False)
        fig.add_annotation(text="fraud rate", x=0.5, y=0.42, font_size=11,
                           font=dict(color='#5a7ab0'), showarrow=False)
        fig.update_layout(**dark_layout(), height=280, margin=dict(t=10,b=10,l=10,r=10),
                          showlegend=True, legend=dict(font=dict(color='#8ab4d8'),bgcolor='rgba(0,0,0,0)'))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("<div class='section-header'>Transaction Amount Distribution</div>", unsafe_allow_html=True)
        fig2 = go.Figure()
        for label,val,color in [('Legitimate',0,'#22c55e'),('Fraud',1,'#ef4444')]:
            sub = df[df['IsFraud']==val]['TransactionAmount']
            fig2.add_trace(go.Histogram(x=np.log1p(sub), name=label, opacity=0.7,
                                        marker_color=color, nbinsx=60, histnorm='density'))
        fig2.update_layout(**dark_layout(), height=280, margin=dict(t=10,b=30,l=40,r=10),
                           barmode='overlay', xaxis_title="log(Amount)", yaxis_title="Density",
                           legend=dict(font=dict(color='#8ab4d8'),bgcolor='rgba(0,0,0,0)'))
        st.plotly_chart(fig2, use_container_width=True)

    col3,col4 = st.columns(2)
    with col3:
        st.markdown("<div class='section-header'>Fraud by Hour of Day</div>", unsafe_allow_html=True)
        hourly = df.groupby('TransactionHour')['IsFraud'].mean().reset_index()
        hourly['FraudRate'] = hourly['IsFraud']*100
        fig3 = go.Figure(go.Bar(x=hourly['TransactionHour'], y=hourly['FraudRate'],
                                marker=dict(color=hourly['FraudRate'],
                                            colorscale=[[0,'#1a3a6a'],[0.5,'#2a6aaa'],[1,'#ef4444']],
                                            showscale=False)))
        fig3.update_layout(**dark_layout(), height=260, margin=dict(t=10,b=30,l=40,r=10),
                           xaxis_title="Hour", yaxis_title="Fraud Rate (%)")
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        st.markdown("<div class='section-header'>Fraud Rate by Payment Method</div>", unsafe_allow_html=True)
        pm = df.groupby('PaymentMethod')['IsFraud'].mean().reset_index().sort_values('IsFraud')
        pm['FraudRate'] = pm['IsFraud']*100
        fig4 = go.Figure(go.Bar(x=pm['FraudRate'], y=pm['PaymentMethod'], orientation='h',
                                marker=dict(color=pm['FraudRate'],
                                            colorscale=[[0,'#1a6a3a'],[0.5,'#aaaa2a'],[1,'#8b1a1a']],
                                            showscale=False),
                                text=[f"{v:.1f}%" for v in pm['FraudRate']],
                                textposition='outside', textfont=dict(color='#8ab4d8')))
        fig4.update_layout(**dark_layout(), height=260, margin=dict(t=10,b=10,l=10,r=60),
                           xaxis_title="Fraud Rate (%)")
        st.plotly_chart(fig4, use_container_width=True)

    col5,col6 = st.columns(2)
    with col5:
        st.markdown("<div class='section-header'>Fraud Rate by Location</div>", unsafe_allow_html=True)
        loc = df.groupby('UserLocation')['IsFraud'].mean().reset_index().sort_values('IsFraud',ascending=False)
        loc['FraudRate'] = loc['IsFraud']*100
        fig5 = px.bar(loc, x='UserLocation', y='FraudRate', color='FraudRate',
                      color_continuous_scale=['#1a3a6a','#2a8ada','#ef4444'])
        fig5.update_layout(**dark_layout(), height=260, margin=dict(t=10,b=30,l=40,r=10), coloraxis_showscale=False)
        st.plotly_chart(fig5, use_container_width=True)

    with col6:
        st.markdown("<div class='section-header'>Account Age vs Fraud</div>", unsafe_allow_html=True)
        fig6 = go.Figure()
        for label,val,color in [('Legitimate',0,'#22c55e'),('Fraud',1,'#ef4444')]:
            sub = df[df['IsFraud']==val]['AccountAge']
            fig6.add_trace(go.Violin(y=sub, name=label, fillcolor=color, line_color=color,
                                     opacity=0.6, box_visible=True, meanline_visible=True))
        fig6.update_layout(**dark_layout(), height=260, margin=dict(t=10,b=10,l=40,r=10),
                           yaxis_title="Account Age (days)",
                           legend=dict(font=dict(color='#8ab4d8'),bgcolor='rgba(0,0,0,0)'))
        st.plotly_chart(fig6, use_container_width=True)

    if "models" not in st.session_state:
        st.markdown("<div class='alert-warning'>⚠️ Models not yet trained. Head to <b>Train Models</b> to build the ML pipeline.</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: TRAIN MODELS
# ═══════════════════════════════════════════════════════════════════════════════
elif page_key == "Train Models":
    st.markdown("""
    <div style='margin-bottom:2rem;'>
        <div style='font-family:Space Mono,monospace;font-size:28px;color:#e8f0ff;font-weight:700;'>Model Training Pipeline</div>
        <div style='color:#3a6aaa;font-size:14px;margin-top:4px;'>Train and compare Logistic Regression · Random Forest · Gradient Boosting</div>
    </div>""", unsafe_allow_html=True)

    col_btn, col_info = st.columns([1,3])
    with col_btn:
        train_btn = st.button("🚀  Train All Models", use_container_width=True)
    with col_info:
        st.markdown("<div style='font-size:13px;color:#5a7ab0;padding:10px 0;'>Trains 3 models on SMOTE-balanced data · Stratified 80/20 split · Evaluates on held-out test set</div>", unsafe_allow_html=True)

    if train_btn:
        pb = st.progress(0); st_txt = st.empty()
        def upd(f, msg):
            pb.progress(f)
            st_txt.markdown(f"<div style='color:#4a9aff;font-size:13px;'>⚙ {msg}</div>", unsafe_allow_html=True)
        state = train_all_models(progress_cb=upd)
        st.session_state['models'] = state
        pb.progress(1.0)
        st_txt.markdown(f"<div style='color:#22c55e;font-size:13px;'>✅ Training complete — Best: <b>{state['best_name']}</b> (AUC {state['results'][state['best_name']]['ROC-AUC']:.4f})</div>", unsafe_allow_html=True)

    if "models" not in st.session_state:
        st.markdown("<div class='alert-warning'>Click <b>Train All Models</b> above to start (≈30–60 seconds).</div>", unsafe_allow_html=True)
    else:
        state   = st.session_state['models']
        results = state['results']

        st.markdown("<br><div class='section-header'>Performance Comparison</div>", unsafe_allow_html=True)
        mrows = []
        for name, r in results.items():
            mrows.append({'Model': f"🏆 {name}" if name==state['best_name'] else name,
                          'Accuracy':r['Accuracy'],'Precision':r['Precision'],
                          'Recall':r['Recall'],'F1 Score':r['F1 Score'],'ROC-AUC':r['ROC-AUC']})
        mdf = pd.DataFrame(mrows).set_index('Model')
        def color_best(s):
            return ['background-color:#0a2a1a;color:#4ade80' if v else 'background-color:#0d1526;color:#c8d6f0'
                    for v in (s==s.max())]
        st.dataframe(mdf.style.apply(color_best).format("{:.4f}"), use_container_width=True, height=160)
        st.markdown("<div class='alert-warning'>💡 <b>Why accuracy alone is misleading:</b> A model predicting 'always legitimate' scores 96.5% accuracy — yet catches zero fraud. <b>Recall</b> and <b>ROC-AUC</b> are the true metrics.</div>", unsafe_allow_html=True)

        st.markdown("<br><div class='section-header'>ROC Curves & Model Comparison</div>", unsafe_allow_html=True)
        col1,col2 = st.columns(2)
        colors = {'Logistic Regression':'#4a9aff','Random Forest':'#22c55e','Gradient Boosting':'#f59e0b'}

        with col1:
            fig_roc = go.Figure()
            for name, r in results.items():
                fpr,tpr,_ = r['roc']
                fig_roc.add_trace(go.Scatter(x=fpr,y=tpr,mode='lines',
                                             name=f"{name} ({r['ROC-AUC']:.3f})",
                                             line=dict(color=colors[name],width=2.5)))
            fig_roc.add_trace(go.Scatter(x=[0,1],y=[0,1],mode='lines',
                                         line=dict(color='#3a5a8a',dash='dash',width=1),showlegend=False))
            fig_roc.update_layout(**dark_layout(),height=340,margin=dict(t=10,b=40,l=50,r=10),
                                  xaxis_title="False Positive Rate",yaxis_title="True Positive Rate",
                                  legend=dict(font=dict(color='#8ab4d8'),bgcolor='rgba(0,0,0,0)',x=0.55,y=0.1))
            st.plotly_chart(fig_roc, use_container_width=True)

        with col2:
            mnames = ['Accuracy','Precision','Recall','F1 Score','ROC-AUC']
            fig_bar = go.Figure()
            for name,color in colors.items():
                fig_bar.add_trace(go.Bar(name=name, x=mnames,
                                         y=[results[name][m] for m in mnames],
                                         marker_color=color, opacity=0.85))
            fig_bar.update_layout(**dark_layout(),height=340,barmode='group',
                                  margin=dict(t=10,b=40,l=50,r=10),
                                  yaxis=dict(range=[0.7,1.0]),
                                  legend=dict(font=dict(color='#8ab4d8'),bgcolor='rgba(0,0,0,0)'))
            st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("<div class='section-header'>Confusion Matrices</div>", unsafe_allow_html=True)
        cm_cols = st.columns(3)
        for col,(name,r) in zip(cm_cols, results.items()):
            with col:
                cm = r['cm']
                fig_cm = ff.create_annotated_heatmap(
                    z=cm[::-1], x=['Pred Legit','Pred Fraud'], y=['Act Fraud','Act Legit'],
                    colorscale=[[0,'#0a1020'],[1,'#1a4a8a']],
                    annotation_text=[[str(v) for v in row] for row in cm[::-1]],
                    font_colors=['#e8f0ff'])
                fig_cm.update_layout(**dark_layout(),height=260,margin=dict(t=30,b=10,l=10,r=10),
                                     title=dict(text=name,font=dict(color='#8ab4d8',size=13)))
                st.plotly_chart(fig_cm, use_container_width=True)

        st.markdown("<div class='section-header'>Feature Importance (Random Forest)</div>", unsafe_allow_html=True)
        imp = pd.Series(state['importances']).sort_values(ascending=True).tail(15)
        fig_imp = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation='h',
                                   marker=dict(color=imp.values,
                                               colorscale=[[0,'#1a3a6a'],[0.5,'#2a7ada'],[1,'#4a9aff']],
                                               showscale=False),
                                   text=[f"{v:.3f}" for v in imp.values],
                                   textposition='outside',textfont=dict(color='#8ab4d8')))
        fig_imp.update_layout(**dark_layout(),height=420,margin=dict(t=10,b=10,l=10,r=60),xaxis_title="Importance")
        st.plotly_chart(fig_imp, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICT
# ═══════════════════════════════════════════════════════════════════════════════
elif page_key == "Predict Transaction":
    st.markdown("""
    <div style='margin-bottom:2rem;'>
        <div style='font-family:Space Mono,monospace;font-size:28px;color:#e8f0ff;font-weight:700;'>Transaction Risk Scorer</div>
        <div style='color:#3a6aaa;font-size:14px;margin-top:4px;'>Enter transaction details to get an instant fraud risk assessment</div>
    </div>""", unsafe_allow_html=True)

    if "models" not in st.session_state:
        st.markdown("<div class='alert-warning'>⚠️ Models not trained yet. Go to <b>Train Models</b> first.</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='section-header'>Transaction Details</div>", unsafe_allow_html=True)
        col1,col2,col3 = st.columns(3)
        with col1:
            amount  = st.number_input("Transaction Amount ($)", min_value=0.01, max_value=100000.0, value=250.0, step=10.0)
            hour    = st.slider("Transaction Hour (0-23)", 0, 23, 14)
            dow     = st.selectbox("Day of Week", options=[0,1,2,3,4,5,6],
                                   format_func=lambda x: ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][x])
        with col2:
            payment  = st.selectbox("Payment Method", ['Credit Card','Debit Card','PayPal','Crypto','Bank Transfer'])
            device   = st.selectbox("Device Type", ['Mobile','Desktop','Tablet'])
            location = st.selectbox("User Location", ['US','UK','DE','FR','IN','CN','BR','NG','RU','VN'])
        with col3:
            user_age    = st.number_input("User Age", min_value=18, max_value=100, value=35)
            account_age = st.number_input("Account Age (days)", min_value=0, max_value=10000, value=365)
            tx_freq     = st.number_input("Transaction Frequency (last 7d)", min_value=1, max_value=100, value=3)

        col4,col5 = st.columns(2)
        with col4:
            is_intl    = st.radio("International Transaction?", [0,1], format_func=lambda x: "No" if x==0 else "Yes", horizontal=True)
        with col5:
            prev_fraud = st.radio("Previous Fraud History?", [0,1], format_func=lambda x: "No" if x==0 else "Yes", horizontal=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔍  Analyze Transaction"):
            inp = {'TransactionAmount':amount,'TransactionHour':hour,'DayOfWeek':dow,
                   'UserAge':user_age,'UserLocation':location,'DeviceType':device,
                   'PaymentMethod':payment,'AccountAge':account_age,
                   'TransactionFrequency':tx_freq,'IsInternational':is_intl,
                   'PreviousFraudHistory':prev_fraud}
            result = predict_single(inp, st.session_state['models'])

            st.markdown("---")
            st.markdown("<div class='section-header'>Risk Assessment Report</div>", unsafe_allow_html=True)
            rc1,rc2 = st.columns([1,1.5])

            with rc1:
                score = result['score']; color = result['color']
                fig_g = go.Figure(go.Indicator(
                    mode="gauge+number", value=score,
                    number={'suffix':'/100','font':{'color':color,'size':36,'family':'Space Mono'}},
                    gauge={'axis':{'range':[0,100],'tickcolor':'#3a5a8a','tickfont':{'color':'#3a5a8a','size':10}},
                           'bar':{'color':color,'thickness':0.28},'bgcolor':'#0d1526','borderwidth':0,
                           'steps':[{'range':[0,30],'color':'#0a2a1a'},{'range':[30,70],'color':'#2a1e00'},
                                    {'range':[70,100],'color':'#2a0808'}],
                           'threshold':{'line':{'color':color,'width':3},'thickness':0.75,'value':score}}))
                fig_g.update_layout(**dark_layout(),height=280,margin=dict(t=20,b=10,l=30,r=30))
                st.plotly_chart(fig_g, use_container_width=True)
                badge = 'risk-low' if result['category']=='Low Risk' else ('risk-medium' if result['category']=='Medium Risk' else 'risk-high')
                st.markdown(f"<div style='text-align:center;margin-top:-10px;'><span class='{badge}' style='font-size:16px;padding:10px 24px;'>{result['category']}</span></div>", unsafe_allow_html=True)

            with rc2:
                prob_pct = result['probability']*100
                st.markdown(f"""
                <div style='margin-bottom:2rem;'>
                    <div style='font-family:Space Mono,monospace;font-size:13px;color:#5a7ab0;margin-bottom:8px;'>FRAUD PROBABILITY</div>
                    <div style='font-family:Space Mono,monospace;font-size:42px;color:{color};font-weight:700;'>{prob_pct:.1f}%</div>
                    <div class='score-track' style='margin-top:12px;'>
                        <div style='width:{prob_pct:.1f}%;height:100%;background:{color};border-radius:99px;'></div>
                    </div>
                </div>""", unsafe_allow_html=True)
                for k,v in [('Amount',f"${amount:,.2f}"),('Payment Method',payment),
                             ('International',"Yes" if is_intl else "No"),
                             ('Tx Frequency (7d)',tx_freq),('Account Age',f"{account_age} days"),
                             ('Prev Fraud History',"Yes" if prev_fraud else "No"),
                             ('Model Used',st.session_state['models']['best_name'])]:
                    st.markdown(f"<div style='display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #1a2a4a;font-size:13px;'><span style='color:#5a7ab0;'>{k}</span><span style='color:#c8d6f0;font-weight:500;'>{v}</span></div>", unsafe_allow_html=True)

            icon = "✅" if result['alert']=='success' else ("⚠️" if result['alert']=='warning' else "🚨")
            st.markdown(f"<br><div class='alert-{result['alert']}' style='font-size:15px;'>{icon} <b>Recommendation:</b> {result['rec']}</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════════
elif page_key == "Analytics":
    df = load_data()
    st.markdown("""
    <div style='margin-bottom:2rem;'>
        <div style='font-family:Space Mono,monospace;font-size:28px;color:#e8f0ff;font-weight:700;'>Analytics Deep Dive</div>
        <div style='color:#3a6aaa;font-size:14px;margin-top:4px;'>Explore fraud patterns across all dimensions</div>
    </div>""", unsafe_allow_html=True)

    f1,f2,f3,f4 = st.columns(4)
    with f1: sel_methods = st.multiselect("Payment Method", df['PaymentMethod'].unique().tolist(), default=df['PaymentMethod'].unique().tolist())
    with f2: sel_devices = st.multiselect("Device Type", df['DeviceType'].unique().tolist(), default=df['DeviceType'].unique().tolist())
    with f3: amt_range   = st.slider("Amount Range ($)", float(df['TransactionAmount'].min()), float(df['TransactionAmount'].quantile(0.99)), (0.0, float(df['TransactionAmount'].quantile(0.99))))
    with f4: intl_f      = st.radio("International", ["All","Yes","No"], horizontal=True)

    mask = df['PaymentMethod'].isin(sel_methods) & df['DeviceType'].isin(sel_devices) & df['TransactionAmount'].between(*amt_range)
    if intl_f=="Yes": mask &= df['IsInternational']==1
    if intl_f=="No":  mask &= df['IsInternational']==0
    dff = df[mask]
    st.markdown(f"<div style='font-size:12px;color:#3a5a8a;margin:8px 0 16px;'>Showing {len(dff):,} transactions ({dff['IsFraud'].sum():,} fraud)</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-header'>Fraud Heatmap — Hour × Day of Week</div>", unsafe_allow_html=True)
    hd = dff.groupby(['DayOfWeek','TransactionHour'])['IsFraud'].mean().unstack(fill_value=0)*100
    fig_hm = go.Figure(go.Heatmap(z=hd.values, x=[f"{h:02d}:00" for h in hd.columns],
                                   y=[['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][i] for i in hd.index],
                                   colorscale=[[0,'#0a1020'],[0.3,'#1a3a6a'],[0.7,'#aa4a1a'],[1,'#ef4444']],
                                   text=np.round(hd.values,1), texttemplate="%{text}%", textfont_size=9))
    fig_hm.update_layout(**dark_layout(),height=300,margin=dict(t=10,b=40,l=50,r=10))
    st.plotly_chart(fig_hm, use_container_width=True)

    col1,col2 = st.columns(2)
    with col1:
        st.markdown("<div class='section-header'>Fraud Rate by Account Age Bucket</div>", unsafe_allow_html=True)
        dff2 = dff.copy()
        dff2['AgeBucket'] = pd.cut(dff2['AccountAge'],bins=[0,30,90,180,365,730,3650,10000],
                                   labels=['<30d','30-90d','90-180d','180d-1y','1-2y','2-10y','10y+'])
        ab = dff2.groupby('AgeBucket')['IsFraud'].mean().reset_index()
        ab['FraudRate'] = ab['IsFraud']*100
        fig_ab = go.Figure(go.Bar(x=ab['AgeBucket'].astype(str), y=ab['FraudRate'],
                                   marker=dict(color=ab['FraudRate'],
                                               colorscale=[[0,'#1a6a3a'],[0.5,'#aaaa2a'],[1,'#8b1a1a']],
                                               showscale=False),
                                   text=[f"{v:.1f}%" for v in ab['FraudRate']],
                                   textposition='outside',textfont=dict(color='#8ab4d8')))
        fig_ab.update_layout(**dark_layout(),height=300,margin=dict(t=10,b=40,l=50,r=10),
                             xaxis_title="Account Age",yaxis_title="Fraud Rate (%)")
        st.plotly_chart(fig_ab, use_container_width=True)

    with col2:
        st.markdown("<div class='section-header'>International vs Domestic Fraud by Payment</div>", unsafe_allow_html=True)
        id_ = dff.groupby(['IsInternational','PaymentMethod'])['IsFraud'].mean().reset_index()
        id_['FraudRate'] = id_['IsFraud']*100
        id_['Group'] = id_['IsInternational'].map({0:'Domestic',1:'International'})
        fig_id = px.bar(id_, x='PaymentMethod', y='FraudRate', color='Group', barmode='group',
                        color_discrete_map={'Domestic':'#4a9aff','International':'#ef4444'})
        fig_id.update_layout(**dark_layout(),height=300,margin=dict(t=10,b=40,l=50,r=10),
                             yaxis_title="Fraud Rate (%)",
                             legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#8ab4d8')))
        st.plotly_chart(fig_id, use_container_width=True)

    col3,col4 = st.columns(2)
    with col3:
        st.markdown("<div class='section-header'>Frequency vs Amount</div>", unsafe_allow_html=True)
        smp = dff.sample(min(3000,len(dff)),random_state=42)
        fig_sc = go.Figure()
        for lbl,val,color in [('Legitimate',0,'#22c55e'),('Fraud',1,'#ef4444')]:
            s = smp[smp['IsFraud']==val]
            fig_sc.add_trace(go.Scatter(x=s['TransactionFrequency'],y=np.log1p(s['TransactionAmount']),
                                        mode='markers',name=lbl,marker=dict(color=color,size=4,opacity=0.5)))
        fig_sc.update_layout(**dark_layout(),height=280,margin=dict(t=10,b=40,l=50,r=10),
                             xaxis_title="Tx Frequency (7d)",yaxis_title="log(Amount)",
                             legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#8ab4d8')))
        st.plotly_chart(fig_sc, use_container_width=True)

    with col4:
        st.markdown("<div class='section-header'>Previous Fraud History Impact by Device</div>", unsafe_allow_html=True)
        ph = dff.groupby(['PreviousFraudHistory','DeviceType'])['IsFraud'].mean().reset_index()
        ph['FraudRate'] = ph['IsFraud']*100
        ph['Group'] = ph['PreviousFraudHistory'].map({0:'No History',1:'Has History'})
        fig_ph = px.bar(ph,x='DeviceType',y='FraudRate',color='Group',barmode='group',
                        color_discrete_map={'No History':'#4a9aff','Has History':'#ef4444'})
        fig_ph.update_layout(**dark_layout(),height=280,margin=dict(t=10,b=30,l=50,r=10),
                             yaxis_title="Fraud Rate (%)",
                             legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(color='#8ab4d8')))
        st.plotly_chart(fig_ph, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: DATA EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════
elif page_key == "Data Explorer":
    df = load_data()
    st.markdown("""
    <div style='margin-bottom:2rem;'>
        <div style='font-family:Space Mono,monospace;font-size:28px;color:#e8f0ff;font-weight:700;'>Data Explorer</div>
        <div style='color:#3a6aaa;font-size:14px;margin-top:4px;'>Browse, filter, and export the transaction dataset</div>
    </div>""", unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    for col,label,val in [(c1,"ROWS",f"{len(df):,}"),(c2,"COLUMNS",f"{df.shape[1]}"),
                           (c3,"FRAUD",f"{df['IsFraud'].sum():,}"),(c4,"MISSING",f"{df.isnull().sum().sum()}")]:
        col.markdown(f"<div class='metric-card'><div class='metric-label'>{label}</div><div class='metric-value' style='font-size:1.5rem;'>{val}</div></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    with st.expander("📋 Column Info & Descriptive Stats"):
        tab1,tab2 = st.tabs(["Numeric Stats","Categorical Counts"])
        with tab1:
            st.dataframe(df.describe().T.round(2), use_container_width=True)
        with tab2:
            for col in ['PaymentMethod','DeviceType','UserLocation']:
                st.markdown(f"**{col}**")
                vc = df[col].value_counts().reset_index()
                vc.columns = [col,'Count']
                vc['Fraud Rate %'] = vc[col].map(df.groupby(col)['IsFraud'].mean()*100).round(2)
                st.dataframe(vc, use_container_width=True)

    st.markdown("<div class='section-header'>Browse Transactions</div>", unsafe_allow_html=True)
    f1,f2,f3,f4 = st.columns(4)
    with f1: fraud_f  = st.selectbox("Show",["All","Fraud Only","Legitimate Only"])
    with f2: method_f = st.multiselect("Payment Method",df['PaymentMethod'].unique().tolist(),default=df['PaymentMethod'].unique().tolist())
    with f3: loc_f    = st.multiselect("Location",df['UserLocation'].unique().tolist(),default=df['UserLocation'].unique().tolist())
    with f4: sort_col = st.selectbox("Sort by",['TransactionAmount','AccountAge','TransactionFrequency','UserAge','IsFraud'])

    mask = df['PaymentMethod'].isin(method_f) & df['UserLocation'].isin(loc_f)
    if fraud_f=="Fraud Only":      mask &= df['IsFraud']==1
    if fraud_f=="Legitimate Only": mask &= df['IsFraud']==0
    dff = df[mask].sort_values(sort_col,ascending=False)

    st.markdown(f"<div style='font-size:12px;color:#3a5a8a;margin:8px 0;'>{len(dff):,} transactions shown</div>", unsafe_allow_html=True)

    display_cols = ['TransactionID','TransactionAmount','TransactionHour','PaymentMethod',
                    'UserLocation','DeviceType','AccountAge','TransactionFrequency',
                    'IsInternational','PreviousFraudHistory','IsFraud']
    def hl_fraud(val):
        if val==1: return 'background-color:#2a0808;color:#f87171;'
        return 'background-color:#0a2a1a;color:#4ade80;'
    st.dataframe(dff[display_cols].head(500).style.applymap(hl_fraud,subset=['IsFraud']),
                 use_container_width=True, height=420)

    st.download_button("⬇ Download Filtered CSV", dff.to_csv(index=False).encode(),
                       "filtered_transactions.csv","text/csv")

    st.markdown("<br><div class='section-header'>Data Quality Report</div>", unsafe_allow_html=True)
    quality = pd.DataFrame({'Column':df.columns,'Type':df.dtypes.values,
                             'Nulls':df.isnull().sum().values,'Unique':df.nunique().values})
    st.dataframe(quality, use_container_width=True)
