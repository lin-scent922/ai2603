import streamlit as st
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# 設定網頁標題與圖示
st.set_page_config(page_title="酒類預測 ML 專題", page_icon="🍷", layout="wide")

# 套用自定義 CSS 美化
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
    }
    .stDataFrame {
        border: 1px solid #e6e9ef;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.title("🍷 模型選單")
model_name = st.sidebar.selectbox(
    "請選擇機器學習模型：",
    ("KNN", "羅吉斯迴歸", "XGBoost", "隨機森林")
)

st.sidebar.divider()
st.sidebar.subheader("📊 資料集資訊")
wine = datasets.load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df['target'] = wine.target

st.sidebar.info(f"""
**資料集名稱：** 酒類 (Wine)
**樣本數：** {df.shape[0]}
**特徵數：** {df.shape[1] - 1}
**類別數：** {len(np.unique(wine.target))} (Class 0, 1, 2)
""")

# --- Main Area ---
st.title("🍷 酒類資料集機器學習預測")
st.write(f"當前選擇的模型： **{model_name}**")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📋 資料集前 5 筆")
    st.dataframe(df.head())

with col2:
    st.subheader("📈 特徵統計資訊")
    st.dataframe(df.describe().T)

st.divider()

# --- ML Logic ---
st.subheader("🚀 模型訓練與預測")

if st.button("開始進行預測並分析準確度"):
    # 準備資料
    X = wine.data
    y = wine.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 模型實例化
    if model_name == "KNN":
        model = KNeighborsClassifier(n_neighbors=5)
    elif model_name == "羅吉斯迴歸":
        model = LogisticRegression(max_iter=5000)
    elif model_name == "XGBoost":
        model = XGBClassifier()
    elif model_name == "隨機森林":
        model = RandomForestClassifier(n_estimators=100)

    # 訓練模型
    with st.spinner("模型訓練中..."):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

    # 顯示結果
    res_col1, res_col2 = st.columns(2)
    with res_col1:
        st.success(f"### 模型準確度 (Accuracy)")
        st.metric(label="Accuracy Score", value=f"{acc:.2%}")
    
    with res_col2:
        st.info("### 預測結果摘要")
        st.write(f"測試集樣本數: {len(y_test)}")
        st.write(f"正確預測數: {int(acc * len(y_test))}")

    # 顯示比對表格
    st.write("#### 預測值 vs 實際值 (前 10 筆)")
    comparison_df = pd.DataFrame({
        '實際標籤': y_test[:10],
        '預測標籤': y_pred[:10]
    })
    st.table(comparison_df.T)

st.sidebar.caption("Made with ❤️ by Antigravity")
