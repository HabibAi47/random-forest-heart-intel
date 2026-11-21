"""
Random Forest Lab - Streamlit app (Intel Images + Heart Disease)
Run:  python -m streamlit run app11.py
"""
import os, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

st.set_page_config(page_title="Random Forest Lab", page_icon="🌲", layout="wide")
SIZE, SEED = 32, 42

st.markdown("""
<style>
.block-container {padding-top: 2rem;}
.hero {background: linear-gradient(90deg,#14532d,#16a34a); color: white; padding: 1.2rem 1.6rem;
border-radius: 14px; margin-bottom: 1rem;}
.hero h1 {margin: 0; font-size: 2rem; color: white;}
.hero p {margin: .2rem 0 0 0; opacity: .9;}
</style>
<div class="hero"><h1>🌲 Random Forest Lab</h1>
<p>Question 2 · Part B: trees vs accuracy · Decision Tree vs Random Forest · SVM comparison</p></div>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="Images load ho rahi hain...")
def load_images(folder, limit):
    X, y = [], []
    classes = sorted(c for c in os.listdir(folder) if os.path.isdir(os.path.join(folder, c)))
    for label, cls in enumerate(classes):
        for f in sorted(os.listdir(os.path.join(folder, cls)))[:limit]:
            img = Image.open(os.path.join(folder, cls, f)).convert("RGB").resize((SIZE, SIZE))
            X.append(np.array(img).flatten() / 255.0)
            y.append(label)
    return np.array(X), np.array(y), classes


def make_model(name, n_trees=100, prob=False):
    return {"Decision Tree": DecisionTreeClassifier(random_state=SEED),
            "Random Forest": RandomForestClassifier(n_estimators=n_trees, random_state=SEED, n_jobs=-1),
            "SVM (RBF)": SVC(kernel="rbf", probability=prob, random_state=SEED),
            "SVM (linear)": SVC(kernel="linear", random_state=SEED)}[name]


def fit_timed(model, X, y):
    t = time.time()
    model.fit(X, y)
    return model, time.time() - t


def save_result(key, df):
    st.session_state.setdefault("results", {})[key] = df


def prep_image(img, d):
    x = np.array(img.convert("RGB").resize((SIZE, SIZE))).flatten().reshape(1, -1) / 255.0
    x = d["scaler"].transform(x)
    return d["pca"].transform(x) if d["pca"] is not None else x


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
st.sidebar.header("⚙️ Settings")
dataset = st.sidebar.radio("Dataset", ["Intel Images", "Heart Disease"])

if dataset == "Intel Images":
    train_path = st.sidebar.text_input("seg_train", r"C:\Users\DELL\Downloads\seg_train\seg_train")
    test_path = st.sidebar.text_input("seg_test", r"C:\Users\DELL\Downloads\seg_test\seg_test")
    pred_path = st.sidebar.text_input("seg_pred", r"C:\Users\DELL\Downloads\seg_pred\seg_pred")
    limit = st.sidebar.slider("Train images per class", 50, 1500, 500, step=50)
    use_pca = st.sidebar.checkbox("PCA (95% variance)", value=True)
else:
    csv_path = st.sidebar.text_input("CSV path", r"C:\Users\DELL\Downloads\archive (3)\heart_cleveland_upload.csv")
    csv_up = st.sidebar.file_uploader("...ya CSV upload karein", type="csv")
    test_size = st.sidebar.slider("Test size", 0.1, 0.4, 0.2, step=0.05)

if st.sidebar.button("🚀 Load & Preprocess", type="primary", use_container_width=True):
    try:
        st.session_state.pop("results", None)
        st.session_state.pop("pred_models", None)
        if dataset == "Intel Images":
            Xtr, ytr, classes = load_images(train_path, limit)
            Xte, yte, _ = load_images(test_path, max(limit // 3, 30))
            scaler = StandardScaler().fit(Xtr)
            Xtr_p, Xte_p = scaler.transform(Xtr), scaler.transform(Xte)
            pca = None
            if use_pca:
                pca = PCA(n_components=0.95, random_state=SEED).fit(Xtr_p)
                Xtr_p, Xte_p = pca.transform(Xtr_p), pca.transform(Xte_p)
            d = dict(name=dataset, Xtr=Xtr_p, Xte=Xte_p, ytr=ytr, yte=yte, classes=classes,
            scaler=scaler, pca=pca, train_path=train_path, pred_path=pred_path)
        else:
            df = pd.read_csv(csv_up if csv_up is not None else csv_path)
            df = df.replace("?", np.nan).apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)
            target = next((c for c in ["target", "condition", "num"] if c in df.columns), df.columns[-1])
            y = (df[target] > 0).astype(int).values
            X = df.drop(columns=[target])
            Xtr, Xte, ytr, yte = train_test_split(X.values, y, test_size=test_size, stratify=y, random_state=SEED)
            scaler = StandardScaler().fit(Xtr)
            d = dict(name=dataset, Xtr=scaler.transform(Xtr), Xte=scaler.transform(Xte), ytr=ytr, yte=yte,
                    classes=["Healthy", "Disease"], scaler=scaler, pca=None,
                df=df, target=target, features=list(X.columns))
        st.session_state["data"] = d
        st.sidebar.success(f"Loaded ✔ train {d['Xtr'].shape} | test {d['Xte'].shape}")
    except Exception as e:
        st.sidebar.error(f"Error: {e}")

d = st.session_state.get("data")
if d is None:
    st.info("👈 Sidebar se dataset chunein aur **Load & Preprocess** dabayein.")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Dataset", d["name"])
c2.metric("Train samples", d["Xtr"].shape[0])
c3.metric("Test samples", d["Xte"].shape[0])
c4.metric("Features (after preprocessing)", d["Xtr"].shape[1])

tabs = st.tabs(["📊 Data", "🌳 Exp 1: Trees vs Accuracy", "⚖️ Exp 2: DT vs RF",
                "🏁 Exp 3: SVM Comparison", "🔮 Predict", "📑 Results"])

# ---- Data ------------------------------------------------------------------
with tabs[0]:
    if d["name"] == "Heart Disease":
        df = d["df"]
        st.write(f"**{df.shape[0]} rows × {df.shape[1]} columns** | label column: `{d['target']}`")
        st.dataframe(df.head(10))
        a, b = st.columns(2)
        counts = pd.Series(np.concatenate([d["ytr"], d["yte"]])).map({0: "Healthy", 1: "Disease"}).value_counts()
        a.markdown("**Class balance**"); a.bar_chart(counts)
        fig, ax = plt.subplots(figsize=(5, 4))
        corr = df.corr()
        im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
        ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.columns, fontsize=7)
        fig.colorbar(im); b.markdown("**Correlation heatmap**"); b.pyplot(fig)
    else:
        counts = pd.Series(np.bincount(d["ytr"]), index=d["classes"])
        st.markdown("**Train images per class**"); st.bar_chart(counts)
        st.markdown("**Ek sample image har class se**")
        cols = st.columns(len(d["classes"]))
        for col, cls in zip(cols, d["classes"]):
            folder = os.path.join(d["train_path"], cls)
            col.image(Image.open(os.path.join(folder, sorted(os.listdir(folder))[0])), caption=cls)

# ---- Exp 1 -----------------------------------------------------------------
with tabs[1]:
    trees = st.multiselect("n_estimators values", [1, 5, 10, 25, 50, 100, 200, 300],
                           default=[1, 10, 50, 100, 300])
    repeats = st.slider("Har value kitni baar (alag random seed)", 1, 5, 5 if d["name"] == "Heart Disease" else 1)
    if st.button("▶ Run Experiment 1"):
        rows, bar = [], st.progress(0.0)
        for i, n in enumerate(sorted(trees)):
            accs = []
            for r in range(repeats):
                rf = RandomForestClassifier(n_estimators=n, random_state=SEED + r, n_jobs=-1).fit(d["Xtr"], d["ytr"])
                accs.append(accuracy_score(d["yte"], rf.predict(d["Xte"])))
            rows.append({"n_estimators": n, "mean_accuracy": np.mean(accs), "std": np.std(accs)})
            bar.progress((i + 1) / len(trees))
        save_result("Exp1: accuracy vs trees", pd.DataFrame(rows).round(4))
    if "Exp1: accuracy vs trees" in st.session_state.get("results", {}):
        res = st.session_state["results"]["Exp1: accuracy vs trees"]
        a, b = st.columns(2)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.errorbar(res.n_estimators, res.mean_accuracy, yerr=res["std"], marker="o", capsize=3, color="#16a34a")
        ax.set_xscale("log"); ax.set_xlabel("n_estimators"); ax.set_ylabel("Test accuracy")
        ax.set_title("Accuracy vs Number of Trees"); ax.grid(alpha=.3)
        a.pyplot(fig); b.dataframe(res)

# ---- Exp 2 -----------------------------------------------------------------
with tabs[2]:
    st.caption("Default parameters ke saath Decision Tree vs Random Forest")
    if st.button("▶ Run Experiment 2"):
        rows = []
        for name in ["Decision Tree", "Random Forest"]:
            m, t_fit = fit_timed(make_model(name), d["Xtr"], d["ytr"])
            t = time.time(); p = m.predict(d["Xte"]); t_pred = time.time() - t
            rows.append({"Model": name, "Train accuracy": accuracy_score(d["ytr"], m.predict(d["Xtr"])),
                        "Test accuracy": accuracy_score(d["yte"], p),
                        "Training time (s)": t_fit, "Predict time (s)": t_pred})
        save_result("Exp2: DT vs RF", pd.DataFrame(rows).set_index("Model").round(4))
    if "Exp2: DT vs RF" in st.session_state.get("results", {}):
        res = st.session_state["results"]["Exp2: DT vs RF"]
        st.dataframe(res)
        a, b = st.columns(2)
        a.markdown("**Test accuracy**"); a.bar_chart(res["Test accuracy"])
        b.markdown("**Training time (s)**"); b.bar_chart(res["Training time (s)"])
        gap = res.loc["Decision Tree", "Train accuracy"] - res.loc["Decision Tree", "Test accuracy"]
        st.info(f"Decision Tree ka train-test gap = {gap:.3f} (bara gap = overfitting).")

# ---- Exp 3 -----------------------------------------------------------------
with tabs[3]:
    do_cv = st.checkbox("5-fold cross-validation bhi karein", value=(d["name"] == "Heart Disease"))
    if st.button("▶ Run Experiment 3"):
        rows, preds = [], {}
        cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
        with st.spinner("Models train ho rahe hain..."):
            for name in ["Decision Tree", "Random Forest", "SVM (RBF)", "SVM (linear)"]:
                m, t_fit = fit_timed(make_model(name), d["Xtr"], d["ytr"])
                p = m.predict(d["Xte"]); preds[name] = p
                row = {"Model": name, "Test accuracy": accuracy_score(d["yte"], p), "Training time (s)": t_fit}
                if do_cv:
                    s = cross_val_score(make_model(name), d["Xtr"], d["ytr"], cv=cv, n_jobs=-1)
                    row["CV mean"], row["CV std"] = s.mean(), s.std()
                rows.append(row)
        st.session_state["preds"] = preds
        save_result("Exp3: model comparison", pd.DataFrame(rows).set_index("Model").round(4))
    if "Exp3: model comparison" in st.session_state.get("results", {}):
        res = st.session_state["results"]["Exp3: model comparison"]
        st.dataframe(res)
        best = res["Test accuracy"].idxmax()
        st.success(f"Best test accuracy: **{best}** ({res.loc[best, 'Test accuracy']:.4f})")
        a, b = st.columns(2)
        a.markdown("**Test accuracy**"); a.bar_chart(res["Test accuracy"])
        b.markdown("**Training time (s)**"); b.bar_chart(res["Training time (s)"])
        pick = st.selectbox("Detail: model chunein", list(st.session_state["preds"]))
        p = st.session_state["preds"][pick]
        a, b = st.columns(2)
        a.text(classification_report(d["yte"], p, target_names=d["classes"], zero_division=0))
        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay(confusion_matrix(d["yte"], p), display_labels=d["classes"]).plot(
            ax=ax, colorbar=False, xticks_rotation=45)
        b.pyplot(fig)

# ---- Predict ---------------------------------------------------------------
with tabs[4]:
    model_name = st.selectbox("Model", ["Random Forest", "SVM (RBF)", "Decision Tree"])
    pm = st.session_state.setdefault("pred_models", {})
    if model_name not in pm:
        with st.spinner("Model train ho raha hai..."):
            pm[model_name], _ = fit_timed(make_model(model_name, prob=True), d["Xtr"], d["ytr"])
    model = pm[model_name]

    if d["name"] == "Heart Disease":
        st.caption("Mareez ki maloomat bharein (sirf seekhne ke liye, medical advice nahi).")
        df = d["df"]
        vals, cols = {}, st.columns(3)
        for i, f in enumerate(d["features"]):
            lo, hi, med = float(df[f].min()), float(df[f].max()), float(df[f].median())
            vals[f] = cols[i % 3].number_input(f, min_value=lo, max_value=hi, value=med)
        if st.button("🔮 Predict"):
            x = d["scaler"].transform(pd.DataFrame([vals])[d["features"]].values)
            prob = model.predict_proba(x)[0]
            label = d["classes"][int(np.argmax(prob))]
            (st.error if label == "Disease" else st.success)(f"Prediction: **{label}**")
            st.bar_chart(pd.Series(prob, index=d["classes"]))
    else:
        n_show = st.slider("seg_pred se kitni images", 4, 16, 8, step=4)
        if st.button("🔮 Predict seg_pred images"):
            try:
                files = sorted(os.listdir(d["pred_path"]))
                rng = np.random.RandomState(int(time.time()) % 1000)
                files = list(rng.choice(files, size=min(n_show, len(files)), replace=False))
                cols = st.columns(4)
                for i, f in enumerate(files):
                    img = Image.open(os.path.join(d["pred_path"], f))
                    prob = model.predict_proba(prep_image(img, d))[0]
                    cols[i % 4].image(img, caption=f"{d['classes'][int(np.argmax(prob))]} ({prob.max():.0%})")
            except Exception as e:
                st.error(f"seg_pred path check karein: {e}")
        up = st.file_uploader("Ya apni image upload karein", type=["jpg", "jpeg", "png"])
        if up is not None:
            img = Image.open(up)
            prob = model.predict_proba(prep_image(img, d))[0]
            a, b = st.columns(2)
            a.image(img, caption=f"Prediction: {d['classes'][int(np.argmax(prob))]}", width=300)
            b.bar_chart(pd.Series(prob, index=d["classes"]))

# ---- Results ---------------------------------------------------------------
with tabs[5]:
    results = st.session_state.get("results", {})
    if not results:
        st.info("Abhi koi experiment nahi chala. Pehle Exp 1, 2, 3 chalayein.")
    for key, res in results.items():
        st.markdown(f"**{key}**")
        st.dataframe(res)
        st.download_button(f"⬇ Download CSV: {key}", res.to_csv().encode(),
                                    file_name=f"{d['name'].replace(' ', '_')}_{key[:4]}.csv", key=key)
