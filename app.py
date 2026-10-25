"""YouTube Video Performance & Engagement Analysis - Streamlit app
Author: Mohd Faiz Kashmiri (B1_B3)
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             mean_absolute_error, mean_squared_error, r2_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

st.set_page_config(page_title="YouTube Performance Analytics", page_icon="📊", layout="wide")
sns.set_theme(style="whitegrid")

DATA_FILE = "youtube_videos_dataset.csv"
REQUIRED = ["video_id", "title", "channel", "subscribers", "views", "likes", "comments",
            "duration_minutes", "category", "upload_day", "upload_hour"]
NUM_COLS = ["subscribers", "views", "likes", "comments", "duration_minutes", "upload_hour"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
CAT_VARS = ["category", "upload_day", "time_slot", "high_performer", "is_weekend", "viral"]


# ----------------------------------------------------------------------------- data
def make_demo_data(n=55, seed=42):
    """Stand-in data with the same schema, used only when no real file is available."""
    rng = np.random.default_rng(seed)
    cats = ["Education", "Science & Technology", "Howto & Style", "Travel",
            "Comedy", "Sports", "People & Blogs", "Gaming"]
    cat = rng.choice(cats, size=n, p=[.28, .20, .10, .10, .10, .08, .04, .10])
    subs = (rng.lognormal(np.log(150000), 0.55, n) // 1000 * 1000).astype(int)
    views = (subs * rng.lognormal(np.log(5), 0.35, n) // 10000 * 10000).astype(int)
    likes = np.clip(views * rng.normal(0.052, 0.006, n) // 1000 * 1000, 1000, None).astype(int)
    comments = np.clip(views * rng.normal(0.0037, 0.0006, n) // 100 * 100, 100, None).astype(int)
    return pd.DataFrame({
        "video_id": [f"v{i + 1:03d}" for i in range(n)],
        "title": [f"Demo video {i + 1}" for i in range(n)],
        "channel": [f"Channel {rng.integers(1, 20)}" for _ in range(n)],
        "subscribers": subs, "views": views, "likes": likes, "comments": comments,
        "duration_minutes": rng.integers(6, 43, n), "category": cat,
        "upload_day": rng.choice(DAYS, n), "upload_hour": rng.integers(7, 22, n),
    })


def clean_data(raw):
    d = raw.drop_duplicates().copy()
    for c in NUM_COLS:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    for c in ["title", "channel", "category", "upload_day"]:
        d[c] = d[c].astype(str).str.strip()
    return d.dropna().reset_index(drop=True)


def add_features(d):
    d = d.copy()
    d["engagement_rate"] = (d["likes"] + d["comments"]) / d["views"] * 100
    d["like_rate"] = d["likes"] / d["views"] * 100
    d["views_per_subscriber"] = d["views"] / d["subscribers"]
    d["is_weekend"] = d["upload_day"].isin(["Saturday", "Sunday"]).astype(int)
    d["time_slot"] = pd.cut(d["upload_hour"], bins=[-1, 11, 16, 24],
                            labels=["Morning", "Afternoon", "Evening"]).astype(str)
    d["viral"] = (d["views"] > 1_000_000).astype(int)
    d["high_performer"] = (d["views"] > d["views"].median()).astype(int)
    return d


def make_features(d):
    X = d[["subscribers", "duration_minutes", "upload_hour"]].copy()
    X["is_weekend"] = d["is_weekend"].astype(int)
    return pd.concat([X, pd.get_dummies(d["category"], prefix="cat", drop_first=True, dtype=int)], axis=1)


def load_data():
    st.sidebar.title("📊 YouTube Analytics")
    st.sidebar.caption("Mohd Faiz Kashmiri · B1_B3")
    up = st.sidebar.file_uploader("Upload youtube_videos_dataset.csv", type="csv")
    if up is not None:
        raw, source = pd.read_csv(up), "uploaded file"
    elif os.path.exists(DATA_FILE):
        raw, source = pd.read_csv(DATA_FILE), DATA_FILE
    else:
        raw, source = make_demo_data(), "DEMO DATA"
    missing = [c for c in REQUIRED if c not in raw.columns]
    if missing:
        st.error("The dataset is missing required columns: " + ", ".join(missing))
        st.stop()
    d = add_features(clean_data(raw))
    if len(d) < 20:
        st.error("Need at least 20 valid rows to run the analyses.")
        st.stop()
    if source == "DEMO DATA":
        st.warning("No dataset found, so synthetic demo data is shown. Upload your CSV in the sidebar to analyse real data.")
    st.sidebar.success(f"Data source: {source} ({len(d)} videos)")
    return d


def show(fig):
    st.pyplot(fig)
    plt.close(fig)


def rmse(a, b):
    return float(np.sqrt(mean_squared_error(a, b)))


# ----------------------------------------------------------------------------- pages
def page_overview(df):
    st.title("YouTube Video Performance & Engagement Analysis")
    c = st.columns(4)
    c[0].metric("Videos", len(df))
    c[1].metric("Mean views", f"{df['views'].mean():,.0f}")
    c[2].metric("Median views", f"{df['views'].median():,.0f}")
    c[3].metric("Avg engagement rate", f"{df['engagement_rate'].mean():.2f}%")
    st.subheader("Data preview")
    st.dataframe(df.head(15))
    st.subheader("Descriptive statistics")
    st.dataframe(df[NUM_COLS].describe().T.round(2))
    st.subheader("Views by category")
    g = df.groupby("category")["views"].agg(["count", "mean", "median", "std"]).sort_values("mean", ascending=False)
    st.dataframe(g.round(0))
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.barplot(x=g.index, y=g["mean"], ax=ax)
    ax.tick_params(axis="x", rotation=40)
    ax.set_ylabel("Mean views")
    show(fig)


def page_distributions(df):
    st.header("Distributions & outliers")
    var = st.selectbox("Variable", NUM_COLS + ["engagement_rate", "like_rate", "views_per_subscriber"], index=1)
    log = st.checkbox("Log10 scale", value=False)
    s = df[var]
    if log:
        s = np.log10(s[s > 0])
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.histplot(s, bins=15, kde=True, ax=ax)
        ax.set_title(f"Distribution of {var}")
        show(fig)
    with c2:
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.boxplot(y=s, ax=ax)
        ax.set_title("Box plot")
        show(fig)
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    out = df[(s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)]
    c = st.columns(3)
    c[0].metric("Skewness", f"{s.skew():.2f}")
    c[1].metric("Kurtosis", f"{s.kurt():.2f}")
    c[2].metric("Outliers (IQR rule)", len(out))
    if len(out):
        st.dataframe(out[["title", "channel", "category", var]])


def page_correlation(df):
    st.header("Correlation & covariance")
    cols = st.multiselect("Variables", NUM_COLS, default=NUM_COLS)
    if len(cols) < 2:
        st.info("Select at least two variables.")
        return
    method = st.radio("Correlation method", ["pearson", "spearman"], horizontal=True)
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Correlation")
        fig, ax = plt.subplots(figsize=(7, 5.5))
        sns.heatmap(df[cols].corr(method=method), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        show(fig)
    with c2:
        st.subheader("Covariance")
        st.dataframe(df[cols].cov().round(2))
        st.caption("Covariance depends on units; correlation = covariance / (SD_x × SD_y).")
    st.subheader("Scatter plot")
    a, b = st.columns(2)
    x = a.selectbox("X", cols, index=0)
    y = b.selectbox("Y", cols, index=min(1, len(cols) - 1))
    if x != y:
        r, p = stats.pearsonr(df[x], df[y])
        rho, ps = stats.spearmanr(df[x], df[y])
        st.write(f"Pearson r = **{r:.3f}** (p = {p:.2e}) · Spearman ρ = **{rho:.3f}** (p = {ps:.2e}) · "
                 f"Covariance = **{np.cov(df[x], df[y])[0, 1]:,.2f}**")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.scatterplot(data=df, x=x, y=y, hue="category", ax=ax)
    sns.regplot(data=df, x=x, y=y, scatter=False, color="black", ax=ax)
    show(fig)


def page_probability(df):
    st.header("Probability")
    v = df["views"]
    X = st.slider("Threshold X (views)", min_value=int(v.min()), max_value=int(v.max()),
                  value=int(np.clip(1_000_000, v.min(), v.max())), step=10000)
    mu, sd = v.mean(), v.std()
    shape, loc, scale = stats.lognorm.fit(v, floc=0)
    p_emp = float((v > X).mean())
    p_norm = float(1 - stats.norm.cdf(X, mu, sd))
    p_logn = float(1 - stats.lognorm.cdf(X, shape, loc, scale))
    c = st.columns(3)
    c[0].metric("Empirical P(views > X)", f"{p_emp:.2%}")
    c[1].metric("Normal model", f"{p_norm:.2%}")
    c[2].metric("Log-normal model", f"{p_logn:.2%}")
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(v, bins=15, density=True, alpha=0.5, label="Observed")
    xs = np.linspace(v.min(), v.max(), 300)
    ax.plot(xs, stats.norm.pdf(xs, mu, sd), label="Normal")
    ax.plot(xs, stats.lognorm.pdf(xs, shape, loc, scale), label="Log-normal")
    ax.axvline(X, color="red", linestyle="--")
    ax.legend()
    show(fig)
    st.caption("Views are right-skewed, so the normal model usually overstates the chance of exceeding the mean.")

    st.subheader("Conditional probability and Bayes")
    flag = (df["views"] > X).astype(int)
    p_cat = df["category"].value_counts(normalize=True)
    p_given = flag.groupby(df["category"]).mean()
    p_flag = flag.mean()
    tab = pd.DataFrame({"P(category)": p_cat, "P(views > X | category)": p_given})
    if p_flag > 0:
        tab["P(category | views > X) (Bayes)"] = p_given * p_cat / p_flag
    st.dataframe(tab.sort_values("P(views > X | category)", ascending=False).round(3))

    st.subheader("Binomial calculator")
    n = st.slider("Number of videos picked at random (n)", min_value=1, max_value=30, value=10)
    k = st.number_input("At least k videos exceed X views", min_value=0, max_value=n, value=min(3, n))
    ks = np.arange(0, n + 1)
    pmf = stats.binom.pmf(ks, n, p_emp)
    st.write(f"P(X ≥ {int(k)}) = **{1 - stats.binom.cdf(int(k) - 1, n, p_emp):.4f}** · "
             f"Expected count = {n * p_emp:.2f}")
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.bar(ks, pmf)
    ax.set_xlabel("Number of videos above X")
    ax.set_ylabel("Probability")
    show(fig)


def draw_sample(df, method, n, rng):
    n = int(min(n, len(df)))
    if method == "Simple random":
        return df.sample(n=n, random_state=int(rng.integers(1_000_000)))
    if method == "Stratified (by category)":
        return df.groupby("category").sample(frac=n / len(df), random_state=int(rng.integers(1_000_000)))
    step = max(len(df) // n, 1)
    return df.iloc[int(rng.integers(step))::step]


def page_sampling(df):
    st.header("Sampling & Central Limit Theorem")
    pop_mean, pop_sd = df["views"].mean(), df["views"].std(ddof=0)
    method = st.selectbox("Sampling method", ["Simple random", "Stratified (by category)", "Systematic"])
    n = st.slider("Sample size n", min_value=5, max_value=len(df), value=min(30, len(df)))
    seed = st.number_input("Random seed", min_value=0, max_value=9999, value=42)
    rng = np.random.default_rng(int(seed))
    smp = draw_sample(df, method, n, rng)
    c = st.columns(3)
    c[0].metric("Population mean", f"{pop_mean:,.0f}")
    c[1].metric("Sample mean", f"{smp['views'].mean():,.0f}", f"{smp['views'].mean() - pop_mean:,.0f}")
    c[2].metric("Sample size drawn", len(smp))
    with st.expander("Show sampled videos"):
        st.dataframe(smp[["video_id", "title", "category", "views"]])

    st.subheader("Sampling distribution of the mean")
    reps = st.slider("Number of repeated samples", min_value=200, max_value=3000, value=1000, step=100)
    n_clt = st.slider("Sample size for repeated sampling", min_value=2, max_value=100, value=30)
    means = np.array([df["views"].sample(n=n_clt, replace=True, random_state=int(rng.integers(1_000_000))).mean()
                      for _ in range(reps)])
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(means, bins=30, density=True, alpha=0.6)
    xs = np.linspace(means.min(), means.max(), 200)
    ax.plot(xs, stats.norm.pdf(xs, pop_mean, pop_sd / np.sqrt(n_clt)), color="red", label="Normal from CLT")
    ax.set_xlabel("Sample mean views")
    ax.legend()
    show(fig)
    st.write(f"Mean of sample means = **{means.mean():,.0f}** · empirical SE = **{means.std():,.0f}** · "
             f"theoretical SE (σ/√n) = **{pop_sd / np.sqrt(n_clt):,.0f}**")

    st.subheader("Confidence interval")
    conf = st.slider("Confidence level (%)", min_value=80, max_value=99, value=95) / 100
    if len(smp) > 1:
        lo, hi = stats.t.interval(conf, df=len(smp) - 1, loc=smp["views"].mean(), scale=stats.sem(smp["views"]))
        inside = lo <= pop_mean <= hi
        st.write(f"{conf:.0%} t-interval for mean views: **[{lo:,.0f}, {hi:,.0f}]** — "
                 f"{'contains' if inside else 'does not contain'} the population mean.")


def page_hypothesis(df):
    st.header("Hypothesis testing")
    alpha = st.select_slider("Significance level α", options=[0.01, 0.05, 0.10], value=0.05)
    test = st.selectbox("Test", ["One-sample z-test", "One-sample t-test", "Two-sample t-test (Welch)",
                                 "Two-sample z-test", "One-way ANOVA", "Chi-square independence",
                                 "Chi-square goodness of fit"])
    numeric_vars = ["views", "likes", "comments", "subscribers", "duration_minutes", "engagement_rate"]
    stat = p = None
    h0 = ""

    if test in ("One-sample z-test", "One-sample t-test"):
        var = st.selectbox("Variable", numeric_vars)
        mu0 = st.number_input("Hypothesised mean μ0", value=float(round(df[var].mean(), 2)) if var != "views" else 1_000_000.0)
        alt = st.selectbox("Alternative", ["two-sided", "greater", "less"])
        x = df[var]
        h0 = f"μ({var}) = {mu0:,.2f}"
        if test == "One-sample z-test":
            sigma = x.std(ddof=0)
            stat = (x.mean() - mu0) / (sigma / np.sqrt(len(x)))
            p = {"two-sided": 2 * stats.norm.sf(abs(stat)), "greater": stats.norm.sf(stat), "less": stats.norm.cdf(stat)}[alt]
            st.caption("σ is taken as the dataset's population standard deviation (treated as known).")
        else:
            stat, p = stats.ttest_1samp(x, mu0, alternative=alt)
        st.write(f"Sample mean = {x.mean():,.2f}, n = {len(x)}")

    elif test in ("Two-sample t-test (Welch)", "Two-sample z-test"):
        var = st.selectbox("Variable", numeric_vars)
        mode = st.radio("Groups", ["Weekend vs weekday", "Two categories"], horizontal=True)
        if mode == "Weekend vs weekday":
            a, b = df.loc[df["is_weekend"] == 1, var], df.loc[df["is_weekend"] == 0, var]
            h0 = f"mean {var} (weekend) = mean {var} (weekday)"
        else:
            cats = sorted(df["category"].unique())
            c1, c2 = st.columns(2)
            ca = c1.selectbox("Group A", cats, index=0)
            cb = c2.selectbox("Group B", cats, index=min(1, len(cats) - 1))
            a, b = df.loc[df["category"] == ca, var], df.loc[df["category"] == cb, var]
            h0 = f"mean {var} ({ca}) = mean {var} ({cb})"
        if len(a) < 2 or len(b) < 2:
            st.warning("Each group needs at least 2 videos.")
            return
        st.write(f"Group A: n = {len(a)}, mean = {a.mean():,.2f} · Group B: n = {len(b)}, mean = {b.mean():,.2f}")
        if test.startswith("Two-sample t"):
            stat, p = stats.ttest_ind(a, b, equal_var=False)
        else:
            stat = (a.mean() - b.mean()) / np.sqrt(a.var() / len(a) + b.var() / len(b))
            p = 2 * stats.norm.sf(abs(stat))

    elif test == "One-way ANOVA":
        var = st.selectbox("Variable", numeric_vars)
        groups = [g[var].values for _, g in df.groupby("category") if len(g) >= 2]
        if len(groups) < 2:
            st.warning("Need at least two categories with 2+ videos.")
            return
        stat, p = stats.f_oneway(*groups)
        h0 = f"all category means of {var} are equal"
        fig, ax = plt.subplots(figsize=(9, 4))
        sns.boxplot(data=df, x="category", y=var, ax=ax)
        ax.tick_params(axis="x", rotation=40)
        show(fig)

    elif test == "Chi-square independence":
        c1, c2 = st.columns(2)
        va = c1.selectbox("Variable 1", CAT_VARS, index=0)
        vb = c2.selectbox("Variable 2", CAT_VARS, index=3)
        if va == vb:
            st.info("Pick two different variables.")
            return
        ct = pd.crosstab(df[va], df[vb])
        if min(ct.shape) < 2:
            st.warning("Each variable needs at least two levels.")
            return
        stat, p, dof, exp = stats.chi2_contingency(ct)
        v = np.sqrt(stat / (ct.values.sum() * (min(ct.shape) - 1)))
        h0 = f"{va} and {vb} are independent"
        st.dataframe(ct)
        st.write(f"dof = {dof} · Cramér's V = {v:.3f}")
        if (exp < 5).mean() > 0.2:
            st.warning("More than 20% of cells have expected counts below 5, so the chi-square approximation is unreliable.")

    else:  # goodness of fit
        var = st.selectbox("Variable", ["upload_day", "category", "time_slot"])
        counts = df[var].value_counts()
        if var == "upload_day":
            counts = counts.reindex(DAYS).fillna(0)
        stat, p = stats.chisquare(counts)
        h0 = f"{var} categories are equally likely"
        st.dataframe(counts.to_frame("observed"))
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.bar(counts.index.astype(str), counts.values)
        ax.axhline(counts.mean(), color="red", linestyle="--", label="Expected (uniform)")
        ax.tick_params(axis="x", rotation=40)
        ax.legend()
        show(fig)

    st.markdown(f"**H0:** {h0}")
    c = st.columns(3)
    c[0].metric("Test statistic", f"{stat:.4f}")
    c[1].metric("p-value", f"{p:.4f}")
    c[2].metric("Decision", "Reject H0" if p < alpha else "Fail to reject H0")
    if p < alpha:
        st.success(f"p = {p:.4f} < α = {alpha}: evidence against H0.")
    else:
        st.info(f"p = {p:.4f} ≥ α = {alpha}: not enough evidence to reject H0.")


def page_regression(df):
    st.header("Regression")
    tab1, tab2 = st.tabs(["Simple linear", "Multiple linear"])
    with tab1:
        x = st.selectbox("Predictor", ["subscribers", "duration_minutes", "upload_hour", "likes", "comments"])
        res = stats.linregress(df[x], df["views"])
        c = st.columns(3)
        c[0].metric("R²", f"{res.rvalue ** 2:.3f}")
        c[1].metric("Slope", f"{res.slope:,.3f}")
        c[2].metric("Slope p-value", f"{res.pvalue:.2e}")
        st.write(f"views = {res.intercept:,.0f} + {res.slope:,.3f} × {x}")
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.regplot(data=df, x=x, y="views", ax=ax)
            show(fig)
        with c2:
            fitted = res.intercept + res.slope * df[x]
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.scatter(fitted, df["views"] - fitted)
            ax.axhline(0, color="red", linestyle="--")
            ax.set_xlabel("Fitted")
            ax.set_ylabel("Residual")
            show(fig)
    with tab2:
        st.caption("Predictors: subscribers, duration, upload hour, weekend flag, category. "
                   "Likes and comments are excluded because they are consequences of views.")
        test_size = st.slider("Test size", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
        use_log = st.checkbox("Model log(views)", value=False)
        X, y = make_features(df), df["views"]
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=42)
        model = LinearRegression().fit(Xtr, np.log(ytr) if use_log else ytr)
        pred = model.predict(Xte)
        pred = np.exp(pred) if use_log else pred
        c = st.columns(3)
        c[0].metric("Test R²", f"{r2_score(yte, pred):.3f}")
        c[1].metric("RMSE", f"{rmse(yte, pred):,.0f}")
        c[2].metric("MAE", f"{mean_absolute_error(yte, pred):,.0f}")
        st.dataframe(pd.DataFrame({"feature": X.columns, "coefficient": model.coef_}).round(4))
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.scatter(yte, pred)
        lims = [min(yte.min(), pred.min()), max(yte.max(), pred.max())]
        ax.plot(lims, lims, "r--")
        ax.set_xlabel("Actual views")
        ax.set_ylabel("Predicted views")
        show(fig)


def page_ml(df):
    st.header("KNN & Decision Tree: predict high-performing videos")
    st.caption("Target: views above the median. Features: subscribers, duration, upload hour, weekend flag, category.")
    c1, c2, c3 = st.columns(3)
    k = c1.slider("KNN: k", min_value=1, max_value=15, value=5)
    depth = c2.slider("Tree: max depth", min_value=1, max_value=8, value=3)
    test_size = c3.slider("Test size", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
    X, y = make_features(df), df["high_performer"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    knn = make_pipeline(StandardScaler(), KNeighborsClassifier(k)).fit(Xtr, ytr)
    tree = DecisionTreeClassifier(max_depth=depth, random_state=42).fit(Xtr, ytr)
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    labels = ["Below median", "Above median"]

    col_a, col_b = st.columns(2)
    for col, name, mdl in [(col_a, "KNN", knn), (col_b, "Decision tree", tree)]:
        pred = mdl.predict(Xte)
        with col:
            st.subheader(name)
            m1, m2 = st.columns(2)
            m1.metric("Test accuracy", f"{accuracy_score(yte, pred):.3f}")
            m2.metric("5-fold CV accuracy", f"{cross_val_score(mdl, X, y, cv=skf).mean():.3f}")
            fig, ax = plt.subplots(figsize=(4.5, 3.5))
            sns.heatmap(confusion_matrix(yte, pred, labels=[0, 1]), annot=True, fmt="d", cmap="Blues",
                        xticklabels=labels, yticklabels=labels, ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            show(fig)
            rep = classification_report(yte, pred, labels=[0, 1], target_names=labels, output_dict=True, zero_division=0)
            st.dataframe(pd.DataFrame(rep).T.round(3))

    st.subheader("How accuracy changes with k")
    ks = list(range(1, 16))
    accs = [cross_val_score(make_pipeline(StandardScaler(), KNeighborsClassifier(i)), X, y, cv=skf).mean() for i in ks]
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(ks, accs, marker="o")
    ax.axvline(k, color="red", linestyle="--")
    ax.set_xlabel("k")
    ax.set_ylabel("CV accuracy")
    show(fig)

    st.subheader("Decision tree structure")
    fig, ax = plt.subplots(figsize=(16, 7))
    plot_tree(tree, feature_names=list(X.columns), class_names=labels, filled=True, rounded=True, fontsize=8, ax=ax)
    show(fig)
    imp = pd.Series(tree.feature_importances_, index=X.columns).sort_values()
    imp = imp[imp > 0]
    if len(imp):
        fig, ax = plt.subplots(figsize=(7, 3.5))
        imp.plot.barh(ax=ax)
        ax.set_title("Feature importance")
        show(fig)


def page_predict(df):
    st.header("Predict a video's performance")
    c = st.columns(3)
    subs = c[0].number_input("Channel subscribers", min_value=1000, max_value=5_000_000, value=150_000, step=5000)
    dur = c[1].number_input("Duration (minutes)", min_value=1, max_value=180, value=15)
    hour = c[2].slider("Upload hour (0-23)", min_value=0, max_value=23, value=18)
    c = st.columns(2)
    day = c[0].selectbox("Upload day", DAYS, index=0)
    cat = c[1].selectbox("Category", sorted(df["category"].unique()))

    X, y = make_features(df), df["high_performer"]
    row = pd.DataFrame([{"subscribers": subs, "duration_minutes": dur, "upload_hour": hour,
                         "is_weekend": int(day in ("Saturday", "Sunday"))}])
    for col in X.columns:
        if col.startswith("cat_"):
            row[col] = int(col == f"cat_{cat}")
    row = row[X.columns]

    lin = LinearRegression().fit(X, np.log(df["views"]))
    knn = make_pipeline(StandardScaler(), KNeighborsClassifier(5)).fit(X, y)
    tree = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X, y)
    est = float(np.exp(lin.predict(row)[0]))
    p_knn = float(knn.predict_proba(row)[0][1])
    p_tree = float(tree.predict_proba(row)[0][1])

    m = st.columns(3)
    m[0].metric("Estimated views (log-linear model)", f"{est:,.0f}")
    m[1].metric("KNN: P(above median)", f"{p_knn:.0%}")
    m[2].metric("Tree: P(above median)", f"{p_tree:.0%}")
    st.caption(f"Median views in the data: {df['views'].median():,.0f}. These are statistical estimates from a small "
               "dataset and are not guarantees.")


PAGES = {
    "Overview": page_overview,
    "Distributions": page_distributions,
    "Correlation & Covariance": page_correlation,
    "Probability": page_probability,
    "Sampling & CLT": page_sampling,
    "Hypothesis Tests": page_hypothesis,
    "Regression": page_regression,
    "KNN & Decision Tree": page_ml,
    "Predict": page_predict,
}


def main():
    df = load_data()
    page = st.sidebar.radio("Navigate", list(PAGES.keys()))
    PAGES[page](df)


if __name__ == "__main__":
    main()
