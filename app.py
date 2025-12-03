import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, r2_score, silhouette_score
from sklearn.impute import SimpleImputer

# Algorithms
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN

# --- Streamlit UI ---
st.title("General ML Model Runner")

uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])
task_type = st.selectbox("Select Task Type", ["Classification", "Regression", "Clustering"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # --- Target selection for supervised tasks ---
    target_col = None
    feature_cols = []

    if task_type in ["Classification", "Regression"]:
        target_col = st.selectbox("Select Target Column", df.columns)

        # Checkbox: use all other columns as features
        use_all_features = st.checkbox("Use all columns except target as features", value=True)

        if use_all_features:
            feature_cols = [c for c in df.columns if c != target_col]
        else:
            feature_cols = st.multiselect("Select Feature Columns", [c for c in df.columns if c != target_col])

    else:
        feature_cols = df.columns.tolist()

    # --- Preprocessing with Imputers ---
    categorical_features = df[feature_cols].select_dtypes(include=['object']).columns.tolist()
    numerical_features = df[feature_cols].select_dtypes(exclude=['object']).columns.tolist()

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),   # fill NaN with most frequent category
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),            # fill NaN with mean value
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_features),
            ('num', numeric_transformer, numerical_features)
        ]
    )

    # --- Train/Test Split for supervised tasks ---
    if task_type in ["Classification", "Regression"]:
        X = df[feature_cols]
        y = df[target_col]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y if task_type=="Classification" else None
        )

    st.subheader("Results")

    # --- Run models ---
    results = {}

    if task_type == "Classification":
        models = {
            "Logistic Regression": LogisticRegression(max_iter=500),
            "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
            "SVM": SVC(probability=True)
        }
        for name, clf in models.items():
            pipe = Pipeline([("preprocessor", preprocessor), ("model", clf)])
            pipe.fit(X_train, y_train)
            preds = pipe.predict(X_test)
            acc = accuracy_score(y_test, preds)
            results[name] = acc

    elif task_type == "Regression":
        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest Regressor": RandomForestRegressor(n_estimators=200, random_state=42),
            "SVR": SVR()
        }
        for name, reg in models.items():
            pipe = Pipeline([("preprocessor", preprocessor), ("model", reg)])
            pipe.fit(X_train, y_train)
            preds = pipe.predict(X_test)
            score = r2_score(y_test, preds)
            results[name] = score

    elif task_type == "Clustering":
        X = df[feature_cols]
        pipe = Pipeline([("preprocessor", preprocessor)])
        X_proc = pipe.fit_transform(X)

        models = {
            "KMeans": KMeans(n_clusters=3, random_state=42),
            "Agglomerative": AgglomerativeClustering(n_clusters=3),
            "DBSCAN": DBSCAN(eps=0.5, min_samples=5)
        }
        for name, clu in models.items():
            labels = clu.fit_predict(X_proc)
            if len(set(labels)) > 1:
                score = silhouette_score(X_proc, labels)
                results[name] = score
            else:
                results[name] = "Only one cluster formed"

    # --- Show results ---
    st.write("### Accuracy Report")
    for model, score in results.items():
        st.write(f"{model}: {score}")
