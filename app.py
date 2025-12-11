import gradio as gr
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import SelectKBest, chi2, f_classif, RFE
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import io

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

def plot_bar(df, value_col, top_n=15):
    fig, ax = plt.subplots(figsize=(8, 4))
    df_head = df.head(top_n)
    ax.bar(df_head.iloc[:, 0].astype(str), df_head[value_col])
    ax.set_xticklabels(df_head.iloc[:, 0].astype(str), rotation=60, ha="right", fontsize=8)
    ax.set_ylabel(value_col)
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    img = plt.imread(buf, format="png")
    plt.close(fig)
    return img

def run_filter(method, k):
    selector = SelectKBest(chi2 if method == "Chi-square" else f_classif, k=k)
    selector.fit(X, y)
    scores = selector.scores_
    df = pd.DataFrame({"Feature": X.columns, "Score": scores}).sort_values(by="Score", ascending=False).reset_index(drop=True)
    img = plot_bar(df, "Score")
    return df, img

def run_wrapper(model_choice, n_select):
    model = LogisticRegression(max_iter=1000, solver="liblinear") if model_choice == "Logistic Regression" else RandomForestClassifier(n_estimators=100)
    rfe = RFE(model, n_features_to_select=n_select)
    rfe.fit(X, y)
    df = pd.DataFrame({"Feature": X.columns, "Ranking": rfe.ranking_, "Selected": rfe.support_}).sort_values(by="Ranking").reset_index(drop=True)
    return df

def run_embedded(embed_method, lasso_alpha):
    if embed_method == "Lasso Regression":
        lasso = Lasso(alpha=lasso_alpha, max_iter=10000)
        lasso.fit(X, y)
        coeffs = lasso.coef_
        df = pd.DataFrame({"Feature": X.columns, "Coefficient": coeffs}).sort_values(by="Coefficient", ascending=False).reset_index(drop=True)
        img = plot_bar(df, "Coefficient")
        return df, img
    else:
        rf = RandomForestClassifier(n_estimators=200)
        rf.fit(X, y)
        importances = rf.feature_importances_
        df = pd.DataFrame({"Feature": X.columns, "Importance": importances}).sort_values(by="Importance", ascending=False).reset_index(drop=True)
        img = plot_bar(df, "Importance")
        return df, img

with gr.Blocks() as demo:
    gr.Markdown("# Feature Selection Dashboard (Filter, Wrapper, Embedded)")
    gr.Markdown("Submitted by: Arafath Ali")

    with gr.Tab("Filter Methods"):
        with gr.Row():
            with gr.Column(scale=1):
                method = gr.Radio(choices=["Chi-square", "ANOVA F-test"], value="ANOVA F-test", label="Filter method")
                k = gr.Slider(minimum=1, maximum=X.shape[1], step=1, value=5, label="Select K features")
                run_f = gr.Button("Run Filter")
            with gr.Column(scale=2):
                out_table_f = gr.Dataframe(headers=["Feature", "Score"], interactive=False)
                out_plot_f = gr.Image(type="numpy", label="Feature Importance Plot")
        run_f.click(fn=run_filter, inputs=[method, k], outputs=[out_table_f, out_plot_f])

    with gr.Tab("Wrapper Methods"):
        with gr.Row():
            with gr.Column(scale=1):
                model_choice = gr.Radio(choices=["Logistic Regression", "Random Forest"], value="Logistic Regression", label="Model for RFE")
                n_select = gr.Slider(minimum=1, maximum=X.shape[1], step=1, value=5, label="Number of features to select (RFE)")
                run_w = gr.Button("Run Wrapper (RFE)")
            with gr.Column(scale=2):
                out_table_w = gr.Dataframe(headers=["Feature", "Ranking", "Selected"], interactive=False)
        run_w.click(fn=run_wrapper, inputs=[model_choice, n_select], outputs=[out_table_w])

    with gr.Tab("Embedded Methods"):
        with gr.Row():
            with gr.Column(scale=1):
                embed_method = gr.Radio(choices=["Lasso Regression", "Random Forest Importance"], value="Random Forest Importance", label="Embedded method")
                lasso_alpha = gr.Slider(minimum=0.001, maximum=1.0, step=0.001, value=0.01, label="Lasso alpha (only if Lasso chosen)")
                run_e = gr.Button("Run Embedded")
            with gr.Column(scale=2):
                out_table_e = gr.Dataframe(headers=["Feature", "Coefficient/Importance"], interactive=False)
                out_plot_e = gr.Image(type="numpy", label="Embedded Feature Plot")

        def wrapped_embedded(embed_method, lasso_alpha):
            df, img = run_embedded(embed_method, lasso_alpha)
            if "Coefficient" in df.columns:
                df = df.rename(columns={"Coefficient": "Coefficient/Importance"})
            elif "Importance" in df.columns:
                df = df.rename(columns={"Importance": "Coefficient/Importance"})
            return df, img

        run_e.click(fn=wrapped_embedded, inputs=[embed_method, lasso_alpha], outputs=[out_table_e, out_plot_e])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, ssr_mode=False)
