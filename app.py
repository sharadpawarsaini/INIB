import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from PIL import Image

# --- PAGE SETUP ---
st.set_page_config(
    page_title="Advanced KNN Iris Dashboard",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- MODERN STYLING (GLASSMORPHISM / DARK THEME) ---
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Space+Grotesk:wght@300;400;600;700&display=swap');
    
    /* Main Layout Styling */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0d0e15 0%, #151828 100%);
        color: #e2e8f0;
    }
    
    /* Card design */
    .premium-card {
        background: rgba(30, 41, 59, 0.45);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
    }
    
    /* Gradient headers */
    .gradient-text {
        background: linear-gradient(90deg, #a78bfa 0%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
    }
    
    .sub-gradient-text {
        background: linear-gradient(90deg, #ec4899 0%, #f43f5e 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 600;
    }

    /* Metric visualizers */
    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        color: #3b82f6;
        text-shadow: 0 0 10px rgba(59, 130, 246, 0.3);
    }
    
    /* Status banner */
    .status-banner {
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid rgba(16, 185, 129, 0.3);
        color: #34d399;
        border-radius: 8px;
        padding: 12px;
        font-weight: 600;
        text-align: center;
    }
</style>
""", unsafe_allow_value=True)

# --- LOAD DATA ---
@st.cache_data
def get_dataset():
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name="target")
    df = X.copy()
    df['species'] = y.map(dict(enumerate(iris.target_names)))
    return df, X, y, iris.target_names, iris.feature_names

df_full, X_raw, y_raw, target_names, feature_names = get_dataset()

# --- HELPER FUNCTIONS ---
def get_local_model_or_default():
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "iris_knn_pipeline.joblib")
    if os.path.exists(model_path):
        try:
            return joblib.load(model_path)
        except Exception:
            pass
    
    # Fallback default training if no saved model is found
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier(n_neighbors=5, weights='distance'))
    ])
    pipeline.fit(X_raw, y_raw)
    return pipeline

model_pipeline = get_local_model_or_default()

# --- SIDEBAR NAVIGATION ---
st.sidebar.markdown("<h2 class='gradient-text'>🧠 Advanced KNN</h2>", unsafe_allow_value=True)
st.sidebar.markdown("---")

logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "logo.png")
if os.path.exists(logo_path):
    st.sidebar.image(logo_path, use_container_width=True)

page = st.sidebar.radio(
    "Navigate Project",
    ["🏠 Home & Overview", "📊 Interactive EDA", "🤖 Model Training", "🔮 Real-time Prediction", "📈 Model Evaluation"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="font-size: 0.85rem; color: #94a3b8;">
<b>Tech Stack:</b><br/>
• Python 3.14.5<br/>
• Streamlit Dashboard<br/>
• Scikit-learn Pipeline<br/>
• Plotly Graphics
</div>
""", unsafe_allow_value=True)

# --- PAGE 1: HOME & OVERVIEW ---
if page == "🏠 Home & Overview":
    st.markdown("<h1 class='gradient-text' style='font-size:3rem;'>Iris Classification & Advanced KNN</h1>", unsafe_allow_value=True)
    st.markdown("<h3>An end-to-end Machine Learning project using custom scaling, dimensionality reduction, hyperparameter optimization, and validation pipelines.</h3>", unsafe_allow_value=True)
    
    st.markdown("<br/>", unsafe_allow_value=True)
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        <div class="premium-card">
            <h3>📌 Project Introduction</h3>
            <p>This project elevates the classic Iris flower classification problem into an <b>advanced production-ready workflow</b>. Rather than relying on simple, default K-Nearest Neighbors setups, we implement:</p>
            <ul>
                <li><b>Robust scaling</b> to prevent distance calculation skew.</li>
                <li><b>Hyperparameter tuning grids</b> search over K values, weight models, and distance metrics.</li>
                <li><b>Model validation</b> with Stratified K-Fold Cross-Validation.</li>
                <li><b>PCA Space Projection</b> for visual analysis of classification boundaries.</li>
            </ul>
        </div>
        """, unsafe_allow_value=True)
        
        st.markdown("""
        <div class="premium-card">
            <h3>🔬 Dataset Details</h3>
            <p>The Iris dataset consists of 150 samples from three species (<i>Setosa</i>, <i>Versicolor</i>, <i>Virginica</i>). Four physical dimensions were measured for each sample:</p>
            <table style="width:100%; border-collapse: collapse; margin-top: 10px;">
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.1); padding: 8px;">
                    <th style="text-align:left; padding: 8px;">Feature</th>
                    <th style="text-align:left; padding: 8px;">Type</th>
                    <th style="text-align:left; padding: 8px;">Description</th>
                </tr>
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
                    <td style="padding: 8px;">Sepal Length</td>
                    <td style="padding: 8px; color: #a78bfa;">Numerical (cm)</td>
                    <td style="padding: 8px;">Length of the flower sepal</td>
                </tr>
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
                    <td style="padding: 8px;">Sepal Width</td>
                    <td style="padding: 8px; color: #a78bfa;">Numerical (cm)</td>
                    <td style="padding: 8px;">Width of the flower sepal</td>
                </tr>
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
                    <td style="padding: 8px;">Petal Length</td>
                    <td style="padding: 8px; color: #a78bfa;">Numerical (cm)</td>
                    <td style="padding: 8px;">Length of the flower petal</td>
                </tr>
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
                    <td style="padding: 8px;">Petal Width</td>
                    <td style="padding: 8px; color: #a78bfa;">Numerical (cm)</td>
                    <td style="padding: 8px;">Width of the flower petal</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_value=True)

    with col2:
        st.markdown("""
        <div class="premium-card" style="text-align: center;">
            <h4 class="sub-gradient-text">QUICK STATS</h4>
            <div style="margin-top:20px;">
                <div class="metric-value">150</div>
                <div style="color: #94a3b8; font-size: 0.9rem;">Total Samples</div>
            </div>
            <div style="margin-top:20px;">
                <div class="metric-value">4</div>
                <div style="color: #94a3b8; font-size: 0.9rem;">Input Features</div>
            </div>
            <div style="margin-top:20px;">
                <div class="metric-value">3</div>
                <div style="color: #94a3b8; font-size: 0.9rem;">Target Classes</div>
            </div>
        </div>
        """, unsafe_allow_value=True)
        
        # Display dataset preview
        st.markdown("<div class='premium-card'>", unsafe_allow_value=True)
        st.subheader("📋 Dataset Preview")
        st.dataframe(df_full.sample(6, random_state=42), use_container_width=True)
        st.markdown("</div>", unsafe_allow_value=True)

# --- PAGE 2: INTERACTIVE EDA ---
elif page == "📊 Interactive EDA":
    st.markdown("<h1 class='gradient-text'>Interactive Exploratory Data Analysis</h1>", unsafe_allow_value=True)
    st.markdown("### Visualize patterns, distributions, and separations within the Iris attributes.")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.markdown("<div class='premium-card'>", unsafe_allow_value=True)
        st.subheader("Configure Plots")
        x_axis = st.selectbox("X-Axis Feature", feature_names, index=2)
        y_axis = st.selectbox("Y-Axis Feature", feature_names, index=3)
        color_theme = st.selectbox("Color Theme", ["plasma", "viridis", "inferno", "rainbow"])
        plot_type = st.radio("Plot Type", ["2D Scatter Plot", "3D Space Plot", "Distribution Density (KDE)", "Correlation Matrix"])
        st.markdown("</div>", unsafe_allow_value=True)
        
    with col2:
        st.markdown("<div class='premium-card'>", unsafe_allow_value=True)
        if plot_type == "2D Scatter Plot":
            fig = px.scatter(
                df_full, x=x_axis, y=y_axis, color="species",
                color_discrete_sequence=px.colors.sequential.Plasma if color_theme == "plasma" else px.colors.sequential.Viridis,
                title=f"{y_axis} vs {x_axis}",
                template="plotly_dark",
                size=[10]*len(df_full)
            )
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
            
        elif plot_type == "3D Space Plot":
            fig = px.scatter_3d(
                df_full, x=feature_names[0], y=feature_names[2], z=feature_names[3],
                color="species",
                title="3D Feature Space Distribution",
                template="plotly_dark"
            )
            fig.update_layout(scene=dict(bgcolor='rgba(0,0,0,0)'), paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
            
        elif plot_type == "Distribution Density (KDE)":
            fig = px.histogram(
                df_full, x=x_axis, color="species", marginal="box",
                title=f"Distribution of {x_axis}",
                template="plotly_dark",
                barmode="overlay",
                opacity=0.7
            )
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
            
        elif plot_type == "Correlation Matrix":
            corr = X_raw.corr()
            fig = px.imshow(
                corr, text_auto=True, color_continuous_scale="RdBu_r",
                title="Feature Correlation Matrix",
                template="plotly_dark"
            )
            fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_value=True)

# --- PAGE 3: MODEL TRAINING ---
elif page == "🤖 Model Training":
    st.markdown("<h1 class='gradient-text'>Hyperparameter Tuning & Training</h1>", unsafe_allow_value=True)
    st.markdown("### Fine-tune the pipeline and explore cross-validation scores in real-time.")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("<div class='premium-card'>", unsafe_allow_value=True)
        st.subheader("Grid Search Params")
        
        cv_folds = st.slider("Cross-Validation Folds", 2, 10, 5)
        
        weights_options = st.multiselect("KNN Weights", ["uniform", "distance"], default=["uniform", "distance"])
        metrics_options = st.multiselect("Distance Metrics", ["euclidean", "manhattan", "minkowski"], default=["euclidean", "manhattan"])
        
        max_k = st.slider("Max K value for search", 5, 30, 20)
        
        train_btn = st.button("🚀 Run Grid Search Tuning", use_container_width=True)
        st.markdown("</div>", unsafe_allow_value=True)
        
    with col2:
        if train_btn:
            st.markdown("<div class='premium-card'>", unsafe_allow_value=True)
            with st.spinner("Tuning model parameters..."):
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('knn', KNeighborsClassifier())
                ])
                
                param_grid = {
                    'knn__n_neighbors': list(range(1, max_k + 1)),
                    'knn__weights': weights_options,
                    'knn__metric': metrics_options
                }
                
                cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
                grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
                grid_search.fit(X_raw, y_raw)
                
                st.success("Tuning Complete!")
                st.markdown(f"<div class='status-banner'>Best Parameters Found: {grid_search.best_params_}</div>", unsafe_allow_value=True)
                st.markdown(f"<h4>Best 5-Fold Cross-Validation Accuracy: <span class='sub-gradient-text'>{grid_search.best_score_*100:.2f}%</span></h4>", unsafe_allow_value=True)
                
                # Show optimization path
                cv_results = pd.DataFrame(grid_search.cv_results_)
                fig = px.line(
                    cv_results, x='param_knn__n_neighbors', y='mean_test_score',
                    color='param_knn__weights', line_dash='param_knn__metric',
                    title='Grid Search Accuracy vs Neighbors (K)',
                    labels={'param_knn__n_neighbors': 'Number of Neighbors (K)', 'mean_test_score': 'Validation Accuracy'},
                    template="plotly_dark"
                )
                st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_value=True)
        else:
            st.markdown("""
            <div class="premium-card">
                <h3>💡 Tip</h3>
                <p>Click the <b>Run Grid Search Tuning</b> button to execute a full, stratified cross-validation search across different numbers of neighbors, distance metrics, and weight profiles.</p>
            </div>
            """, unsafe_allow_value=True)

# --- PAGE 4: REAL-TIME PREDICTION ---
elif page == "🔮 Real-time Prediction":
    st.markdown("<h1 class='gradient-text'>Real-time Classification Interface</h1>", unsafe_allow_value=True)
    st.markdown("### Drag sliders representing plant dimensions to predict species instantly.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("<div class='premium-card'>", unsafe_allow_value=True)
        st.subheader("Iris Leaf Metrics (cm)")
        
        # User input sliders based on real bounds
        sepal_length = st.slider("Sepal Length", 4.0, 8.0, 5.8, step=0.1)
        sepal_width = st.slider("Sepal Width", 2.0, 4.5, 3.0, step=0.1)
        petal_length = st.slider("Petal Length", 1.0, 7.0, 4.3, step=0.1)
        petal_width = st.slider("Petal Width", 0.1, 2.6, 1.3, step=0.1)
        
        st.markdown("</div>", unsafe_allow_value=True)
        
    with col2:
        st.markdown("<div class='premium-card'>", unsafe_allow_value=True)
        st.subheader("Prediction Outputs")
        
        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        prediction = model_pipeline.predict(input_data)[0]
        probabilities = model_pipeline.predict_proba(input_data)[0]
        
        predicted_species = target_names[prediction]
        
        st.markdown(f"<h3>Predicted Species: <span class='gradient-text'>{predicted_species.capitalize()}</span></h3>", unsafe_allow_value=True)
        
        # Draw confidence bars
        prob_df = pd.DataFrame({
            'Species': [name.capitalize() for name in target_names],
            'Confidence (%)': probabilities * 100
        })
        
        fig = px.bar(
            prob_df, x='Confidence (%)', y='Species', orientation='h',
            text='Confidence (%)', color='Species',
            color_discrete_map={'Setosa': '#a78bfa', 'Versicolor': '#3b82f6', 'Virginica': '#ec4899'},
            template="plotly_dark",
            title="Classification Probability Distribution"
        )
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(xaxis=dict(range=[0, 115]), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_value=True)

# --- PAGE 5: MODEL EVALUATION ---
elif page == "📈 Model Evaluation":
    st.markdown("<h1 class='gradient-text'>Advanced Model Metrics & Visuals</h1>", unsafe_allow_value=True)
    st.markdown("### In-depth performance curves, feature importances, and decision boundary plots.")
    
    tab1, tab2, tab3 = st.tabs(["📊 Performance Curves", "🗺️ Decision Boundaries (PCA)", "🔑 Feature Importances"])
    
    with tab1:
        st.markdown("<div class='premium-card'>", unsafe_allow_value=True)
        st.subheader("Grid Search Evaluation Metrics")
        
        # Load local plots if generated
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.markdown("#### Confusion Matrix Heatmap")
            cm_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "plots", "confusion_matrix.png")
            if os.path.exists(cm_path):
                st.image(cm_path, use_container_width=True)
            else:
                st.info("Run core training script `iris_advanced_knn.py` to generate static confusion matrix.")
                
        with col_c2:
            st.markdown("#### ROC Curve Analysis")
            roc_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "plots", "roc_curves.png")
            if os.path.exists(roc_path):
                st.image(roc_path, use_container_width=True)
            else:
                st.info("Run core training script `iris_advanced_knn.py` to generate static ROC curves.")
        st.markdown("</div>", unsafe_allow_value=True)
        
    with tab2:
        st.markdown("<div class='premium-card'>", unsafe_allow_value=True)
        st.subheader("Decision Boundaries in PCA 2D Projected Space")
        db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "plots", "decision_boundaries.png")
        if os.path.exists(db_path):
            st.image(db_path, use_container_width=True)
        else:
            st.info("Run core training script `iris_advanced_knn.py` to pre-generate decision boundaries plot.")
        st.markdown("</div>", unsafe_allow_value=True)
        
    with tab3:
        st.markdown("<div class='premium-card'>", unsafe_allow_value=True)
        st.subheader("Permutation Feature Importance")
        fi_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "plots", "feature_importance.png")
        if os.path.exists(fi_path):
            st.image(fi_path, use_container_width=True)
        else:
            st.info("Run core training script `iris_advanced_knn.py` to compute permutation importances.")
        st.markdown("</div>", unsafe_allow_value=True)
