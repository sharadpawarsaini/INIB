import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA

# Set plotting style for professional look
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 16
})

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(BASE_DIR, "assets", "plots")
MODEL_PATH = os.path.join(BASE_DIR, "iris_knn_pipeline.joblib")

os.makedirs(PLOTS_DIR, exist_ok=True)

def load_and_prepare_data():
    """Loads the Iris dataset and returns train/test splits as DataFrames."""
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name="target")
    
    # Class names map
    target_names = iris.target_names
    
    # Stratified split to maintain class ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test, target_names, X, y

def generate_eda_plots(X, y, target_names):
    """Generates and saves exploratory data analysis plots."""
    df = X.copy()
    df['species'] = y.map(dict(enumerate(target_names)))
    
    # 1. Pairplot
    plt.figure(figsize=(10, 8))
    pairplot = sns.pairplot(df, hue='species', palette='viridis', diag_kind='kde')
    pairplot.fig.suptitle("Iris Dataset Pairwise Relationships", y=1.02)
    pairplot.savefig(os.path.join(PLOTS_DIR, "pairplot.png"), bbox_inches='tight', dpi=150)
    plt.close()
    
    # 2. Correlation Heatmap (numerical features only)
    plt.figure(figsize=(8, 6))
    sns.heatmap(X.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, square=True)
    plt.title("Correlation Matrix of Iris Features")
    plt.savefig(os.path.join(PLOTS_DIR, "correlation_heatmap.png"), bbox_inches='tight', dpi=150)
    plt.close()

def train_advanced_knn(X_train, y_train):
    """Trains a KNN model using Pipeline and Grid Search CV."""
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier())
    ])
    
    # Extensive hyperparameter grid
    param_grid = {
        'knn__n_neighbors': list(range(1, 26)),
        'knn__weights': ['uniform', 'distance'],
        'knn__metric': ['euclidean', 'manhattan', 'minkowski'],
        'knn__p': [1, 2, 3] # p=1 (manhattan), p=2 (euclidean), p=3 (minkowski p=3)
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=cv, scoring='accuracy', n_jobs=-1, return_train_score=True
    )
    grid_search.fit(X_train, y_train)
    
    return grid_search

def plot_hyperparameter_tuning(grid_search):
    """Plots validation accuracy vs K neighbors for different weight options."""
    results = pd.DataFrame(grid_search.cv_results_)
    
    # Filter for minkowski metric (default) to make the plot readable
    minkowski_results = results[results['param_knn__metric'] == 'minkowski']
    
    plt.figure(figsize=(10, 6))
    for weight in ['uniform', 'distance']:
        subset = minkowski_results[(minkowski_results['param_knn__weights'] == weight) & (minkowski_results['param_knn__p'] == 2)]
        plt.plot(subset['param_knn__n_neighbors'], subset['mean_test_score'], 
                 marker='o', label=f'Minkowski (Euclidean) - {weight}')
        
    plt.xlabel('Number of Neighbors (K)')
    plt.ylabel('5-Fold CV Accuracy')
    plt.title('Hyperparameter Tuning: Accuracy vs K Neighbors')
    plt.legend()
    plt.savefig(os.path.join(PLOTS_DIR, "hyperparameter_tuning.png"), bbox_inches='tight', dpi=150)
    plt.close()

def plot_learning_curve(estimator, X, y):
    """Plots the learning curve of the best estimator."""
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, scoring='accuracy', n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10), random_state=42
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, 'o-', color="r", label="Training Score")
    plt.plot(train_sizes, test_mean, 'o-', color="g", label="Cross-Validation Score")
    
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="g")
    
    plt.xlabel("Training Examples")
    plt.ylabel("Accuracy Score")
    plt.title("Learning Curves (Best KNN Classifier)")
    plt.legend(loc="best")
    plt.savefig(os.path.join(PLOTS_DIR, "learning_curve.png"), bbox_inches='tight', dpi=150)
    plt.close()

def evaluate_model(model, X_test, y_test, target_names):
    """Generates evaluation metrics and figures: Confusion Matrix, ROC curves."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix (Test Set)')
    plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrix.png"), bbox_inches='tight', dpi=150)
    plt.close()
    
    # 2. Multi-class ROC Curve (One-vs-Rest)
    plt.figure(figsize=(10, 6))
    for i, class_name in enumerate(target_names):
        # Binarize labels for current class
        y_test_bin = (y_test == i).astype(int)
        fpr, tpr, _ = roc_curve(y_test_bin, y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{class_name} (AUC = {roc_auc:.3f})')
        
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves (One-vs-Rest)')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(PLOTS_DIR, "roc_curves.png"), bbox_inches='tight', dpi=150)
    plt.close()
    
    # Print report
    report = classification_report(y_test, y_pred, target_names=target_names)
    print("Classification Report:\n", report)
    return report

def plot_decision_boundaries(X, y, target_names):
    """Fits PCA to 2 components and plots KNN decision boundaries."""
    # Scale first
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA to 2D
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Fit KNN on 2D
    knn_2d = KNeighborsClassifier(n_neighbors=5, weights='distance')
    knn_2d.fit(X_pca, y)
    
    # Meshgrid
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    
    Z = knn_2d.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolor='k', s=50)
    
    # Add legend
    handles, labels = scatter.legend_elements()
    plt.legend(handles, target_names, loc="upper right", title="Species")
    
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('KNN Decision Boundaries in 2D PCA Space')
    plt.savefig(os.path.join(PLOTS_DIR, "decision_boundaries.png"), bbox_inches='tight', dpi=150)
    plt.close()

def plot_feature_importance(model, X_test, y_test):
    """Calculates and plots permutation feature importance."""
    result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
    sorted_importances_idx = result.importances_mean.argsort()
    
    plt.figure(figsize=(10, 6))
    plt.boxplot(
        result.importances[sorted_importances_idx].T,
        vert=False,
        tick_labels=np.array(X_test.columns)[sorted_importances_idx],
    )
    plt.title("Permutation Feature Importances (Test Set)")
    plt.xlabel("Decrease in Accuracy Score")
    plt.savefig(os.path.join(PLOTS_DIR, "feature_importance.png"), bbox_inches='tight', dpi=150)
    plt.close()

def main():
    print("Loading data...")
    X_train, X_test, y_train, y_test, target_names, X, y = load_and_prepare_data()
    
    print("Generating EDA plots...")
    generate_eda_plots(X, y, target_names)
    
    print("Training KNN via Grid Search...")
    grid_search = train_advanced_knn(X_train, y_train)
    best_model = grid_search.best_estimator_
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    print("Generating training plots...")
    plot_hyperparameter_tuning(grid_search)
    plot_learning_curve(best_model, X, y)
    plot_decision_boundaries(X, y, target_names)
    plot_feature_importance(best_model, X_test, y_test)
    
    print("Evaluating model...")
    evaluate_model(best_model, X_test, y_test, target_names)
    
    print(f"Saving model pipeline to {MODEL_PATH}...")
    joblib.dump(best_model, MODEL_PATH)
    print("Core ML module execution complete!")

if __name__ == "__main__":
    main()
