from sklearn.metrics import roc_auc_score, roc_curve,precision_score, recall_score, f1_score, confusion_matrix, classification_report, auc, precision_recall_curve, log_loss
from sklearn.model_selection import cross_validate
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def train_model(name, model, X_train, y_train, scoring_metrics=['roc_auc', 'f1', 'precision', 'recall']):
    """
    Train and evaluate a single model using cross-validation.
    
    Parameters:
    -----------
    name : str
        Name of the model for display/storage
    model : sklearn estimator
        The model to train and evaluate
    X_train : array-like
        Training features
    y_train : array-like
        Training target
    scoring_metrics : list
        List of metrics to evaluate
        
    Returns:
    --------
    dict : Dictionary with model results
    """
    
    # Perform cross-validation with multiple metrics
    cv_results = cross_validate(model, X_train, y_train, cv=5, scoring=scoring_metrics)
    
    # Build results dictionary
    results = {
        'model': model,
        'cv_roc_auc': cv_results['test_roc_auc'].mean(),
        'cv_roc_auc_std': cv_results['test_roc_auc'].std(),
        'cv_f1': cv_results['test_f1'].mean(),
        'cv_f1_std': cv_results['test_f1'].std(),
        'cv_precision': cv_results['test_precision'].mean(),
        'cv_recall': cv_results['test_recall'].mean(),
        'cv_results': cv_results
    }
    
    # Print results
    print(f"{name}:")
    print(f"  ROC AUC: {cv_results['test_roc_auc'].mean():.4f} ± {cv_results['test_roc_auc'].std():.4f}")
    print(f"  F1:      {cv_results['test_f1'].mean():.4f} ± {cv_results['test_f1'].std():.4f}")
    print(f"  Precision: {cv_results['test_precision'].mean():.4f}")
    print(f"  Recall:    {cv_results['test_recall'].mean():.4f}")
    print()
    
    return results

def model_evaluation(model, X_train, y_train, X_test, y_test, baseline_f1=0.574):
    """
    Comprehensive evaluation of your tuned model, including threshold-based and threshold-free metrics.
    """
    print("\n1. 🌐PERFORMANCE")

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_f1 = f1_score(y_train, y_train_pred)
    test_f1 = f1_score(y_test, y_test_pred)

    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_test_proba = model.predict_proba(X_test)[:, 1]
    train_roc_auc = roc_auc_score(y_train, y_train_proba)
    test_roc_auc = roc_auc_score(y_test, y_test_proba)

    print(f"Training F1:    {train_f1:.4f}")
    print(f"Test F1:        {test_f1:.4f}")
    print(f"F1 Difference:  {abs(train_f1 - test_f1):.4f} {'✅' if abs(train_f1 - test_f1) < 0.05 else '⚠️ Possible overfitting'}")
    print()
    print(f"Training ROC AUC:   {train_roc_auc:.4f}")
    print(f"Test ROC AUC:       {test_roc_auc:.4f}")
    print(f"ROC AUC Difference: {abs(train_roc_auc - test_roc_auc):.4f}")
    
    # 3. Baseline Comparison
    print(f"\n2. 🎯 BASELINE COMPARISON")
    print(f"Baseline F1:        {baseline_f1:.4f}")
    print(f"Your Model F1:      {test_f1:.4f}")
    print(f"Improvement:        {test_f1 - baseline_f1:+.4f} ({(test_f1/baseline_f1-1)*100:+.1f}%)")
    
    # 4. Detailed Classification Report
    print(f"\n3. 📋 DETAILED CLASSIFICATION REPORT")
    print(classification_report(y_test, y_test_pred, target_names=['No Churn', 'Churn']))
    
    # 5. Confusion Matrix Analysis
    print(f"\n4. 🔍 CONFUSION MATRIX ANALYSIS")
    cm = confusion_matrix(y_test, y_test_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"True Negatives (Correct No-Churn):  {tn}")
    print(f"False Positives (Wrong Churn):      {fp}")
    print(f"False Negatives (Missed Churn):     {fn}")
    print(f"True Positives (Correct Churn):     {tp}")
    print()
    
   # Calculate Recall Metrics
    total_actual_churn = tp + fn
    total_actual_no_churn = tn + fp
    recall_churn = tp / total_actual_churn if total_actual_churn > 0 else 0  # Recall for churn
    recall_no_churn = tn / total_actual_no_churn if total_actual_no_churn > 0 else 0  # Recall for no-churn
    macro_recall = (recall_churn + recall_no_churn) / 2  # Macro average recall
    
    print(f"Recall for Churn:    {recall_churn:.1%} (Proportion of churners correctly identified)")
    print(f"Recall for No-Churn: {recall_no_churn:.1%} (Proportion of no-churners correctly identified)")
    print(f"Macro Average Recall:{macro_recall:.1%} (Unweighted average of both recalls)")
    print()
    
    precision_rate = tp / (tp + fp) if (tp + fp) > 0 else 0  # Precision for context
    print(f"Precision Rate:     {precision_rate:.1%} (Accuracy of churn predictions)")
    
    return {
        'test_f1': test_f1,
        'test_roc_auc': test_roc_auc,
        'train_f1': train_f1,
        'train_roc_auc': train_roc_auc,
        'recall_churn': recall_churn,
        'macro_recall': macro_recall,
        'confusion_matrix': cm
    }

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def plot_roc_curve(model, X_test, y_test, model_name):
    """
    Plot an enhanced ROC curve with professional styling and detailed annotations.
    
    This function creates a publication-ready ROC curve visualization with:
    - Modern color scheme and typography
    - Confidence intervals (optional)
    - Enhanced grid and styling
    - Professional annotations and metrics display
    
    Parameters
    ----------
    model : classifier object
        A trained classifier that supports the `predict_proba` method
    X_test : array-like of shape (n_samples, n_features)
        Feature matrix for the test set
    y_test : array-like of shape (n_samples,)
        True binary labels for the test set
    model_name : str
        Name of the model to display in the plot title and legend
        
    Returns
    -------
    None
        Displays an enhanced ROC curve plot with comprehensive metrics
    """
    # Get probability scores
    y_test_proba = model.predict_proba(X_test)[:, 1]
    
    # Compute ROC curve and ROC AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_test_proba)
    roc_auc = auc(fpr, tpr)
    
    # Create figure with enhanced styling
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor('white')
    
    # Define modern color palette
    primary_color = '#2E86AB'
    secondary_color = '#A23B72'
    accent_color = '#F18F01'
    neutral_color = '#C73E1D'
    
    # Plot ROC curve with gradient-like effect
    ax.plot(fpr, tpr, color=primary_color, linewidth=3.5, 
            label=f'{model_name}\n(AUC = {roc_auc:.3f})', alpha=0.9)
    
    # Fill area under curve for visual impact
    ax.fill_between(fpr, tpr, alpha=0.15, color=primary_color)
    
    # Plot diagonal reference line
    ax.plot([0, 1], [0, 1], color=neutral_color, linewidth=2.5, 
            linestyle='--', alpha=0.8, label='Random Classifier\n(AUC = 0.500)')
    
    # Enhanced grid
    ax.grid(True, alpha=0.3, linewidth=0.8)
    ax.set_axisbelow(True)
    
    # Styling and labels
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    
    # Enhanced labels with better typography
    ax.set_xlabel('False Positive Rate (1 - Specificity)', 
                  fontsize=14, fontweight='600', color='#2c3e50', labelpad=12)
    ax.set_ylabel('True Positive Rate (Sensitivity)', 
                  fontsize=14, fontweight='600', color='#2c3e50', labelpad=12)
    
    # Professional title
    title = f'ROC Analysis: {model_name} Performance'
    ax.set_title(title, fontsize=16, fontweight='700', color='#34495e', pad=20)
    
    # Enhanced legend
    legend = ax.legend(loc='lower right', frameon=True, fancybox=True, 
                      shadow=True, fontsize=11, framealpha=0.95)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('#bdc3c7')
    
    # Add performance annotation box
    textstr = f'Model Performance:\n• AUC Score: {roc_auc:.3f}\n• Classification Quality: '
    if roc_auc >= 0.9:
        quality = 'Excellent'
        quality_color = '#27ae60'
    elif roc_auc >= 0.8:
        quality = 'Good'
        quality_color = '#f39c12'
    elif roc_auc >= 0.7:
        quality = 'Fair'
        quality_color = '#e67e22'
    else:
        quality = 'Poor'
        quality_color = '#e74c3c'
    
    textstr += quality
    
    # Create annotation box
    props = dict(boxstyle='round,pad=0.8', facecolor='white', alpha=0.9, 
                edgecolor='#bdc3c7', linewidth=1.5)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props, color='#2c3e50')
    
    # Add quality indicator
    ax.text(0.02, 0.75, f'● {quality}', transform=ax.transAxes, 
            fontsize=12, fontweight='bold', color=quality_color)
    
    # Enhance tick labels
    ax.tick_params(axis='both', which='major', labelsize=11, colors='#2c3e50')
    
    # Add subtle border
    for spine in ax.spines.values():
        spine.set_color('#bdc3c7')
        spine.set_linewidth(1.2)
    
    plt.tight_layout()
    plt.show()


def plot_precision_recall_curve(model, X_test, y_test, model_name: str) -> None:
    """
    Plot an enhanced Precision-Recall curve with professional styling and insights.
    
    This function creates a publication-ready PR curve visualization featuring:
    - Modern design with gradient fills
    - Comprehensive performance metrics
    - Professional color scheme and typography
    - Optimal threshold annotations
    
    Args:
        model: Trained binary classification model with `predict_proba` method
        X_test (pd.DataFrame or np.ndarray): Test feature set
        y_test (pd.Series or np.ndarray): True binary labels for test data
        model_name (str): Model name for plot title and legend
        
    Returns:
        None. Displays an enhanced matplotlib PR curve plot with detailed metrics
    """
    y_test_proba = model.predict_proba(X_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, y_test_proba)
    pr_auc = auc(recall, precision)
    
    # Calculate baseline (random classifier performance)
    baseline = np.sum(y_test) / len(y_test)
    
    # Find optimal threshold (highest F1 score)
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1])
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    optimal_precision = precision[optimal_idx]
    optimal_recall = recall[optimal_idx]
    optimal_f1 = f1_scores[optimal_idx]
    
    # Create enhanced figure
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor('white')
    
    # Modern color palette
    primary_color = '#8E44AD'
    secondary_color = '#16A085'
    accent_color = '#E67E22'
    baseline_color = '#95A5A6'
    
    # Plot PR curve with enhanced styling
    ax.plot(recall, precision, color=primary_color, linewidth=3.5, alpha=0.9,
            label=f'{model_name}\n(PR AUC = {pr_auc:.3f})')
    
    # Fill area under curve
    ax.fill_between(recall, precision, alpha=0.15, color=primary_color)
    
    # Plot baseline (random classifier)
    ax.axhline(y=baseline, color=baseline_color, linestyle='--', linewidth=2.5,
               alpha=0.8, label=f'Random Baseline\n(PR AUC ≈ {baseline:.3f})')
    
    # Highlight optimal threshold point
    ax.scatter(optimal_recall, optimal_precision, color=accent_color, s=120, 
              zorder=5, alpha=0.9, edgecolors='white', linewidth=2)
    
    # Add optimal point annotation
    ax.annotate(f'Optimal Threshold\n({optimal_recall:.2f}, {optimal_precision:.2f})\nF1 = {optimal_f1:.3f}',
                xy=(optimal_recall, optimal_precision), xytext=(0.3, 0.7),
                textcoords='axes fraction', fontsize=10, color='#2c3e50',
                arrowprops=dict(arrowstyle='->', color=accent_color, lw=2),
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                         alpha=0.9, edgecolor=accent_color))
    
    # Enhanced grid
    ax.grid(True, alpha=0.3, linewidth=0.8)
    ax.set_axisbelow(True)
    
    # Set limits with padding
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    
    # Professional labels
    ax.set_xlabel('Recall (Sensitivity)', fontsize=14, fontweight='600', 
                  color='#2c3e50', labelpad=12)
    ax.set_ylabel('Precision (Positive Predictive Value)', fontsize=14, 
                  fontweight='600', color='#2c3e50', labelpad=12)
    
    # Enhanced title
    title = f'Precision-Recall Analysis: {model_name} Performance'
    ax.set_title(title, fontsize=16, fontweight='700', color='#34495e', pad=20)
    
    # Professional legend
    legend = ax.legend(loc='lower left', frameon=True, fancybox=True, 
                      shadow=True, fontsize=11, framealpha=0.95)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('#bdc3c7')

    
    # Determine performance quality
    if pr_auc >= 0.8:
        quality = 'Excellent'
        quality_color = '#27ae60'
    elif pr_auc >= 0.6:
        quality = 'Good'
        quality_color = '#f39c12'
    elif pr_auc >= 0.4:
        quality = 'Fair'
        quality_color = '#e67e22'
    else:
        quality = 'Needs Improvement'
        quality_color = '#e74c3c'
    
    # Add metrics annotation
    props = dict(boxstyle='round,pad=0.8', facecolor='white', alpha=0.9, 
                edgecolor='#bdc3c7', linewidth=1.5)
    
    # Add quality indicator
    ax.text(0.02, 0.65, f'● Quality: {quality}', transform=ax.transAxes, 
            fontsize=12, fontweight='bold', color=quality_color)
    
    # Enhanced tick styling
    ax.tick_params(axis='both', which='major', labelsize=11, colors='#2c3e50')
    
    # Subtle border styling
    for spine in ax.spines.values():
        spine.set_color('#bdc3c7')
        spine.set_linewidth(1.2)
    
    plt.tight_layout()
    plt.show()
