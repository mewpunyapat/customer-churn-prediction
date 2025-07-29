from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, auc, precision_recall_curve, log_loss
from sklearn.model_selection import cross_validate
import numpy as np

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

def find_optimal_threshold(y_true, y_proba, metric='f1'):
    """
    Find optimal threshold based on different metrics
    """
    thresholds = np.arange(0.1, 0.9, 0.01)
    scores = []
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        
        if metric == 'f1':
            score = f1_score(y_true, y_pred)
        elif metric == 'precision':
            score = precision_score(y_true, y_pred)
        elif metric == 'recall':
            score = recall_score(y_true, y_pred)
        
        scores.append(score)
    
    optimal_idx = np.argmax(scores)
    optimal_threshold = thresholds[optimal_idx]
    optimal_score = scores[optimal_idx]
    
    return optimal_threshold, optimal_score, thresholds, scores

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