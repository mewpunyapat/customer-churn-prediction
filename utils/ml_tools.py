from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_validate

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
    Comprehensive evaluation of your tuned model.
    """
    print("=== PHASE 1: COMPREHENSIVE MODEL EVALUATION ===")
    
    # 1. Fit the model (if not already fitted)
    if not hasattr(model, 'classes_'):
        model.fit(X_train, y_train)
    
    # 2. Training vs Test Performance
    print("1. 📊 TRAINING vs TEST PERFORMANCE")
    
    # Training performance
    y_train_pred = model.predict(X_train)
    y_train_proba = model.predict_proba(X_train)[:, 1]
    
    train_f1 = f1_score(y_train, y_train_pred)
    train_roc_auc = roc_auc_score(y_train, y_train_proba)
    
    # Test performance  
    y_test_pred = model.predict(X_test)
    y_test_proba = model.predict_proba(X_test)[:, 1]
    
    test_f1 = f1_score(y_test, y_test_pred)
    test_roc_auc = roc_auc_score(y_test, y_test_proba)
    
    print(f"Training F1:    {train_f1:.4f}")
    print(f"Test F1:        {test_f1:.4f}")
    print(f"F1 Difference:  {abs(train_f1 - test_f1):.4f} {'✅' if abs(train_f1 - test_f1) < 0.05 else '⚠️ Possible overfitting'}")
    print()
    print(f"Training ROC:   {train_roc_auc:.4f}")
    print(f"Test ROC:       {test_roc_auc:.4f}")
    print(f"ROC Difference: {abs(train_roc_auc - test_roc_auc):.4f}")
    
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
    
    # Business metrics
    total_actual_churn = tp + fn
    total_predicted_churn = tp + fp
    churn_capture_rate = tp / total_actual_churn if total_actual_churn > 0 else 0
    precision_rate = tp / total_predicted_churn if total_predicted_churn > 0 else 0
    
    print(f"Churn Capture Rate: {churn_capture_rate:.1%} (How many churners did we catch?)")
    print(f"Precision Rate:     {precision_rate:.1%} (How accurate are our churn predictions?)")
    
    return {
        'test_f1': test_f1,
        'test_roc_auc': test_roc_auc,
        'train_f1': train_f1,
        'train_roc_auc': train_roc_auc,
        'confusion_matrix': cm,
        'y_test_proba': y_test_proba
    }