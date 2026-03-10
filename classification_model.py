# ============================================================
# classification_model.py - Grade Classification Module
# ============================================================

import numpy as np
import pandas as pd
import joblib
import pickle
from typing import Dict, List, Tuple, Optional, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score, accuracy_score
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONSTANTS (from your notebook analysis)
# ============================================================

# Top 80 features selected via Mutual Information
TOP_80_MI_FEATURES = [
    'root_adv_var', 'root_verb_var', 'simp_pron_var', 'simp_adp_var',
    'SYNSTRUTt', 'simp_part_var', 'formulas.linsear_write_formula',
    'simp_aux_var', 'eg.trans_X_-', 'eg.trans_O_O', 'eg.trans_X_S',
    'aggregates.average_sentence_length', 'simp_cconj_var', 'a_adv_ps',
    'a_n_ent_ordinal_pw', 'coref.num_chains', 'coref.max_chain_span',
    'DESSLd', 'simp_sconj_var', 'eg.trans_S_-', 'LSASSpd', 'lex.avg_chain_span',
    'simp_det_var', 'a_kup_pw', 'coref.avg_chain_span', 'WRDMEAc',
    'a_adj_ps', 'a_num_ps', 'lex.long_span_chains', 'LSAGNd',
    'CRFNO1', 'PCSYNz', 'WRDFRQa', 'eg.trans_X_X', 'coref.long_span_chains',
    'PCVERBp', 'simp_num_var', 'CRFANPa', 'WRDPOLc', 'CNCAll',
    'CRFCWOad', 'a_bry_pw', 'SMCAUSr', 'SMINTEr', 'PCDCz',
    'LDVOCD', 'a_n_ent_person_pw', 'a_intj_pw', 'simp_space_var',
    'WRDPRP1p', 'a_n_ent_time_pw', 'a_n_ent_product_pw', 'PCVERBz',
    'CRFCWO1d', 'a_punct_pw', 'LDMTLD', 'WRDPRP2', 'eg.trans_O_S',
    'eg.trans_O_X', 'eg.trans_S_X', 'coref.avg_inference_distance',
    'coref.long_span_chains_norm', 'a_part_ps', 'a_det_ps', 'a_noun_ps',
    'simp_punct_var', 'simp_cconj_var', 'root_part_var', 'root_sconj_var',
    'root_pron_var', 'aggregates.average_verbs_per_sentence',
    'aggregates.average_pronouns_per_sentence', 'ent.avg_entities_per_sentence',
    'ent.avg_unique_entities_per_sentence', 'ent.entity_density',
    'const.avg_tree_height', 'const.vp_density', 'const.np_density',
    'pos.noun_density', 'pos.unique_noun_density', 'zipf_mean', 'zipf_std'
]

# Grade mapping (from your encoding)
GRADE_MAPPING = {0: 2, 1: 3, 2: 4, 3: 5, 4: 6, 5: 7, 6: 8}
REVERSE_GRADE_MAPPING = {2: 0, 3: 1, 4: 2, 5: 3, 6: 4, 7: 5, 8: 6}

# Boundary features for adjacent grades (from your analysis)
BOUNDARY_FEATURES = {
    (0, 1): ['root_adv_var', 'simp_part_var', 'simp_aux_var', 'eg.trans_S_-', 
             'eg.trans_O_X', 'coref.num_chains', 'simp_pron_var'],
    (1, 2): ['simp_pron_var', 'eg.trans_X_-', 'root_verb_var', 'formulas.linsear_write_formula',
             'LSASSpd', 'a_intj_pw', 'a_n_ent_ordinal_pw'],
    (2, 3): ['PCSYNz', 'a_adj_ps', 'root_adv_var', 'CRFNO1', 'const.adjp_density',
             'SYNSTRUTt', 'aggregates.average_sentence_length'],
    (3, 4): ['simp_pron_var', 'LSASS1d', 'a_num_ps', 'WRDPOLc', 'a_kup_pw',
             'SYNLE', 'eg.trans_X_X', 'eg.trans_O_O'],
    (4, 5): ['simp_adp_var', 'root_verb_var', 'coref.num_chains', 'root_sconj_var',
             'lex.long_span_chains', 'SYNSTRUTt', 'SMINTEr'],
    (5, 6): ['root_adv_var', 'simp_det_var', 'simp_adp_var', 'simp_pron_var',
             'root_verb_var', 'eg.trans_O_S', 'coref.long_span_chains']
}


# ============================================================
# GRADE CLASSIFIER
# ============================================================

class GradeClassifier:
    """
    Signature refinement classifier for predicting story grades (2-8)
    
    Architecture:
    1. Global Random Forest predicts all grades with confidence scores
    2. For uncertain predictions (confidence gap < 0.2), boundary-specific
       models refine between adjacent grades
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.global_model = None
        self.boundary_models = {}
        self.feature_names = TOP_80_MI_FEATURES
        self.grade_mapping = GRADE_MAPPING
        self.reverse_mapping = REVERSE_GRADE_MAPPING
        self.is_fitted = False
        
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> 'GradeClassifier':
        """
        Fit the classifier using training data
        
        Args:
            X: Feature DataFrame (should contain at least TOP_80_MI_FEATURES)
            y: Encoded grade labels (0-6)
        """
        # Ensure we have the required features
        available_features = [f for f in self.feature_names if f in X.columns]
        if len(available_features) < len(self.feature_names):
            print(f"Warning: Using {len(available_features)}/{len(self.feature_names)} features")
        
        X_subset = X[available_features].copy()
        self.available_features = available_features
        
        # 1. Train global model
        self.global_model = RandomForestClassifier(
            n_estimators=500,
            max_features='log2',
            random_state=self.random_state,
            n_jobs=-1
        )
        self.global_model.fit(X_subset, y)
        
        # 2. Train boundary models
        unique_grades = sorted(np.unique(y))
        for (g1, g2), feat_list in BOUNDARY_FEATURES.items():
            if g1 not in unique_grades or g2 not in unique_grades:
                continue
                
            # Filter to available features
            avail_boundary_feats = [f for f in feat_list if f in X_subset.columns]
            if len(avail_boundary_feats) < 3:
                continue
                
            # Get samples for these two grades
            mask = np.isin(y, [g1, g2])
            X_pair = X_subset.loc[mask, avail_boundary_feats]
            y_pair = y[mask]
            
            # Convert to binary (1 for higher grade)
            y_binary = (y_pair == g2).astype(int)
            
            # Train boundary model
            boundary_model = RandomForestClassifier(
                n_estimators=300,
                random_state=self.random_state,
                n_jobs=-1
            )
            boundary_model.fit(X_pair, y_binary)
            
            self.boundary_models[(g1, g2)] = {
                'model': boundary_model,
                'features': avail_boundary_feats
            }
        
        self.is_fitted = True
        return self
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get probability predictions from global model"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
            
        X_subset = X[[f for f in self.available_features if f in X.columns]]
        return self.global_model.predict_proba(X_subset)
    
    def predict(self, X: pd.DataFrame, confidence_threshold: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict grades with signature refinement
        
        Returns:
            - final_predictions: Refined grade predictions (encoded 0-6)
            - confidence_scores: Confidence scores for final predictions
            - continuous_scores: Continuous difficulty scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
            
        # Get global probabilities
        X_subset = X[[f for f in self.available_features if f in X.columns]]
        probs = self.global_model.predict_proba(X_subset)
        
        unique_grades = self.global_model.classes_
        final_preds = []
        confidences = []
        continuous_scores = []
        
        for i in range(len(X_subset)):
            p = probs[i]
            
            # Continuous difficulty score
            cont_score = np.sum(p * unique_grades)
            continuous_scores.append(cont_score)
            
            # Top-2 classes
            top2_idx = np.argsort(p)[-2:]
            top2 = unique_grades[top2_idx]
            top2_probs = p[top2_idx]
            
            # Sort by probability
            sorted_idx = np.argsort(top2_probs)
            g1, g2 = top2[sorted_idx[1]], top2[sorted_idx[0]]  # g1 is highest
            confidence_gap = p[g1] - p[g2]
            confidences.append(p[g1])
            
            # Check if refinement is needed
            need_refinement = (abs(g1 - g2) == 1) and (confidence_gap < confidence_threshold)
            
            if need_refinement and (g1, g2) in self.boundary_models:
                # Use boundary model
                low, high = min(g1, g2), max(g1, g2)
                boundary_info = self.boundary_models[(low, high)]
                
                # Get features for this sample
                X_sample = X_subset.iloc[i:i+1][boundary_info['features']]
                boundary_pred = boundary_info['model'].predict(X_sample)[0]
                
                final_pred = high if boundary_pred == 1 else low
            else:
                final_pred = g1
            
            final_preds.append(final_pred)
        
        return np.array(final_preds), np.array(confidences), np.array(continuous_scores)
    
    def predict_grade(self, X: pd.DataFrame, confidence_threshold: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict actual grade numbers (2-8)
        
        Returns:
            - grade_predictions: Grade numbers (2-8)
            - confidence_scores: Confidence scores
            - continuous_scores: Continuous difficulty scores
        """
        encoded_preds, confidences, continuous = self.predict(X, confidence_threshold)
        grade_preds = np.array([self.grade_mapping.get(p, p) for p in encoded_preds])
        return grade_preds, confidences, continuous
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get global feature importance"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
            
        importance_df = pd.DataFrame({
            'feature': self.available_features,
            'importance': self.global_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def get_boundary_importance(self, grade_pair: Tuple[int, int]) -> Optional[pd.DataFrame]:
        """Get feature importance for a boundary model"""
        if grade_pair not in self.boundary_models:
            return None
            
        info = self.boundary_models[grade_pair]
        model = info['model']
        
        importance_df = pd.DataFrame({
            'feature': info['features'],
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df


# ============================================================
# CROSS-VALIDATION EVALUATION
# ============================================================

def evaluate_model(X: pd.DataFrame, y: np.ndarray, n_splits: int = 5, random_state: int = 42) -> Dict[str, float]:
    """
    Evaluate classifier with stratified cross-validation
    
    Returns dictionary of metrics
    """
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    all_true = []
    all_pred = []
    all_scores = []
    
    for train_idx, val_idx in kf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Train model
        clf = GradeClassifier(random_state=random_state)
        clf.fit(X_train, y_train)
        
        # Predict
        preds, _, scores = clf.predict(X_val)
        
        all_true.extend(y_val)
        all_pred.extend(preds)
        all_scores.extend(scores)
    
    # Calculate metrics
    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    all_scores = np.array(all_scores)
    
    accuracy = accuracy_score(all_true, all_pred)
    qwk = cohen_kappa_score(all_true, all_pred, weights='quadratic')
    pm1 = np.mean(np.abs(all_true - all_pred) <= 1)
    spearman_corr, _ = spearmanr(all_scores, all_true)
    
    return {
        'accuracy': accuracy,
        'qwk': qwk,
        'pm1': pm1,
        'spearman': spearman_corr
    }


# ============================================================
# MODEL SERIALIZATION
# ============================================================

def save_model(model: GradeClassifier, filepath: str):
    """Save trained model to disk"""
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)


def load_model(filepath: str) -> GradeClassifier:
    """Load trained model from disk"""
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    return model


# ============================================================
# EXPLANATION GENERATION
# ============================================================

def generate_prediction_explanation(
    features: Dict[str, float],
    prediction: int,
    confidence: float,
    continuous_score: float,
    top_features: List[Tuple[str, float]],
    boundary_used: Optional[Tuple[int, int]] = None
) -> Dict[str, Any]:
    """
    Generate human-readable explanation for a prediction
    """
    grade_labels = ['Grade 2', 'Grade 3', 'Grade 4', 'Grade 5', 'Grade 6', 'Grade 7', 'Grade 8']

    def _format_grade_label(value: int) -> str:
        ivalue = int(value)
        if 2 <= ivalue <= 8:
            return f"Grade {ivalue}"
        if 0 <= ivalue < len(grade_labels):
            return grade_labels[ivalue]
        return str(ivalue)
    
    explanation = {
        'prediction': prediction,
        'confidence': confidence,
        'continuous_score': continuous_score,
        'interpretation': '',
        'key_factors': [],
        'boundary_info': None
    }
    
    # Interpret confidence
    prediction_label = _format_grade_label(prediction)
    if confidence >= 0.8:
        explanation['interpretation'] = f"High confidence prediction for {prediction_label}"
    elif confidence >= 0.6:
        explanation['interpretation'] = f"Moderate confidence prediction for {prediction_label}"
    else:
        explanation['interpretation'] = f"Low confidence prediction for {prediction_label}"
    
    # Add top contributing features
    for feat, importance in top_features[:5]:
        if feat in features:
            explanation['key_factors'].append({
                'feature': feat,
                'value': features[feat],
                'importance': importance,
                'direction': 'positive' if importance > 0 else 'negative'
            })
    
    # Add boundary information if used
    if boundary_used:
        g1, g2 = boundary_used
        explanation['boundary_info'] = {
            'grades': f"{_format_grade_label(g1)} vs {_format_grade_label(g2)}",
            'reason': f"Ambiguous between adjacent grades (confidence gap {1-confidence:.2f})"
        }
    
    return explanation