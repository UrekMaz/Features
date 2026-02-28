# ============================================================
# FASTAPI ENDPOINT FOR FEATURE EXTRACTION
# ============================================================

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import os
import nltk
import torch
import spacy
import stanza
from sentence_transformers import SentenceTransformer

# Import the complete feature extraction pipeline
from discourse_pipeline import (
    extract_all_features,
    get_feature_summary,
    COH_METRIX_CLI_PATH,
    COH_METRIX_OUTPUT_DIR
)

# Download NLTK data
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('universal_tagset', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nltk.download('punkt_tab', quiet=True)

# Load NLP models
print("Loading spaCy model...")
spacy_nlp = spacy.load("en_core_web_sm")
spacy_nlp.add_pipe(
    "fastcoref",
    config={
        "model_architecture": "FCoref",
        "model_path": "biu-nlp/f-coref",
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }
)

print("Loading Stanza model...")
stanza.download("en", verbose=False)
stanza_nlp = stanza.Pipeline(
    lang="en",
    processors="tokenize,pos,lemma,constituency,depparse",
    verbose=False
)

print("Loading Sentence-BERT model...")
sbert_model = SentenceTransformer("all-mpnet-base-v2")

# Create output directory for Coh-Metrix if needed
if not os.path.exists(COH_METRIX_OUTPUT_DIR):
    os.makedirs(COH_METRIX_OUTPUT_DIR)

app = FastAPI(
    title="Discourse Feature Extraction API",
    description="Extract 200+ discourse features from text including coreference, entity grid, lexical chains, constituency, readability, LFTK, and Coh-Metrix",
    version="1.0.0"
)

class TextRequest(BaseModel):
    text: str
    include_lftk: bool = True
    include_cohmetrix: bool = False

class FeatureResponse(BaseModel):
    feature_count: int
    features: Dict[str, float]
    categories: Dict[str, int]

@app.get("/")
def root():
    """Root endpoint with API information"""
    return {
        "name": "Discourse Feature Extraction API",
        "version": "1.0.0",
        "endpoints": {
            "/extract-features": "POST - Extract all features from text",
            "/feature-summary": "GET - Get summary of all available features",
            "/health": "GET - Health check"
        }
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": {
            "spacy": True,
            "stanza": True,
            "sbert": True,
            "lftk": True
        },
        "cohmetrix_available": os.path.exists(COH_METRIX_CLI_PATH)
    }

@app.get("/feature-summary")
def feature_summary():
    """Get summary of all available features by category"""
    return get_feature_summary()

@app.post("/extract-features", response_model=FeatureResponse)
def extract_features_endpoint(req: TextRequest):
    """
    Extract all discourse features from input text
    
    - **text**: The input text to analyze
    - **include_lftk**: Whether to include LFTK features (default: True)
    - **include_cohmetrix**: Whether to include Coh-Metrix features (default: False)
    
    Returns:
    - **feature_count**: Total number of features extracted
    - **features**: Dictionary of feature names and values
    - **categories**: Count of features per category
    """
    
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    # Extract all features
    all_features = extract_all_features(
        text=req.text,
        spacy_nlp=spacy_nlp,
        stanza_nlp=stanza_nlp,
        sbert_model=sbert_model,
        include_lftk=req.include_lftk,
        include_cohmetrix=req.include_cohmetrix,
        cohmetrix_cli_path=COH_METRIX_CLI_PATH if req.include_cohmetrix else None
    )
    
    # Count features by category
    categories = {}
    for key in all_features.keys():
        category = key.split('.')[0] if '.' in key else 'other'
        categories[category] = categories.get(category, 0) + 1
    
    return {
        "feature_count": len(all_features),
        "features": all_features,
        "categories": categories
    }

@app.post("/extract-features-batch")
def extract_features_batch(texts: List[str], include_lftk: bool = True, include_cohmetrix: bool = False):
    """
    Extract features from multiple texts
    """
    results = []
    for text in texts:
        try:
            features = extract_all_features(
                text=text,
                spacy_nlp=spacy_nlp,
                stanza_nlp=stanza_nlp,
                sbert_model=sbert_model,
                include_lftk=include_lftk,
                include_cohmetrix=include_cohmetrix,
                cohmetrix_cli_path=COH_METRIX_CLI_PATH if include_cohmetrix else None
            )
            results.append({
                "success": True,
                "feature_count": len(features),
                "features": features
            })
        except Exception as e:
            results.append({
                "success": False,
                "error": str(e)
            })
    
    return results

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)