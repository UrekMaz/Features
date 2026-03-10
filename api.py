"""
api.py
════════════════════════════════════════════════════════════════════════
FastAPI backend for the Readability Classification App.

Endpoints:
    GET  /health          — server + model status
    POST /classify        — predict grade + feature importance
    GET  /grade-means     — per-grade mean feature values (for radar chart)
    POST /nudge           — what to change to reach target grade

Run:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload
════════════════════════════════════════════════════════════════════════
"""

# ── NumPy compatibility shim — MUST be first ──────────────────────────────────
# Artifacts saved in Colab (numpy 2.x) reference numpy._core.* which does not
# exist in numpy 1.x. Create aliases so pickle can resolve them.
import sys
import importlib
try:
    import numpy.core as _np_core
    sys.modules.setdefault("numpy._core", _np_core)
    for _sub in ["multiarray", "umath", "numeric", "_multiarray_umath", "fromnumeric"]:
        try:
            sys.modules.setdefault(
                f"numpy._core.{_sub}",
                importlib.import_module(f"numpy.core.{_sub}")
            )
        except ModuleNotFoundError:
            pass
except Exception:
    pass

import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List
from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Load artifacts ────────────────────────────────────────────────────────────
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")

def load_artifact(name):
    with open(os.path.join(ARTIFACTS_DIR, name), "rb") as f:
        return pickle.load(f)

print("[api] Loading artifacts...")
model         = load_artifact("model.pkl")
scaler        = load_artifact("scaler.pkl")
fs5_features  = load_artifact("fs5_features.pkl")
grade_means   = load_artifact("grade_means.pkl")
label_encoder = load_artifact("label_encoder.pkl")
print(f"[api] Artifacts loaded — {len(fs5_features)} FS5 features, "
      f"grades {label_encoder.classes_}, scaler n_features={scaler.n_features_in_}")

# ── Load FS5 extractor ────────────────────────────────────────────────────────
print("[api] Loading FS5 extractor...")
from fs5_extractor import FS5Extractor
extractor = FS5Extractor()
print("[api] FS5 extractor ready.")

# ── Feature meta — plain English descriptions ─────────────────────────────────
FEATURE_META = {
    "coref.active_chains_per_entity":   ("Coreference",    "Ratio of coreference chains to total entity mentions"),
    "coref.avg_inference_distance":     ("Coreference",    "Average token distance between consecutive mentions of the same entity"),
    "coref.avg_chain_span":             ("Coreference",    "Average span (in tokens) of each coreference chain"),
    "coref.avg_refs_per_chain":         ("Coreference",    "Average number of mentions per coreference chain"),
    "eg.trans_O_S":                     ("Entity Grid",    "Proportion of Object→Subject transitions in entity grid"),
    "eg.trans_O_O":                     ("Entity Grid",    "Proportion of Object→Object transitions in entity grid"),
    "eg.trans_X_X":                     ("Entity Grid",    "Proportion of Other→Other transitions in entity grid"),
    "eg.trans_S_O":                     ("Entity Grid",    "Proportion of Subject→Object transitions in entity grid"),
    "eg.max_entity_persistence":        ("Entity Grid",    "Maximum fraction of sentences any single entity appears in"),
    "const.np_density":                 ("Syntax",         "Noun phrase density per 100 words"),
    "const.advp_density":               ("Syntax",         "Adverb phrase density per 100 words"),
    "const.adjp_density":               ("Syntax",         "Adjective phrase density per 100 words"),
    "pos.first_person_pronoun_density": ("POS",            "First person pronoun density (% of tokens)"),
    "pos.adverb_density":               ("POS",            "Adverb density per 100 words"),
    "lex.long_span_chains":             ("Lexical",        "Proportion of coreference chains spanning more than half the document"),
    "formulas.linsear_write_formula":   ("Readability",    "Linsear Write readability formula score"),
    "WTopc10_S":  ("World Knowledge",   "Topics identified using Wikipedia (100 topics)"),
    "WTopc15_S":  ("World Knowledge",   "Topics identified using Wikipedia (150 topics)"),
    "WTopc05_S":  ("World Knowledge",   "Topics identified using Wikipedia (50 topics)"),
    "WClar20_S":  ("World Knowledge",   "Semantic clarity using Wikipedia (200 topics)"),
    "WClar10_S":  ("World Knowledge",   "Semantic clarity using Wikipedia (100 topics)"),
    "WNois10_S":  ("World Knowledge",   "Semantic noise using Wikipedia (100 topics)"),
    "WNois05_S":  ("World Knowledge",   "Semantic noise using Wikipedia (50 topics)"),
    "WRich05_S":  ("World Knowledge",   "Semantic richness using Wikipedia (50 topics)"),
    "WRich10_S":  ("World Knowledge",   "Semantic richness using Wikipedia (100 topics)"),
    "BRich15_S":  ("World Knowledge",   "Semantic richness using WeeBit corpus (150 topics)"),
    "BNois10_S":  ("World Knowledge",   "Semantic noise using WeeBit corpus (100 topics)"),
    "BNois05_S":  ("World Knowledge",   "Semantic noise using WeeBit corpus (50 topics)"),
    "OClar20_S":  ("World Knowledge",   "Semantic clarity using OneStopEng (200 topics)"),
    "OClar05_S":  ("World Knowledge",   "Semantic clarity using OneStopEng (50 topics)"),
    "ONois05_S":  ("World Knowledge",   "Semantic noise using OneStopEng (50 topics)"),
    "OTopc15_S":  ("World Knowledge",   "Topics identified using OneStopEng (150 topics)"),
    "OTopc20_S":  ("World Knowledge",   "Topics identified using OneStopEng (200 topics)"),
    "OTopc05_S":  ("World Knowledge",   "Topics identified using OneStopEng (50 topics)"),
    "ORich05_S":  ("World Knowledge",   "Semantic richness using OneStopEng (50 topics)"),
    "SquaNoV_S":  ("Lexical Diversity", "Squared noun variation"),
    "SquaVeV_S":  ("Lexical Diversity", "Squared verb variation"),
    "SquaAvV_S":  ("Lexical Diversity", "Squared adverb variation"),
    "SimpAjV_S":  ("Lexical Diversity", "Simple adjective variation"),
    "CorrAjV_S":  ("Lexical Diversity", "Corrected adjective variation"),
    "LoCohPA_S":  ("Discourse",         "Local coherence PA score"),
    "LoCoDPU_S":  ("Discourse",         "Local coherence DPU score"),
    "ra_CoNoT_C": ("Syntax",            "Ratio of coordinating conjunctions to nouns"),
    "ra_PrSuP_C": ("Syntax",            "Ratio of prepositional phrases to subordinate clauses"),
    "ra_SuAjP_C": ("Syntax",            "Ratio of subordinate clauses to adjective phrases"),
    "ra_NoAjP_C": ("Syntax",            "Ratio of nouns to adjective phrases"),
    "ra_PrAvP_C": ("Syntax",            "Ratio of prepositional phrases to adverb phrases"),
    "ra_AvPrP_C": ("Syntax",            "Ratio of adverb phrases to prepositional phrases"),
    "ra_PrVeP_C": ("Syntax",            "Ratio of prepositional phrases to verb phrases"),
    "ra_VeSuT_C": ("Syntax",            "Ratio of verbs to subordinating conjunctions"),
    "ra_NoVeT_C": ("Syntax",            "Ratio of nouns to verbs"),
    "ra_SuPrP_C": ("Syntax",            "Ratio of subordinate clauses to prepositional phrases"),
    "ra_SuAvT_C": ("Syntax",            "Ratio of subordinating conjunctions to adverbs"),
    "ra_SuCoT_C": ("Syntax",            "Ratio of subordinating to coordinating conjunctions"),
    "ra_AvVeT_C": ("Syntax",            "Ratio of adverbs to verbs"),
    "ra_AvSuT_C": ("Syntax",            "Ratio of adverbs to subordinating conjunctions"),
    "ra_AvNoT_C": ("Syntax",            "Ratio of adverbs to nouns"),
    "ra_CoAjT_C": ("Syntax",            "Ratio of coordinating conjunctions to adjectives"),
    "ra_SuPrP_C": ("Syntax",            "Ratio of subordinate to prepositional phrases"),
    "ra_PrSuP_C": ("Syntax",            "Ratio of prepositional to subordinate phrases"),
    "at_SbL1W_C": ("Lexical",           "Average syllable count per word"),
    "at_ContW_C": ("Lexical",           "Average content words per token"),
    "at_AAKuL_C": ("Psycholinguistic",  "Average Age of Acquisition (Kuperman) per lemma"),
    "at_FuncW_C": ("Lexical",           "Average function words per token"),
    "at_TreeH_C": ("Syntax",            "Average parse tree height"),
    "at_AjPhr_C": ("Syntax",            "Average adjective phrases per sentence"),
    "at_AABrL_C": ("Psycholinguistic",  "Average Age of Acquisition (Bristol) per lemma"),
    "at_SbFrL_C": ("Lexical",           "Average syllable frequency per lemma"),
    "as_EntiM_C": ("Entities",          "Average named entity mentions per sentence"),
    "as_CoTag_C": ("Syntax",            "Average coordinating conjunctions per sentence"),
    "as_AjPhr_C": ("Syntax",            "Average adjective phrases per sentence"),
    "as_UEnti_C": ("Entities",          "Average unique entity types per sentence"),
    "as_NoTag_C": ("Syntax",            "Average nouns per sentence"),
    "SMCAUSwn":   ("Cohesion",          "Causal verb overlap using WordNet"),
    "SMCAUSv":    ("Cohesion",          "Causal verb incidence"),
    "SMINTEp":    ("Cohesion",          "Intentional causal connective incidence"),
    "WRDPOLc":    ("Word Information",  "Word polysemy count"),
    "WRDFAMc":    ("Word Information",  "Word familiarity count"),
    "WRDCNCc":    ("Word Information",  "Word concreteness count"),
    "WRDMEAc":    ("Word Information",  "Word meaningfulness count"),
    "WRDHYPnv":   ("Word Information",  "Hypernymy score for nouns and verbs"),
    "WRDFRQmc":   ("Word Information",  "Word frequency (MC) count"),
    "WRDFRQa":    ("Word Information",  "Word frequency (age-of-acquisition) count"),
    "WRDAOAc":    ("Word Information",  "Word age-of-acquisition count"),
    "CNCTemp":    ("Connectives",       "Temporal connective count"),
    "CNCTempx":   ("Connectives",       "Extended temporal connective count"),
    "CNCCaus":    ("Connectives",       "Causal connective count"),
    "CNCNeg":     ("Connectives",       "Negative connective count"),
    "PCTEMPp":    ("Text Structure",    "Temporal connective incidence per 1000 words"),
    "PCCONNp":    ("Text Structure",    "All connective incidence per 1000 words"),
    "PCSYNp":     ("Text Structure",    "Syntactic complexity percentage"),
    "SYNSTRUTt":  ("Syntax",            "Syntactic structure similarity across sentences"),
    "SYNNP":      ("Syntax",            "Noun phrase incidence"),
    "SYNLE":      ("Syntax",            "Left embeddedness — words before main verb"),
    "DRPVAL":     ("Discourse",         "Pronoun ratio"),
    "DRNEG":      ("Discourse",         "Negation ratio"),
    "CRFCWOad":   ("Cohesion",          "Content word overlap across adjacent sentences"),
    "CRFCWO1d":   ("Cohesion",          "Content word overlap sentence 1 vs all others"),
    "CRFAO1":     ("Cohesion",          "Argument overlap sentence 1 vs all others"),
    "LSAGNd":     ("Cohesion",          "LSA given/new — new information per sentence"),
    "LSASSpd":    ("Cohesion",          "LSA sentence similarity std deviation"),
    "LSAGN":      ("Cohesion",          "LSA given/new score"),
    "LSASS1":     ("Cohesion",          "LSA sentence 1 similarity"),
    "LSASS1d":    ("Cohesion",          "LSA sentence 1 similarity (discourse)"),
    "LSASSp":     ("Cohesion",          "LSA sentence similarity mean"),
    "WRDPRP3p":   ("Word Information",  "Third person pronoun incidence per 1000 words"),
    "PCREFp":     ("Cohesion",          "Referential cohesion — adjacent sentences sharing noun/pronoun"),
    "simp_num_var":   ("Lexical Variety", "Number word variation"),
    "simp_adp_var":   ("Lexical Variety", "Adposition variation"),
    "simp_cconj_var": ("Lexical Variety", "Coordinating conjunction variation"),
    "simp_pron_var":  ("Lexical Variety", "Pronoun variation"),
    "simp_det_var":   ("Lexical Variety", "Determiner variation"),
    "simp_noun_var":  ("Lexical Variety", "Noun variation"),
    "corr_adp_var":   ("Lexical Variety", "Corrected adposition variation"),
    "corr_adv_var":   ("Lexical Variety", "Corrected adverb variation"),
    "corr_propn_var": ("Lexical Variety", "Corrected proper noun variation"),
    "root_num_var":   ("Lexical Variety", "Root number variation"),
    "root_punct_var": ("Lexical Variety", "Root punctuation variation"),
    "root_pron_var":  ("Lexical Variety", "Root pronoun variation"),
    "n_adv":          ("Surface", "Adverb count"),
    "n_upunct":       ("Surface", "Unique punctuation count"),
    "n_space":        ("Surface", "Space token count"),
    "n_udet":         ("Syntax",  "Unique determiners"),
    "n_upart":        ("Syntax",  "Unique particles"),
    "n_uaux":         ("Syntax",  "Unique auxiliary verbs"),
    "n_intj":         ("Syntax",  "Interjection count"),
    "a_n_ent_norp_pw":     ("Entities", "NORP entity density per word"),
    "a_n_ent_date_ps":     ("Entities", "Date entity density per sentence"),
    "a_n_ent_org_pw":      ("Entities", "Organisation entity density per word"),
    "a_n_ent_cardinal_ps": ("Entities", "Cardinal entity density per sentence"),
    "a_n_ent_product_pw":  ("Entities", "Product entity density per word"),
    "a_n_ent_language_ps": ("Entities", "Language entity density per sentence"),
    "a_n_ent_time_ps":     ("Entities", "Time entity density per sentence"),
    "a_n_ent_art_pw":      ("Entities", "Art entity density per word"),
    "a_n_ent_ordinal_ps":  ("Entities", "Ordinal entity density per sentence"),
    "a_n_ent_gpe_ps":      ("Entities", "GPE entity density per sentence"),
    "a_n_ent_fac_ps":      ("Entities", "Facility entity density per sentence"),
    "a_n_ent_quantity_ps": ("Entities", "Quantity entity density per sentence"),
    "a_subtlex_us_zipf_pw":("Psycholinguistic", "SUBTLEX-US Zipf frequency per word"),
    "a_char_pw":           ("Surface",          "Average characters per word"),
    "a_bry_pw":            ("Psycholinguistic", "Brysbaert concreteness per word"),
}

# ── Nudge direction: +1 = increase to go harder, -1 = decrease to go harder ──
FEATURE_DIRECTION = {
    "at_AAKuL_C": +1, "at_ContW_C": +1, "WRDPOLc": +1, "WRDHYPnv": +1,
    "BRich15_S": +1, "WRich05_S": +1, "WRich10_S": +1,
    "WTopc10_S": +1, "WTopc15_S": +1, "WTopc05_S": +1,
    "SquaNoV_S": +1, "SquaVeV_S": +1,
    "SMCAUSwn": +1, "SMCAUSv": +1, "CNCCaus": +1,
    "CRFCWOad": +1, "LSAGNd": +1, "CRFCWO1d": +1,
    "ra_NoVeT_C": +1, "ra_VeSuT_C": +1, "ra_PrVeP_C": +1,
    "coref.avg_inference_distance": +1, "coref.avg_chain_span": +1,
    "eg.max_entity_persistence": +1, "lex.long_span_chains": +1,
    "const.np_density": +1, "n_uaux": +1, "n_adv": +1,
    "simp_adp_var": +1, "simp_num_var": +1, "corr_adp_var": +1,
    "formulas.linsear_write_formula": +1, "SYNLE": +1,
    "a_char_pw": +1, "ONois05_S": +1, "DRNEG": +1,
    "PCCONNp": +1, "PCSYNp": +1, "WRDAOAc": +1,
    "at_TreeH_C": +1, "SYNNP": +1, "CNCTempx": +1,
    "a_bry_pw": -1, "a_subtlex_us_zipf_pw": -1,
    "WRDFRQa": -1, "WRDFRQmc": -1, "WRDFAMc": -1,
    "PCNARp": -1, "CNCTemp": -1, "SYNSTRUTt": -1,
    "at_FuncW_C": -1, "as_CoTag_C": -1, "ra_CoNoT_C": -1,
    "simp_det_var": -1, "simp_pron_var": -1, "simp_cconj_var": -1,
    "LoCohPA_S": -1, "LoCoDPU_S": -1,
    "pos.first_person_pronoun_density": -1, "WRDPRP3p": -1,
    "PCTEMPp": -1,
}

NUDGE_ADVICE = {
    "at_AAKuL_C":   ("Use more advanced vocabulary",     "Replace common words with sophisticated alternatives"),
    "at_ContW_C":   ("Increase information density",     "Pack more content words per sentence — reduce filler phrases"),
    "WRDPOLc":      ("Use more nuanced vocabulary",      "Choose words with multiple meanings or connotations"),
    "WRDHYPnv":     ("Use more abstract language",       "Replace concrete terms with more general/abstract concepts"),
    "BRich15_S":    ("Enrich topic coverage",            "Introduce more domain-specific terminology"),
    "WTopc10_S":    ("Broaden topic range",              "Cover more topics or subtopics within the text"),
    "SquaNoV_S":    ("Diversify noun vocabulary",        "Use a wider range of nouns rather than repeating the same ones"),
    "SMCAUSwn":     ("Add causal language",              "Use 'therefore', 'consequently', 'as a result'"),
    "CRFCWOad":     ("Increase sentence cohesion",       "Repeat key content words across adjacent sentences"),
    "LSAGNd":       ("Balance given/new information",    "Introduce new concepts gradually — scaffold before presenting new ideas"),
    "ra_NoVeT_C":   ("Add more noun phrases",            "Use more nominalisations and complex noun phrases"),
    "coref.avg_inference_distance": ("Space out references", "Refer back to entities across longer distances"),
    "n_uaux":       ("Use varied auxiliary verbs",       "Use 'might', 'could', 'should', 'would' for modality"),
    "simp_adp_var": ("Vary prepositions",                "Use a wider range of prepositions"),
    "const.np_density": ("Add more noun phrases",        "Increase use of complex noun phrases"),
    "SYNLE":        ("Use more left-embedded sentences", "Add more words/phrases before the main verb"),
    "PCSYNp":       ("Add subordinate clauses",          "Use 'although', 'because', 'while' for complexity"),
    "PCNARp":       ("Reduce narrative tone",            "Shift from story-telling to expository/analytical writing"),
    "CNCTemp":      ("Reduce temporal connectives",      "Use fewer time-sequence markers like 'then', 'next', 'finally'"),
    "SYNSTRUTt":    ("Vary sentence structure",          "Use more varied sentence patterns"),
    "at_FuncW_C":   ("Use simpler grammar",              "Reduce grammatical complexity — fewer function words"),
    "as_CoTag_C":   ("Shorten compound sentences",       "Split long compound sentences into shorter ones"),
    "ra_CoNoT_C":   ("Reduce compound structures",       "Use fewer 'and'/'or' compounds"),
    "simp_det_var": ("Simplify determiner usage",        "Use simpler, more consistent determiners"),
    "simp_pron_var":("Simplify pronoun usage",           "Use fewer types of pronouns for clearer reference"),
    "LoCohPA_S":    ("Improve local coherence",          "Ensure entities are carried across sentences consistently"),
    "pos.first_person_pronoun_density": ("Reduce first person", "Use fewer 'I'/'we' pronouns — shift to third person"),
    "a_char_pw":    ("Use longer words",                 "Replace short simple words with longer more precise vocabulary"),
    "formulas.linsear_write_formula": ("Increase readability complexity", "Use longer sentences and more complex words"),
    "ONois05_S":    ("Increase semantic complexity",     "Introduce more nuanced and ambiguous vocabulary"),
    "DRNEG":        ("Add negation",                     "Use negation constructions to add nuance and precision"),
    "WRDFAMc":      ("Use less familiar words",          "Replace highly familiar words with more specialised vocabulary"),
    "a_bry_pw":     ("Use more abstract words",          "Replace concrete familiar words with more abstract terms"),
    "at_TreeH_C":   ("Use deeper sentence structures",   "Embed more clauses to increase parse tree height"),
    "simp_cconj_var":("Simplify conjunctions",           "Use simpler coordinating conjunctions"),
}

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Readability Classification API",
    description="Classify text readability grade and explain predictions",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Pydantic models ───────────────────────────────────────────────────────────
class TextRequest(BaseModel):
    text: str

class NudgeRequest(BaseModel):
    text: str
    target_grade: int


def _extract_text(payload) -> str:
    if isinstance(payload, str):
        return payload
    if isinstance(payload, dict):
        for key in ("text", "input_text", "content"):
            value = payload.get(key)
            if isinstance(value, str):
                return value
    return ""

# ── Prediction ────────────────────────────────────────────────────────────────
def predict(features_dict: Dict[str, float]):
    """
    Scale using full 515-feature scaler, then select 136 FS5 features for model.
    The scaler was fit on X (all available features) before correlation filtering.
    """
    all_cols = list(scaler.feature_names_in_)
    row_full = {f: float(features_dict.get(f, 0.0)) for f in all_cols}
    X_full   = pd.DataFrame([row_full])[all_cols]
    X_scaled = pd.DataFrame(scaler.transform(X_full), columns=all_cols)
    X_fs5    = X_scaled[fs5_features]

    pred        = model.predict(X_fs5)[0]
    grade       = int(label_encoder.inverse_transform([pred])[0])
    proba       = model.predict_proba(X_fs5)[0]
    classes     = label_encoder.inverse_transform(model.classes_)
    confidence  = float(proba.max())
    grade_probs = {int(c): round(float(p), 4) for c, p in zip(classes, proba)}
    return grade, confidence, grade_probs, X_fs5.iloc[0].to_numpy(dtype=float)


def get_feature_importance(scaled_vector: np.ndarray, predicted_class_idx: int) -> List[Dict]:
    coef     = model.coef_[predicted_class_idx]
    contribs = coef * scaled_vector
    result   = []
    for i, feat in enumerate(fs5_features):
        group, desc = FEATURE_META.get(feat, ("Other", feat))
        result.append({
            "feature":      feat,
            "group":        group,
            "description":  desc,
            "contribution": round(float(contribs[i]), 4),
            "raw_value":    round(float(scaled_vector[i]), 4),
        })
    result.sort(key=lambda x: abs(x["contribution"]), reverse=True)
    return result

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status":    "ok",
        "model":     "LogisticRegression C=100 l2 liblinear",
        "features":  len(fs5_features),
        "grades":    [int(g) for g in label_encoder.classes_],
        "extractor": "FS5Extractor ready",
    }


@app.post("/classify")
async def classify(request: Request):
    payload = {}
    try:
        payload = await request.json()
    except Exception:
        payload = {}

    text = _extract_text(payload)
    if not text:
        text = request.query_params.get("text", "")

    if not text or not text.strip():
        raise HTTPException(400, "Text cannot be empty")
    if len(text.split()) < 10:
        raise HTTPException(400, "Text too short — please provide at least 10 words")

    try:
        features = await run_in_threadpool(extractor.extract, text)
    except Exception as e:
        raise HTTPException(500, f"Feature extraction failed: {e}")

    grade, confidence, grade_probs, scaled_vec = await run_in_threadpool(predict, features)
    pred_encoded = label_encoder.transform([grade])[0]
    class_idx    = int(np.where(model.classes_ == pred_encoded)[0][0])
    top_features = await run_in_threadpool(get_feature_importance, scaled_vec, class_idx)

    return {
        "grade":        grade,
        "confidence":   round(confidence, 4),
        "grade_probs":  grade_probs,
        "top_features": top_features[:10],
        "all_features": {k: round(float(v), 4) for k, v in features.items()},
        "word_count":   len(text.split()),
    }


@app.get("/grade-means")
def get_grade_means():
    return {
        "grades":   sorted(grade_means.keys()),
        "features": fs5_features,
        "means": {
            str(grade): {
                k: round(float(v), 4)
                for k, v in means.items()
                if k in fs5_features
            }
            for grade, means in grade_means.items()
        }
    }


@app.post("/nudge")
def nudge(req: NudgeRequest):
    if not req.text or not req.text.strip():
        raise HTTPException(400, "Text cannot be empty")

    valid_grades = list(label_encoder.classes_.astype(int))
    if req.target_grade not in valid_grades:
        raise HTTPException(400, f"Invalid target grade. Must be one of {valid_grades}")

    try:
        features = extractor.extract(req.text)
    except Exception as e:
        raise HTTPException(500, f"Feature extraction failed: {e}")

    current_grade, confidence, grade_probs, scaled_vec = predict(features)

    if current_grade == req.target_grade:
        return {
            "current_grade": current_grade,
            "target_grade":  req.target_grade,
            "message":       "Text is already classified at the target grade.",
            "nudges":        []
        }

    target_means = grade_means.get(req.target_grade, {})
    moving_up    = req.target_grade > current_grade
    needed_dir   = +1 if moving_up else -1

    nudges = []
    for feat in fs5_features:
        current_val = features.get(feat, 0.0)
        target_val  = target_means.get(feat, current_val)
        gap         = target_val - current_val
        if abs(gap) < 0.01:
            continue
        if FEATURE_DIRECTION.get(feat, 0) != needed_dir:
            continue

        group, desc   = FEATURE_META.get(feat, ("Other", feat))
        title, advice = NUDGE_ADVICE.get(feat, (feat, "Adjust this feature toward the target grade"))
        nudges.append({
            "feature":       feat,
            "group":         group,
            "description":   desc,
            "title":         title,
            "advice":        advice,
            "current_value": round(float(current_val), 4),
            "target_value":  round(float(target_val),  4),
            "gap":           round(float(gap),          4),
            "direction":     "increase" if gap > 0 else "decrease",
            "priority":      round(abs(gap), 4),
        })

    nudges.sort(key=lambda x: x["priority"], reverse=True)

    return {
        "current_grade": current_grade,
        "target_grade":  req.target_grade,
        "moving":        "up" if moving_up else "down",
        "confidence":    round(confidence, 4),
        "grade_probs":   grade_probs,
        "nudge_count":   len(nudges),
        "nudges":        nudges[:8],
    }