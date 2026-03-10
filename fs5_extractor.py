"""
fs5_extractor.py  —  136-feature FS5 extractor
════════════════════════════════════════════════════════════════════════
Extracts all features for the definitive 77% accuracy model.
X_filtered was already StandardScaler-normalised before RFECV,
so api.py must apply the same scaler (fit on all 301 features)
before selecting the 136 fs5 columns.

Sources
───────
Custom   : coref.*, eg.*, const.*, pos.*, lex.*, formulas.*
LingFeat : full extraction (all submodules)
CohMetrix: CLI + manual Python for missing features
LFTK     : full extraction

Manual CohMetrix features (not in CLI output):
    CRFCWOad, LSAGNd, LSASSpd, WRDPRP3p, PCTEMPp,
    PCSYNp, PCCONNp, CRFCWO1d, LSASS1d, CNCNeg
════════════════════════════════════════════════════════════════════════
"""

import os, sys, subprocess, tempfile
import numpy as np
import spacy, torch, textstat, lftk
from collections import Counter
from typing import Dict
from fastcoref import spacy_component
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# ── LingFeat local import ─────────────────────────────────────────────────────
LINGFEAT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lingfeat")
sys.path = [p for p in sys.path if "lingfeat" not in p.lower()]
sys.path.insert(0, LINGFEAT_PATH)
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "lf_extractor", os.path.join(LINGFEAT_PATH, "lingfeat", "extractor.py"))
lf_extractor_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lf_extractor_mod)

# ── CohMetrix ─────────────────────────────────────────────────────────────────
COH_METRIX_CLI = os.path.join(os.path.dirname(__file__), "CohMetrixCore", "CohMetrixCoreCLI.exe")
COH_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "cohmetrix_output")

# ── Word lists ────────────────────────────────────────────────────────────────
FIRST_PERSON_SINGULAR = {"i", "me", "my", "mine", "myself"}
FIRST_PERSON_PLURAL   = {"we", "us", "our", "ours", "ourselves"}
THIRD_PERSON          = {"he","him","his","himself","she","her","hers","herself",
                          "they","them","their","theirs","themselves","it","its","itself"}
CONTENT_POS           = {"NOUN","VERB","ADJ","ADV"}
NARRATIVE_MARKERS     = {"said","says","told","asked","thought","felt","saw","heard",
                          "went","came","walked","ran","looked","smiled","laughed",
                          "cried","yelled","whispered","replied","answered","wondered"}
TEMPORAL_CONNECTIVES  = {"after","afterward","afterwards","before","previously","earlier",
                          "later","subsequently","then","next","finally","eventually",
                          "meanwhile","simultaneously","when","while","during","since",
                          "until","once","first","second","third","last","initially",
                          "formerly","soon","immediately","recently","currently"}
CAUSAL_CONNECTIVES    = {"because","therefore","thus","hence","consequently","so",
                          "as a result","due to","owing to","since","accordingly",
                          "for this reason","thereby"}
NEGATIVE_WORDS        = {"no","not","never","neither","nor","nobody","nothing",
                          "nowhere","hardly","scarcely","barely","without"}
CONNECTIVES_ALL       = TEMPORAL_CONNECTIVES | CAUSAL_CONNECTIVES | {
    "and","but","or","however","although","though","while","whereas",
    "furthermore","moreover","nevertheless","nonetheless","instead",
    "otherwise","meanwhile","similarly","likewise","conversely"}


# ══════════════════════════════════════════════════════════════════════════════
class FS5Extractor:

    def __init__(self):
        print("[FS5Extractor] Loading models...")
        self._load_spacy()
        self._load_sbert()
        os.makedirs(COH_OUTPUT_DIR, exist_ok=True)
        print("[FS5Extractor] Ready.")

    def _load_spacy(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.nlp.add_pipe("fastcoref", config={
            "model_architecture": "FCoref",
            "model_path": "biu-nlp/f-coref",
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        })

    def _load_sbert(self):
        try:
            self.sbert = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception as e:
            print(f"[SBERT] Not available: {e}")
            self.sbert = None

    # ── Public ────────────────────────────────────────────────────────────────
    def extract(self, text: str) -> Dict[str, float]:
        if not text or not str(text).strip():
            return {}
        out = {}
        out.update(self._extract_custom(text))
        out.update(self._extract_lingfeat(text))
        out.update(self._extract_cohmetrix_cli(text))
        out.update(self._extract_cohmetrix_manual(text))
        out.update(self._extract_lftk(text))
        return out

    # ══════════════════════════════════════════════════════════════════════════
    # CUSTOM
    # ══════════════════════════════════════════════════════════════════════════

    def _extract_custom(self, text: str) -> Dict[str, float]:
        out = {}
        out.update(self._coref_features(text))
        out.update(self._entity_grid_features(text))
        out.update(self._constituency_features(text))
        out.update(self._pos_features(text))
        out.update(self._lex_features(text))
        out.update(self._readability_features(text))
        return out

    # ── Coreference ───────────────────────────────────────────────────────────
    def _coref_features(self, text: str) -> Dict[str, float]:
        defaults = {
            "coref.active_chains_per_entity": 0.0,
            "coref.avg_inference_distance":   0.0,
            "coref.avg_chain_span":           0.0,
            "coref.avg_refs_per_chain":       0.0,
        }
        try:
            doc      = self.nlp(text)
            clusters = doc._.coref_clusters
            if not clusters:
                return defaults

            num_chains     = len(clusters)
            total_mentions = sum(len(c) for c in clusters)

            inference_distances, chain_spans = [], []
            for cluster in clusters:
                mentions = sorted(cluster,
                    key=lambda m: m.start if hasattr(m,"start") else m[0])
                starts = [m.start if hasattr(m,"start") else m[0] for m in mentions]
                ends   = [m.end   if hasattr(m,"end")   else m[1] for m in mentions]
                chain_spans.append(ends[-1] - starts[0])
                for i in range(1, len(mentions)):
                    inference_distances.append(starts[i] - ends[i-1])

            return {
                "coref.active_chains_per_entity": float(num_chains / total_mentions),
                "coref.avg_inference_distance":   float(np.mean(inference_distances)) if inference_distances else 0.0,
                "coref.avg_chain_span":           float(np.mean(chain_spans)),
                "coref.avg_refs_per_chain":       float(total_mentions / num_chains),
            }
        except Exception as e:
            print(f"[coref] {e}")
            return defaults

    # ── Entity grid ───────────────────────────────────────────────────────────
    def _entity_grid_features(self, text: str) -> Dict[str, float]:
        ROLES       = ["S","O","X","-"]
        DEP_TO_ROLE = {"nsubj":"S","nsubjpass":"S","csubj":"S",
                        "dobj":"O","obj":"O","iobj":"O","pobj":"O"}
        defaults = {f"eg.trans_{r1}_{r2}": 0.0 for r1 in ROLES for r2 in ROLES}
        defaults["eg.max_entity_persistence"] = 0.0

        try:
            doc      = self.nlp(text)
            clusters = getattr(doc._,"coref_clusters",[]) or []
            m2e = {}
            for cid, cluster in enumerate(clusters):
                for m in cluster:
                    s = m.start if hasattr(m,"start") else m[0]
                    e = m.end   if hasattr(m,"end")   else m[1]
                    m2e[(s,e)] = f"E{cid}"

            sents_ents = []
            for sent in doc.sents:
                se = {}
                for t in sent:
                    if not t.is_alpha: continue
                    eid = m2e.get((t.i, t.i+1))
                    if eid is None:
                        if t.pos_ in {"NOUN","PROPN"}: eid = t.lemma_.lower()
                        else: continue
                    role = DEP_TO_ROLE.get(t.dep_,"X")
                    if eid not in se or role in {"S","O"}:
                        se[eid] = role
                sents_ents.append(se)

            if len(sents_ents) < 2:
                return defaults

            all_ents = sorted({e for s in sents_ents for e in s})

            # transition probabilities
            transitions = []
            for i in range(len(sents_ents)-1):
                for ent in all_ents:
                    transitions.append((sents_ents[i].get(ent,"-"),
                                        sents_ents[i+1].get(ent,"-")))
            total   = len(transitions)
            counter = Counter(transitions)
            result  = {
                f"eg.trans_{r1}_{r2}": float(counter.get((r1,r2),0)/total) if total else 0.0
                for r1 in ROLES for r2 in ROLES
            }

            # max entity persistence: max fraction of sentences any entity appears in
            max_persist = max(
                sum(1 for se in sents_ents if ent in se) / len(sents_ents)
                for ent in all_ents
            ) if all_ents else 0.0
            result["eg.max_entity_persistence"] = float(max_persist)
            return result

        except Exception as e:
            print(f"[entity_grid] {e}")
            return defaults

    # ── Constituency ──────────────────────────────────────────────────────────
    def _constituency_features(self, text: str) -> Dict[str, float]:
        try:
            doc    = self.nlp(text)
            wc     = max(textstat.lexicon_count(text, removepunct=True), 1)
            scale  = 100 / wc
            np_cnt = len(list(doc.noun_chunks))
            sbar   = sum(1 for t in doc if t.dep_ in {"mark","advcl","relcl","csubj","csubjpass","acl"})
            advp   = sum(1 for t in doc if t.dep_ == "advmod")
            adjp   = sum(1 for t in doc if t.pos_ == "ADJ" and t.dep_ in {"amod","acomp","advmod"})
            return {
                "const.np_density":   float(np_cnt * scale),
                "const.sbar_density": float(sbar   * scale),
                "const.advp_density": float(advp   * scale),
                "const.adjp_density": float(adjp   * scale),
            }
        except Exception as e:
            print(f"[constituency] {e}")
            return {"const.np_density":0.0,"const.sbar_density":0.0,
                    "const.advp_density":0.0,"const.adjp_density":0.0}

    # ── POS density ───────────────────────────────────────────────────────────
    def _pos_features(self, text: str) -> Dict[str, float]:
        try:
            doc   = self.nlp(text)
            wc    = max(textstat.lexicon_count(text, removepunct=True), 1)
            scale = 100 / wc
            tokens_lower = [t.text.lower() for t in doc if t.is_alpha]
            total = max(len(tokens_lower), 1)

            unique_nouns = {t.lemma_.lower() for t in doc if t.pos_=="NOUN"}
            preps        = sum(1 for t in doc if t.pos_=="ADP")
            advs         = sum(1 for t in doc if t.pos_=="ADV")
            fp_sing      = sum(1 for t in tokens_lower if t in FIRST_PERSON_SINGULAR)
            fp_plur      = sum(1 for t in tokens_lower if t in FIRST_PERSON_PLURAL)
            fp_all       = fp_sing + fp_plur

            return {
                "pos.unique_noun_density":        float(len(unique_nouns) * scale),
                "pos.preposition_density":        float(preps * scale),
                "pos.adverb_density":             float(advs * scale),
                "pos.first_person_pronoun_density": float(fp_all / total * 100),
            }
        except Exception as e:
            print(f"[pos] {e}")
            return {"pos.unique_noun_density":0.0,"pos.preposition_density":0.0,
                    "pos.adverb_density":0.0,"pos.first_person_pronoun_density":0.0}

    # ── Lexical / discourse ───────────────────────────────────────────────────
    def _lex_features(self, text: str) -> Dict[str, float]:
        """
        lex.long_span_chains: proportion of coref chains whose span
        (last mention end − first mention start) exceeds half the doc length.
        """
        try:
            doc      = self.nlp(text)
            clusters = getattr(doc._,"coref_clusters",[]) or []
            doc_len  = len(doc)
            if not clusters or doc_len == 0:
                return {"lex.long_span_chains": 0.0}
            long = sum(1 for cluster in clusters
                       if self._chain_span(cluster) > doc_len / 2)
            return {"lex.long_span_chains": float(long / len(clusters))}
        except Exception as e:
            print(f"[lex] {e}")
            return {"lex.long_span_chains": 0.0}

    @staticmethod
    def _chain_span(cluster) -> int:
        starts = [m.start if hasattr(m,"start") else m[0] for m in cluster]
        ends   = [m.end   if hasattr(m,"end")   else m[1] for m in cluster]
        return max(ends) - min(starts) if starts else 0

    # ── Readability formulas ──────────────────────────────────────────────────
    def _readability_features(self, text: str) -> Dict[str, float]:
        try:
            wc  = max(textstat.lexicon_count(text, removepunct=True), 1)
            sc  = max(textstat.sentence_count(text), 1)
            lc  = textstat.letter_count(text, ignore_spaces=True)
            return {
                "formulas.flesch_reading_ease":          float(textstat.flesch_reading_ease(text)),
                "formulas.flesch_kincaid_grade":         float(textstat.flesch_kincaid_grade(text)),
                "formulas.smog_index":                   float(textstat.smog_index(text)),
                "formulas.automated_readability_index":  float(textstat.automated_readability_index(text)),
                "formulas.coleman_liau_index":           float(textstat.coleman_liau_index(text)),
                "formulas.dale_chall_readability_score": float(textstat.dale_chall_readability_score(text)),
                "formulas.linsear_write_formula":        float(textstat.linsear_write_formula(text)),
                "formulas.gunning_fog":                  float(textstat.gunning_fog(text)),
                "aggregates.average_word_length":        float(lc / wc),
                "aggregates.average_sentence_length":    float(wc / sc),
                "aggregates.type_token_ratio":           float(len(set(text.lower().split())) / wc),
            }
        except Exception as e:
            print(f"[readability] {e}")
            return {}

    # ══════════════════════════════════════════════════════════════════════════
    # LINGFEAT
    # ══════════════════════════════════════════════════════════════════════════

    def _extract_lingfeat(self, text: str) -> Dict[str, float]:
        try:
            LF = lf_extractor_mod.pass_text(text)
            LF.preprocess()
            raw = {}
            for fn in [LF.WoKF_,LF.WBKF_,LF.OSKF_,LF.EnDF_,LF.EnGF_,
                        LF.PhrF_,LF.TrSF_,LF.POSF_,LF.TTRF_,LF.VarF_,
                        LF.PsyF_,LF.WorF_,LF.ShaF_,LF.TraF_]:
                raw.update(fn())
            return {k: float(v) if v is not None else 0.0 for k,v in raw.items()}
        except Exception as e:
            print(f"[lingfeat] {e}")
            return {}

    # ══════════════════════════════════════════════════════════════════════════
    # COHMETRIX CLI
    # ══════════════════════════════════════════════════════════════════════════

    def _extract_cohmetrix_cli(self, text: str) -> Dict[str, float]:
        if not os.path.exists(COH_METRIX_CLI):
            return {}
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".txt",
                    dir=COH_OUTPUT_DIR, delete=False, encoding="utf-8") as f:
                f.write(text); tmp = f.name
            subprocess.run([COH_METRIX_CLI, tmp, COH_OUTPUT_DIR],
                           check=True, capture_output=True, timeout=60)
            csv = tmp + ".csv"
            raw = {}
            if os.path.exists(csv):
                with open(csv, encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split(",")
                        if len(parts) == 2:
                            try: raw[parts[0].strip()] = float(parts[1].strip())
                            except ValueError: pass
                os.remove(csv)
            os.remove(tmp)
            return raw
        except Exception as e:
            print(f"[cohmetrix_cli] {e}")
            return {}

    # ══════════════════════════════════════════════════════════════════════════
    # COHMETRIX MANUAL
    # ══════════════════════════════════════════════════════════════════════════

    def _extract_cohmetrix_manual(self, text: str) -> Dict[str, float]:
        out = {}
        out.update(self._calc_CRFCWOad(text))
        out.update(self._calc_CRFCWO1d(text))
        out.update(self._calc_LSA(text))
        out.update(self._calc_PCTEMPp(text))
        out.update(self._calc_PCCONNp(text))
        out.update(self._calc_PCSYNp(text))
        out.update(self._calc_CNCNeg(text))
        out.update(self._calc_WRDPRP3p(text))
        out.update(self._calc_PCREFp(text))
        return out

    def _calc_CRFCWOad(self, text: str) -> Dict[str, float]:
        """Content word overlap (Jaccard) across adjacent sentences."""
        try:
            doc  = self.nlp(text)
            sents = list(doc.sents)
            if len(sents) < 2: return {"CRFCWOad": 0.0}
            overlaps = []
            for i in range(len(sents)-1):
                s1 = {t.lemma_.lower() for t in sents[i]   if t.pos_ in CONTENT_POS and t.is_alpha}
                s2 = {t.lemma_.lower() for t in sents[i+1] if t.pos_ in CONTENT_POS and t.is_alpha}
                u  = s1 | s2
                if u: overlaps.append(len(s1&s2)/len(u))
            return {"CRFCWOad": float(np.mean(overlaps)) if overlaps else 0.0}
        except Exception as e:
            print(f"[CRFCWOad] {e}"); return {"CRFCWOad": 0.0}

    def _calc_CRFCWO1d(self, text: str) -> Dict[str, float]:
        """Content word overlap between sentence 1 and all others (discourse-level)."""
        try:
            doc  = self.nlp(text)
            sents = list(doc.sents)
            if len(sents) < 2: return {"CRFCWO1d": 0.0}
            s1 = {t.lemma_.lower() for t in sents[0] if t.pos_ in CONTENT_POS and t.is_alpha}
            overlaps = []
            for s in sents[1:]:
                s2 = {t.lemma_.lower() for t in s if t.pos_ in CONTENT_POS and t.is_alpha}
                u  = s1 | s2
                if u: overlaps.append(len(s1&s2)/len(u))
            return {"CRFCWO1d": float(np.mean(overlaps)) if overlaps else 0.0}
        except Exception as e:
            print(f"[CRFCWO1d] {e}"); return {"CRFCWO1d": 0.0}

    def _calc_LSA(self, text: str) -> Dict[str, float]:
        """SBERT-based LSA proxies: LSAGNd, LSASSpd, LSASS1d."""
        defaults = {"LSAGNd":0.0,"LSASSpd":0.0,"LSASS1d":0.0}
        if self.sbert is None: return defaults
        try:
            sents = sent_tokenize(text)
            if len(sents) < 2: return defaults
            emb  = self.sbert.encode(sents, normalize_embeddings=True)
            sims = [float(cosine_similarity(emb[i].reshape(1,-1),
                                             emb[i+1].reshape(1,-1))[0][0])
                    for i in range(len(emb)-1)]
            # LSASS1d: similarity of sentence 1 to all others
            s1_sims = [float(cosine_similarity(emb[0].reshape(1,-1),
                                                emb[j].reshape(1,-1))[0][0])
                       for j in range(1, len(emb))]
            return {
                "LSAGNd":  round(float(1.0 - np.mean(sims)), 6),
                "LSASSpd": round(float(np.std(sims)),         6),
                "LSASS1d": round(float(np.mean(s1_sims)),     6),
            }
        except Exception as e:
            print(f"[LSA] {e}"); return defaults

    def _calc_PCTEMPp(self, text: str) -> Dict[str, float]:
        """Temporal connective incidence per 1000 words."""
        try:
            tokens = word_tokenize(text.lower())
            total  = max(len(tokens), 1)
            count  = sum(1 for t in tokens if t in TEMPORAL_CONNECTIVES)
            return {"PCTEMPp": float(count/total*1000)}
        except Exception as e:
            print(f"[PCTEMPp] {e}"); return {"PCTEMPp": 0.0}

    def _calc_PCCONNp(self, text: str) -> Dict[str, float]:
        """All connective incidence per 1000 words."""
        try:
            tokens = word_tokenize(text.lower())
            total  = max(len(tokens), 1)
            count  = sum(1 for t in tokens if t in CONNECTIVES_ALL)
            return {"PCCONNp": float(count/total*1000)}
        except Exception as e:
            print(f"[PCCONNp] {e}"); return {"PCCONNp": 0.0}

    def _calc_PCSYNp(self, text: str) -> Dict[str, float]:
        """
        PCSYNp: Syntactic complexity percentage.
        Proportion of sentences containing at least one subordinate clause.
        """
        try:
            doc   = self.nlp(text)
            sents = list(doc.sents)
            if not sents: return {"PCSYNp": 0.0}
            sub_deps = {"mark","advcl","relcl","csubj","csubjpass","acl"}
            complex_sents = sum(1 for s in sents if any(t.dep_ in sub_deps for t in s))
            return {"PCSYNp": float(complex_sents/len(sents)*100)}
        except Exception as e:
            print(f"[PCSYNp] {e}"); return {"PCSYNp": 0.0}

    def _calc_CNCNeg(self, text: str) -> Dict[str, float]:
        """Negative connective/word incidence per 1000 words."""
        try:
            tokens = word_tokenize(text.lower())
            total  = max(len(tokens), 1)
            count  = sum(1 for t in tokens if t in NEGATIVE_WORDS)
            return {"CNCNeg": float(count/total*1000)}
        except Exception as e:
            print(f"[CNCNeg] {e}"); return {"CNCNeg": 0.0}

    def _calc_WRDPRP3p(self, text: str) -> Dict[str, float]:
        """Third person pronoun incidence per 1000 words."""
        try:
            tokens = word_tokenize(text.lower())
            total  = max(len(tokens), 1)
            count  = sum(1 for t in tokens if t in THIRD_PERSON)
            return {"WRDPRP3p": float(count/total*1000)}
        except Exception as e:
            print(f"[WRDPRP3p] {e}"); return {"WRDPRP3p": 0.0}

    def _calc_PCREFp(self, text: str) -> Dict[str, float]:
        """Referential cohesion: % adjacent sentence pairs sharing a noun/pronoun."""
        try:
            doc   = self.nlp(text)
            sents = list(doc.sents)
            if len(sents) < 2: return {"PCREFp": 0.0}
            REF = {"NOUN","PROPN","PRON"}
            hits = sum(
                1 for i in range(len(sents)-1)
                if {t.lemma_.lower() for t in sents[i]   if t.pos_ in REF and t.is_alpha} &
                   {t.lemma_.lower() for t in sents[i+1] if t.pos_ in REF and t.is_alpha}
            )
            return {"PCREFp": float(hits/(len(sents)-1)*100)}
        except Exception as e:
            print(f"[PCREFp] {e}"); return {"PCREFp": 0.0}

    # ══════════════════════════════════════════════════════════════════════════
    # LFTK
    # ══════════════════════════════════════════════════════════════════════════

    def _extract_lftk(self, text: str) -> Dict[str, float]:
        try:
            doc = self.nlp(text)
            ext = lftk.Extractor(docs=doc)
            ext.customize(stop_words=True, punctuations=False, round_decimal=6)
            raw = ext.extract()
            return {k: float(v) if v is not None else 0.0 for k,v in raw.items()}
        except Exception as e:
            print(f"[lftk] {e}"); return {}


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import pickle

    sample = (
        "The government announced new policies to address climate change. "
        "These policies will affect industries across the country. "
        "Scientists who study environmental impacts believe that the changes "
        "will be significant. However, some economists argue that the costs "
        "may outweigh the benefits in the short term. Despite these concerns, "
        "officials remain committed to implementing the reforms by next year."
    )

    ext      = FS5Extractor()
    features = ext.extract(sample)
    print(f"\nTotal features extracted: {len(features)}")

    with open("artifacts/fs5_features.pkl", "rb") as f:
        fs5 = pickle.load(f)

    print(f"FS5 features needed : {len(fs5)}")
    missing = [k for k in fs5 if k not in features]
    print(f"Missing            : {len(missing)}")
    if missing:
        for k in missing: print(f"  {k}")
    else:
        print("All FS5 features covered!")