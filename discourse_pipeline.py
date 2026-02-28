# ============================================================
# COMPLETE DISCOURSE FEATURE EXTRACTION PIPELINE
# ============================================================

import re
import math
import textstat
import numpy as np
import pandas as pd
import subprocess
import os
from collections import Counter, defaultdict
from typing import Dict, List, Any, Optional

# NLP imports
import nltk
import spacy
import torch
import stanza
import lftk
from fastcoref import spacy_component
from sentence_transformers import SentenceTransformer
from nltk.tag import pos_tag_sents
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering

# ============================================================
# CONSTANTS
# ============================================================

PER_N_WORDS = 100
WORD = re.compile(r"\w+")

FIRST_PERSON = {"i", "me", "my", "mine", "we", "us", "our", "ours"}
THIRD_PERSON = {
    "he", "him", "his",
    "she", "her", "hers",
    "it", "its",
    "they", "them", "their", "theirs"
}

ROLES = ["S", "O", "X", "-"]
ROLE_PRIORITY = ["S", "O", "X"]

DEP_TO_ROLE = {
    "nsubj": "S",
    "nsubjpass": "S",
    "csubj": "S",
    "dobj": "O",
    "obj": "O",
    "iobj": "O",
    "pobj": "O",
}

# Coh-Metrix config
COH_METRIX_CLI_PATH = "CohMetrixCore/CohMetrixCoreCLI.exe"
COH_METRIX_OUTPUT_DIR = "cohmetrix_output"
TEMP_FILE = "temp_input.txt"

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def _normalize_per_n_words(value: float, text: str, n: int = PER_N_WORDS) -> float:
    """Normalize raw count to per-n-words density"""
    try:
        token_count = textstat.lexicon_count(text, removepunct=True)
        if token_count < 1:
            return 0.0
        return (value / token_count) * n
    except Exception:
        return 0.0

def get_mention_start(m) -> int:
    """Get start index of a mention"""
    return m.start if hasattr(m, "start") else m[0]

def get_mention_end(m) -> int:
    """Get end index of a mention"""
    return m.end if hasattr(m, "end") else m[1]

def get_grammatical_role(token) -> str:
    """Get grammatical role from dependency parse"""
    return DEP_TO_ROLE.get(token.dep_, "X")

def tree_height(tree) -> int:
    """Calculate height of constituency parse tree"""
    if not tree.children:
        return 1
    return 1 + max(tree_height(child) for child in tree.children)

def count_phrases(tree, label: str) -> int:
    """Count phrases of a specific type in parse tree"""
    count = 1 if tree.label == label else 0
    for child in tree.children:
        count += count_phrases(child, label)
    return count

def merge_roles(roles: List[str]) -> str:
    """Merge multiple roles into one based on priority"""
    for r in ROLE_PRIORITY:
        if r in roles:
            return r
    return "-"

# ============================================================
# COREFERENCE FEATURES
# ============================================================

def extract_coref_features(text: str, nlp) -> Dict[str, float]:
    """
    Extract coreference resolution features
    
    Features:
    - coref.num_chains: Number of coreference chains
    - coref.avg_refs_per_chain: Average mentions per chain
    - coref.avg_chain_span: Average sentence span of chains
    - coref.max_chain_span: Maximum sentence span
    - coref.long_span_chains: Chains spanning >=2 sentences
    - coref.avg_inference_distance: Average distance between mentions
    - coref.active_chains_per_word: Chains per word
    - coref.active_chains_per_entity: Chains per entity
    - coref.num_chains_norm: Normalized chain count
    - coref.long_span_chains_norm: Normalized long-span chains
    """
    doc = nlp(text)
    clusters = doc._.coref_clusters
    sentences = list(doc.sents)

    num_chains = len(clusters)
    total_mentions = sum(len(c) for c in clusters)

    if num_chains == 0:
        return {
            "coref.num_chains": 0.0,
            "coref.avg_refs_per_chain": 0.0,
            "coref.avg_chain_span": 0.0,
            "coref.max_chain_span": 0.0,
            "coref.long_span_chains": 0.0,
            "coref.avg_inference_distance": 0.0,
            "coref.active_chains_per_word": 0.0,
            "coref.active_chains_per_entity": 0.0,
            "coref.num_chains_norm": 0.0,
            "coref.long_span_chains_norm": 0.0
        }

    chain_spans = []
    inference_distances = []

    for cluster in clusters:
        mentions = sorted(cluster, key=get_mention_start)

        sent_ids = []
        for m in mentions:
            for i, sent in enumerate(sentences):
                if (get_mention_start(m) >= sent.start and 
                    get_mention_end(m) <= sent.end):
                    sent_ids.append(i)
                    break

        if sent_ids:
            span = max(sent_ids) - min(sent_ids)
            chain_spans.append(span)

        for i in range(1, len(mentions)):
            dist = get_mention_start(mentions[i]) - get_mention_end(mentions[i-1])
            inference_distances.append(dist)

    token_count = len(doc)

    features = {
        "coref.num_chains": float(num_chains),
        "coref.avg_refs_per_chain": float(total_mentions / num_chains),
        "coref.avg_chain_span": float(np.mean(chain_spans)) if chain_spans else 0.0,
        "coref.max_chain_span": float(max(chain_spans)) if chain_spans else 0.0,
        "coref.long_span_chains": float(sum(1 for s in chain_spans if s >= 2)),
        "coref.avg_inference_distance": float(np.mean(inference_distances)) if inference_distances else 0.0,
        "coref.active_chains_per_word": float(num_chains / token_count) if token_count else 0.0,
        "coref.active_chains_per_entity": float(num_chains / total_mentions) if total_mentions else 0.0,
    }
    
    features["coref.num_chains_norm"] = _normalize_per_n_words(num_chains, text)
    features["coref.long_span_chains_norm"] = _normalize_per_n_words(
        features["coref.long_span_chains"], text
    )
    
    return features

# ============================================================
# ENTITY GRID FEATURES
# ============================================================

def extract_entities_with_roles(doc, mention_to_entity: Dict) -> List[Dict]:
    """Extract entities with grammatical roles per sentence"""
    sentences_entities = []

    for sent in doc.sents:
        sent_entities = {}

        for token in sent:
            if not token.is_alpha:
                continue

            span_key = (token.i, token.i + 1)

            if span_key in mention_to_entity:
                entity_id = mention_to_entity[span_key]
            elif token.pos_ in {"NOUN", "PROPN"}:
                entity_id = token.lemma_.lower()
            else:
                continue

            role = get_grammatical_role(token)

            if entity_id not in sent_entities or role in {"S", "O"}:
                sent_entities[entity_id] = role

        sentences_entities.append(sent_entities)

    return sentences_entities

def build_entity_grid(sentences_entities: List[Dict]) -> pd.DataFrame:
    """Build entity grid from sentences_entities"""
    entities = sorted(
        {ent for sent in sentences_entities for ent in sent.keys()}
    )
    grid = []
    for sent in sentences_entities:
        row = [sent.get(ent, "-") for ent in entities]
        grid.append(row)
    return pd.DataFrame(grid, columns=entities)

def get_role_transitions(grid: pd.DataFrame) -> List[tuple]:
    """Extract role transitions across adjacent sentences"""
    transitions = []
    for col in grid.columns:
        roles = grid[col].tolist()
        for i in range(len(roles) - 1):
            transitions.append((roles[i], roles[i + 1]))
    return transitions

def transition_probabilities(grid: pd.DataFrame) -> Dict[str, float]:
    """Calculate transition probabilities between roles"""
    transitions = get_role_transitions(grid)
    total = len(transitions)
    counter = Counter(transitions)
    probs = {}
    for r1 in ROLES:
        for r2 in ROLES:
            key = f"eg.trans_{r1}_{r2}"
            probs[key] = float(counter.get((r1, r2), 0) / total) if total > 0 else 0.0
    return probs

def entity_persistence_features(grid: pd.DataFrame) -> Dict[str, float]:
    """Calculate entity persistence features"""
    lengths = []
    for col in grid.columns:
        roles = grid[col].tolist()
        current = 0
        for r in roles:
            if r != "-":
                current += 1
            else:
                if current > 0:
                    lengths.append(current)
                current = 0
        if current > 0:
            lengths.append(current)
    
    return {
        "eg.avg_entity_persistence": float(sum(lengths) / len(lengths)) if lengths else 0.0,
        "eg.max_entity_persistence": float(max(lengths)) if lengths else 0.0,
    }

def grid_density_features(grid: pd.DataFrame) -> Dict[str, float]:
    """Calculate grid density and sparsity"""
    total_cells = grid.shape[0] * grid.shape[1]
    non_empty = (grid != "-").sum().sum()
    density = non_empty / total_cells if total_cells > 0 else 0.0
    return {
        "eg.grid_density": float(density),
        "eg.grid_sparsity": float(1 - density),
    }

def role_entropy_feature(grid: pd.DataFrame) -> Dict[str, float]:
    """Calculate role entropy"""
    counts = Counter()
    for col in grid.columns:
        counts.update(grid[col].tolist())
    total = sum(counts.values())
    entropy = 0.0
    for r in ROLES:
        p = counts.get(r, 0) / total if total > 0 else 0.0
        if p > 0:
            entropy -= p * math.log(p)
    return {"eg.role_entropy": float(entropy)}

def extract_entity_grid_features(text: str, nlp, sbert_model) -> Dict[str, float]:
    """
    Complete entity grid feature extraction pipeline
    
    Features:
    - eg.trans_S_S, eg.trans_S_O, etc.: Role transition probabilities
    - eg.avg_entity_persistence: Average entity lifespan
    - eg.max_entity_persistence: Maximum entity lifespan
    - eg.grid_density: Density of non-empty cells
    - eg.grid_sparsity: 1 - density
    - eg.role_entropy: Entropy of role distribution
    """
    if not text or not str(text).strip():
        base_features = {f"eg.trans_{r1}_{r2}": 0.0 for r1 in ROLES for r2 in ROLES}
        base_features.update({
            "eg.avg_entity_persistence": 0.0,
            "eg.max_entity_persistence": 0.0,
            "eg.grid_density": 0.0,
            "eg.grid_sparsity": 1.0,
            "eg.role_entropy": 0.0
        })
        return base_features
    
    doc = nlp(text)
    
    # Get coreference clusters
    clusters = getattr(doc._, "coref_clusters", None)
    mention_to_entity = {}
    if clusters:
        for cluster_id, cluster in enumerate(clusters):
            for mention in cluster:
                if hasattr(mention, "start") and hasattr(mention, "end"):
                    start, end = mention.start, mention.end
                elif isinstance(mention, tuple) and len(mention) >= 2:
                    start, end = mention[0], mention[1]
                else:
                    continue
                mention_to_entity[(start, end)] = f"E{cluster_id}"
    
    # Extract entities with roles
    sentences_entities = extract_entities_with_roles(doc, mention_to_entity)
    
    # Build entity grid
    entity_grid = build_entity_grid(sentences_entities)
    
    # Collapse grid using SBERT
    if entity_grid.shape[1] > 0:
        entity_labels = list(entity_grid.columns)
        embeddings = sbert_model.encode(entity_labels, normalize_embeddings=True)
        
        # Cluster entities
        similarity_threshold = 0.75
        distance_threshold = 1 - similarity_threshold
        clustering = AgglomerativeClustering(
            n_clusters=None,
            metric="cosine",
            linkage="average",
            distance_threshold=distance_threshold
        )
        cluster_ids = clustering.fit_predict(embeddings)
        
        # Build cluster map
        cluster_map = {}
        for label, cid in zip(entity_labels, cluster_ids):
            cluster_map.setdefault(cid, []).append(label)
        
        # Collapse grid
        new_columns = {}
        for cluster_entities in cluster_map.values():
            col_name = "_".join(cluster_entities)
            merged_col = []
            for _, row in entity_grid.iterrows():
                roles = [row[e] for e in cluster_entities if e in entity_grid.columns]
                merged_col.append(merge_roles(roles))
            new_columns[col_name] = merged_col
        collapsed_grid = pd.DataFrame(new_columns)
    else:
        collapsed_grid = entity_grid
    
    # Extract features
    features = {}
    features.update(transition_probabilities(collapsed_grid))
    features.update(entity_persistence_features(collapsed_grid))
    features.update(grid_density_features(collapsed_grid))
    features.update(role_entropy_feature(collapsed_grid))
    
    return features

# ============================================================
# LEXICAL CHAIN FEATURES
# ============================================================

def extract_noun_phrases(text: str, nlp) -> List[Dict]:
    """Extract noun phrases from text"""
    if not text or not str(text).strip():
        return []
    doc = nlp(text)
    noun_phrases = []
    for chunk in doc.noun_chunks:
        noun_phrases.append({
            "text": chunk.text,
            "start": chunk.start,
            "end": chunk.end
        })
    return noun_phrases

def embed_noun_phrases(noun_phrases: List[Dict], sbert_model) -> List[Dict]:
    """Embed noun phrases using SBERT"""
    texts = [np["text"] for np in noun_phrases]
    if not texts:
        return noun_phrases
    embeddings = sbert_model.encode(texts, normalize_embeddings=True)
    for i, emb in enumerate(embeddings):
        noun_phrases[i]["embedding"] = emb
    return noun_phrases

def build_lexical_chains(noun_phrases: List[Dict], threshold: float = 0.75) -> List[List]:
    """Build lexical chains using cosine similarity"""
    chains = []
    for np_ in noun_phrases:
        placed = False
        for chain in chains:
            last_np = chain[-1]
            sim = cosine_similarity(
                np_["embedding"].reshape(1, -1),
                last_np["embedding"].reshape(1, -1)
            )[0][0]
            if sim >= threshold:
                chain.append(np_)
                placed = True
                break
        if not placed:
            chains.append([np_])
    return chains

def lexical_chain_features(text: str, nlp, sbert_model) -> Dict[str, float]:
    """
    Extract lexical chain features
    
    Features:
    - lex.num_lexical_chains: Number of lexical chains (normalized)
    - lex.avg_chain_length: Average chain length
    - lex.avg_chain_span: Average chain span in tokens
    - lex.long_span_chains: Long-span chains count (normalized)
    - lex.active_chains_per_word: Chains per word
    - lex.active_chains_per_entity: Chains per entity
    """
    noun_phrases = extract_noun_phrases(text, nlp)
    noun_phrases = embed_noun_phrases(noun_phrases, sbert_model)
    chains = build_lexical_chains(noun_phrases)
    
    total_words = textstat.lexicon_count(text, removepunct=True)
    
    if not chains:
        return {
            "lex.num_lexical_chains": 0.0,
            "lex.avg_chain_length": 0.0,
            "lex.avg_chain_span": 0.0,
            "lex.long_span_chains": 0.0,
            "lex.active_chains_per_word": 0.0,
            "lex.active_chains_per_entity": 0.0
        }
    
    chain_lengths = [len(c) for c in chains]
    chain_spans = [
        c[-1]["start"] - c[0]["start"] + 1
        for c in chains
    ]
    
    num_chains = len(chains)
    avg_length = np.mean(chain_lengths)
    avg_span = np.mean(chain_spans)
    long_span = sum(1 for s in chain_spans if s > avg_span)
    total_entities = sum(chain_lengths)
    
    return {
        "lex.num_lexical_chains": _normalize_per_n_words(num_chains, text),
        "lex.avg_chain_length": float(avg_length),
        "lex.avg_chain_span": float(avg_span),
        "lex.long_span_chains": _normalize_per_n_words(long_span, text),
        "lex.active_chains_per_word": float(num_chains / total_words) if total_words else 0.0,
        "lex.active_chains_per_entity": float(num_chains / total_entities) if total_entities else 0.0
    }

# ============================================================
# CONSTITUENCY FEATURES
# ============================================================

def extract_constituency_features(text: str, stanza_nlp) -> Dict[str, float]:
    """
    Extract constituency parse features
    
    Features:
    - const.avg_tree_height: Average parse tree height
    - const.vp_density: Verb phrase density (normalized)
    - const.np_density: Noun phrase density (normalized)
    - const.adjp_density: Adjective phrase density (normalized)
    - const.advp_density: Adverb phrase density (normalized)
    - const.sbar_density: Subordinate clause density (normalized)
    """
    if not text or not str(text).strip():
        return {
            "const.avg_tree_height": 0.0,
            "const.vp_density": 0.0,
            "const.np_density": 0.0,
            "const.adjp_density": 0.0,
            "const.advp_density": 0.0,
            "const.sbar_density": 0.0,
        }
    
    doc = stanza_nlp(text)
    total_height = 0
    vp = np_ = adjp = advp = sbar = 0
    sentence_count = 0
    
    for sent in doc.sentences:
        if not sent.constituency:
            continue
        tree = sent.constituency
        sentence_count += 1
        total_height += tree_height(tree)
        vp += count_phrases(tree, "VP")
        np_ += count_phrases(tree, "NP")
        adjp += count_phrases(tree, "ADJP")
        advp += count_phrases(tree, "ADVP")
        sbar += count_phrases(tree, "SBAR")
    
    if sentence_count == 0:
        return {
            "const.avg_tree_height": 0.0,
            "const.vp_density": 0.0,
            "const.np_density": 0.0,
            "const.adjp_density": 0.0,
            "const.advp_density": 0.0,
            "const.sbar_density": 0.0,
        }
    
    return {
        "const.avg_tree_height": float(total_height / sentence_count),
        "const.vp_density": _normalize_per_n_words(vp, text),
        "const.np_density": _normalize_per_n_words(np_, text),
        "const.adjp_density": _normalize_per_n_words(adjp, text),
        "const.advp_density": _normalize_per_n_words(advp, text),
        "const.sbar_density": _normalize_per_n_words(sbar, text),
    }

# ============================================================
# ENTITY FEATURES
# ============================================================

def extract_entity_features(text: str, nlp) -> Dict[str, float]:
    """
    Extract entity-related features
    
    Features:
    - ent.total_entities: Total entities (named + general)
    - ent.total_unique_entities: Unique entities
    - ent.avg_entities_per_sentence: Entities per sentence
    - ent.avg_unique_entities_per_sentence: Unique entities per sentence
    - ent.avg_named_entities_per_sentence: Named entities per sentence
    - ent.percent_named_entities: Percentage of named entities
    - ent.percent_general_nouns: Percentage of general nouns
    - ent.entity_density: Entities per 100 words
    - ent.unique_entity_density: Unique entities per 100 words
    - ent.named_entity_density: Named entities per 100 words
    """
    if not text or not str(text).strip():
        return {
            "ent.total_entities": 0.0,
            "ent.total_unique_entities": 0.0,
            "ent.avg_entities_per_sentence": 0.0,
            "ent.avg_unique_entities_per_sentence": 0.0,
            "ent.avg_named_entities_per_sentence": 0.0,
            "ent.percent_named_entities": 0.0,
            "ent.percent_general_nouns": 0.0,
            "ent.entity_density": 0.0,
            "ent.unique_entity_density": 0.0,
            "ent.named_entity_density": 0.0,
        }
    
    doc = nlp(text)
    sentences = sent_tokenize(text)
    
    named_entities = list(doc.ents)
    named_entity_count = len(named_entities)
    
    general_nouns = [
        token for token in doc
        if token.pos_ in {"NOUN", "PROPN"} and token.ent_type_ == ""
    ]
    
    total_entities = named_entity_count + len(general_nouns)
    unique_entities = set(
        [ent.text.lower() for ent in named_entities] +
        [token.lemma_.lower() for token in general_nouns]
    )
    
    num_sentences = len(sentences) if sentences else 1
    
    return {
        "ent.total_entities": float(total_entities),
        "ent.total_unique_entities": float(len(unique_entities)),
        "ent.avg_entities_per_sentence": float(total_entities / num_sentences),
        "ent.avg_unique_entities_per_sentence": float(len(unique_entities) / num_sentences),
        "ent.avg_named_entities_per_sentence": float(named_entity_count / num_sentences),
        "ent.percent_named_entities": float(
            named_entity_count / total_entities * 100 if total_entities > 0 else 0
        ),
        "ent.percent_general_nouns": float(
            len(general_nouns) / total_entities * 100 if total_entities > 0 else 0
        ),
        "ent.entity_density": _normalize_per_n_words(total_entities, text),
        "ent.unique_entity_density": _normalize_per_n_words(len(unique_entities), text),
        "ent.named_entity_density": _normalize_per_n_words(named_entity_count, text),
    }

# ============================================================
# POS FEATURES
# ============================================================

def extract_pos_features(text: str, nlp) -> Dict[str, float]:
    """
    Extract part-of-speech features
    
    Features:
    - pos.noun_density: Noun count per 100 words
    - pos.unique_noun_density: Unique noun types per 100 words
    - pos.preposition_density: Preposition count per 100 words
    - pos.adjective_density: Adjective count per 100 words
    - pos.adverb_density: Adverb count per 100 words
    - pos.first_person_pronoun_density: First-person pronouns per 100 words
    - pos.third_person_pronoun_density: Third-person pronouns per 100 words
    """
    if not text or not str(text).strip():
        return {
            "pos.noun_density": 0.0,
            "pos.unique_noun_density": 0.0,
            "pos.preposition_density": 0.0,
            "pos.adjective_density": 0.0,
            "pos.adverb_density": 0.0,
            "pos.first_person_pronoun_density": 0.0,
            "pos.third_person_pronoun_density": 0.0
        }
    
    doc = nlp(text)
    tokens = [t for t in doc if t.is_alpha]
    
    noun_count = sum(1 for t in tokens if t.pos_ == "NOUN")
    unique_nouns = {t.lemma_.lower() for t in tokens if t.pos_ == "NOUN"}
    prep_count = sum(1 for t in tokens if t.pos_ == "ADP")
    adj_count = sum(1 for t in tokens if t.pos_ == "ADJ")
    adv_count = sum(1 for t in tokens if t.pos_ == "ADV")
    
    first_pron_count = sum(
        1 for t in tokens
        if t.pos_ == "PRON" and t.text.lower() in FIRST_PERSON
    )
    third_pron_count = sum(
        1 for t in tokens
        if t.pos_ == "PRON" and t.text.lower() in THIRD_PERSON
    )
    
    return {
        "pos.noun_density": _normalize_per_n_words(noun_count, text),
        "pos.unique_noun_density": _normalize_per_n_words(len(unique_nouns), text),
        "pos.preposition_density": _normalize_per_n_words(prep_count, text),
        "pos.adjective_density": _normalize_per_n_words(adj_count, text),
        "pos.adverb_density": _normalize_per_n_words(adv_count, text),
        "pos.first_person_pronoun_density": _normalize_per_n_words(first_pron_count, text),
        "pos.third_person_pronoun_density": _normalize_per_n_words(third_pron_count, text),
    }

# ============================================================
# READABILITY AND AGGREGATE FEATURES
# ============================================================

def extract_readability_features(text: str) -> Dict[str, float]:
    """
    Extract readability and aggregate features
    
    Aggregate features (normalized per 100 words):
    - aggregates.syllable_count: Total syllables
    - aggregates.lexicon_count: Total words (raw count)
    - aggregates.sentence_count: Sentence count
    - aggregates.char_count: Character count
    - aggregates.letter_count: Letter count
    - aggregates.polysyllabcount: Polysyllabic words (>=3 syllables)
    - aggregates.monosyllabcount: Monosyllabic words
    - aggregates.average_word_length: Average letters per word
    - aggregates.type_token_ratio: Unique words / total words
    - aggregates.average_sentence_length: Words per sentence
    - aggregates.average_verbs_per_sentence: Verbs per sentence
    - aggregates.average_pronouns_per_sentence: Pronouns per sentence
    
    Readability formulas:
    - formulas.flesch_reading_ease
    - formulas.flesch_kincaid_grade
    - formulas.smog_index
    - formulas.automated_readability_index
    - formulas.coleman_liau_index
    - formulas.dale_chall_readability_score
    - formulas.linsear_write_formula
    - formulas.gunning_fog
    """
    if not text or not str(text).strip():
        return {
            'aggregates.syllable_count': 0.0,
            'aggregates.lexicon_count': 0.0,
            'aggregates.sentence_count': 0.0,
            'aggregates.char_count': 0.0,
            'aggregates.letter_count': 0.0,
            'aggregates.polysyllabcount': 0.0,
            'aggregates.monosyllabcount': 0.0,
            'aggregates.average_word_length': 0.0,
            'aggregates.type_token_ratio': 0.0,
            'aggregates.average_sentence_length': 0.0,
            'aggregates.average_verbs_per_sentence': 0.0,
            'aggregates.average_pronouns_per_sentence': 0.0,
            'formulas.flesch_reading_ease': 0.0,
            'formulas.flesch_kincaid_grade': 0.0,
            'formulas.smog_index': 0.0,
            'formulas.automated_readability_index': 0.0,
            'formulas.coleman_liau_index': 0.0,
            'formulas.dale_chall_readability_score': 0.0,
            'formulas.linsear_write_formula': 0.0,
            'formulas.gunning_fog': 0.0
        }
    
    # Aggregate counts
    lexicon_count = textstat.lexicon_count(text, removepunct=True)
    sentence_count = textstat.sentence_count(text)
    char_count = textstat.char_count(text, ignore_spaces=True)
    letter_count = textstat.letter_count(text, ignore_spaces=True)
    polysyllabcount = textstat.polysyllabcount(text)
    monosyllabcount = textstat.monosyllabcount(text)
    syllable_count = textstat.syllable_count(text)
    
    # Token-based metrics
    words = WORD.findall(text)
    number_of_types = len(set(words))
    
    # Verb and pronoun counts
    tokenized_text = [word_tokenize(sentence) for sentence in sent_tokenize(text)]
    number_of_verbs = 0
    number_of_pronouns = 0
    
    if tokenized_text:
        tagged_text = pos_tag_sents(tokenized_text, tagset="universal")
        for tagged_sentence in tagged_text:
            for tagged_word in tagged_sentence:
                if tagged_word[1] == "VERB":
                    number_of_verbs += 1
                elif tagged_word[1] == "PRON":
                    number_of_pronouns += 1
    
    return {
        # Aggregates (normalized)
        'aggregates.syllable_count': _normalize_per_n_words(syllable_count, text),
        'aggregates.lexicon_count': float(lexicon_count),
        'aggregates.sentence_count': _normalize_per_n_words(sentence_count, text),
        'aggregates.char_count': _normalize_per_n_words(char_count, text),
        'aggregates.letter_count': _normalize_per_n_words(letter_count, text),
        'aggregates.polysyllabcount': _normalize_per_n_words(polysyllabcount, text),
        'aggregates.monosyllabcount': _normalize_per_n_words(monosyllabcount, text),
        'aggregates.average_word_length': float(letter_count / lexicon_count) if lexicon_count else 0.0,
        'aggregates.type_token_ratio': float(number_of_types / lexicon_count) if lexicon_count else 0.0,
        'aggregates.average_sentence_length': float(lexicon_count / sentence_count) if sentence_count else 0.0,
        'aggregates.average_verbs_per_sentence': float(number_of_verbs / sentence_count) if sentence_count else 0.0,
        'aggregates.average_pronouns_per_sentence': float(number_of_pronouns / sentence_count) if sentence_count else 0.0,
        
        # Formulas
        'formulas.flesch_reading_ease': float(textstat.flesch_reading_ease(text)),
        'formulas.flesch_kincaid_grade': float(textstat.flesch_kincaid_grade(text)),
        'formulas.smog_index': float(textstat.smog_index(text)),
        'formulas.automated_readability_index': float(textstat.automated_readability_index(text)),
        'formulas.coleman_liau_index': float(textstat.coleman_liau_index(text)),
        'formulas.dale_chall_readability_score': float(textstat.dale_chall_readability_score(text)),
        'formulas.linsear_write_formula': float(textstat.linsear_write_formula(text)),
        'formulas.gunning_fog': float(textstat.gunning_fog(text))
    }

# ============================================================
# LFTK FEATURES - FIXED VERSION
# ============================================================

def extract_lftk_features(text: str, spacy_nlp) -> Dict[str, float]:
    """
    Extract LFTK (Language Feature Toolkit) features
    
    This includes various linguistic features like:
    - Lexical diversity measures
    - Syntactic complexity measures
    - Readability scores
    - Part-of-speech distributions
    """
    if not text or not str(text).strip():
        return {}
    
    try:
        doc = spacy_nlp(text)
        
        # Initialize extractor with the doc
        extractor = lftk.Extractor(docs=doc)
        extractor.customize(stop_words=True, punctuations=False, round_decimal=3)
        
        # Extract features
        features = extractor.extract()
        
        # prefix to avoid collisions and ensure float values
        return {f"lftk.{k}": float(v) for k, v in features.items()}
    except Exception as e:
        print(f"LFTK extraction error: {e}")
        return {}

# ============================================================
# COH-METRIX FEATURES
# ============================================================

def extract_cohmetrix_features(
    text: str, 
    cli_path: str = COH_METRIX_CLI_PATH,
    output_dir: str = COH_METRIX_OUTPUT_DIR,
    temp_file: str = TEMP_FILE
) -> Dict[str, float]:
    """
    Extract Coh-Metrix features
    
    Coh-Metrix provides over 100 indices of:
    - Cohesion (referential, latent semantic analysis)
    - Readability
    - Syntactic complexity
    - Word information
    - Connectives
    - etc.
    """
    if not os.path.exists(cli_path) or not text or not str(text).strip():
        return {}
    
    # Create output directory if needed
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Write text to temp file
    temp_file_path = os.path.join(output_dir, temp_file)
    with open(temp_file_path, "w", encoding="utf-8") as f:
        f.write(text)
    
    # Run Coh-Metrix CLI
    try:
        subprocess.run([cli_path, temp_file_path, output_dir], check=True, capture_output=True, timeout=30)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return {}
    
    # Parse output
    features = {}
    output_file = os.path.join(output_dir, os.path.basename(temp_file_path) + ".csv")
    
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) == 2:
                    try:
                        features["cm." + parts[0]] = float(parts[1])
                    except (ValueError, TypeError):
                        pass
    
    # Clean up temp files
    try:
        os.remove(temp_file_path)
        if os.path.exists(output_file):
            os.remove(output_file)
    except OSError:
        pass
    
    return features

# ============================================================
# MASTER FEATURE EXTRACTION FUNCTION
# ============================================================

def extract_all_features(
    text: str,
    spacy_nlp,
    stanza_nlp,
    sbert_model,
    include_lftk: bool = True,
    include_cohmetrix: bool = False,
    cohmetrix_cli_path: Optional[str] = None
) -> Dict[str, float]:
    """
    Extract ALL discourse features from text
    
    This combines features from:
    1. Coreference resolution (~10 features)
    2. Entity grid (~20 features)
    3. Lexical chains (~6 features)
    4. Constituency parsing (~6 features)
    5. Entity analysis (~10 features)
    6. POS tagging (~7 features)
    7. Readability/aggregates (~20 features)
    8. LFTK (optional, ~50+ features)
    9. Coh-Metrix (optional, ~100+ features)
    
    Total: 80+ base features, 200+ with optional modules
    """
    
    features = {}
    
    # Core discourse features
    features.update(extract_coref_features(text, spacy_nlp))
    features.update(extract_entity_grid_features(text, spacy_nlp, sbert_model))
    features.update(lexical_chain_features(text, spacy_nlp, sbert_model))
    features.update(extract_constituency_features(text, stanza_nlp))
    features.update(extract_entity_features(text, spacy_nlp))
    features.update(extract_pos_features(text, spacy_nlp))
    features.update(extract_readability_features(text))
    
    # Optional: LFTK features
    if include_lftk:
        features.update(extract_lftk_features(text, spacy_nlp))
    
    # Optional: Coh-Metrix features
    if include_cohmetrix and cohmetrix_cli_path and os.path.exists(cohmetrix_cli_path):
        features.update(extract_cohmetrix_features(text, cli_path=cohmetrix_cli_path))
    
    return features

# ============================================================
# FEATURE SUMMARY FUNCTION
# ============================================================

def get_feature_summary() -> Dict[str, List[str]]:
    """Get summary of all feature categories and their names"""
    return {
        "coreference": [
            "coref.num_chains",
            "coref.avg_refs_per_chain",
            "coref.avg_chain_span",
            "coref.max_chain_span",
            "coref.long_span_chains",
            "coref.avg_inference_distance",
            "coref.active_chains_per_word",
            "coref.active_chains_per_entity",
            "coref.num_chains_norm",
            "coref.long_span_chains_norm"
        ],
        "entity_grid": [
            "eg.trans_S_S", "eg.trans_S_O", "eg.trans_S_X", "eg.trans_S_-",
            "eg.trans_O_S", "eg.trans_O_O", "eg.trans_O_X", "eg.trans_O_-",
            "eg.trans_X_S", "eg.trans_X_O", "eg.trans_X_X", "eg.trans_X_-",
            "eg.trans_-_S", "eg.trans_-_O", "eg.trans_-_X", "eg.trans_-_-",
            "eg.avg_entity_persistence",
            "eg.max_entity_persistence",
            "eg.grid_density",
            "eg.grid_sparsity",
            "eg.role_entropy"
        ],
        "lexical_chains": [
            "lex.num_lexical_chains",
            "lex.avg_chain_length",
            "lex.avg_chain_span",
            "lex.long_span_chains",
            "lex.active_chains_per_word",
            "lex.active_chains_per_entity"
        ],
        "constituency": [
            "const.avg_tree_height",
            "const.vp_density",
            "const.np_density",
            "const.adjp_density",
            "const.advp_density",
            "const.sbar_density"
        ],
        "entity_analysis": [
            "ent.total_entities",
            "ent.total_unique_entities",
            "ent.avg_entities_per_sentence",
            "ent.avg_unique_entities_per_sentence",
            "ent.avg_named_entities_per_sentence",
            "ent.percent_named_entities",
            "ent.percent_general_nouns",
            "ent.entity_density",
            "ent.unique_entity_density",
            "ent.named_entity_density"
        ],
        "pos_tagging": [
            "pos.noun_density",
            "pos.unique_noun_density",
            "pos.preposition_density",
            "pos.adjective_density",
            "pos.adverb_density",
            "pos.first_person_pronoun_density",
            "pos.third_person_pronoun_density"
        ],
        "aggregates": [
            "aggregates.syllable_count",
            "aggregates.lexicon_count",
            "aggregates.sentence_count",
            "aggregates.char_count",
            "aggregates.letter_count",
            "aggregates.polysyllabcount",
            "aggregates.monosyllabcount",
            "aggregates.average_word_length",
            "aggregates.type_token_ratio",
            "aggregates.average_sentence_length",
            "aggregates.average_verbs_per_sentence",
            "aggregates.average_pronouns_per_sentence"
        ],
        "readability": [
            "formulas.flesch_reading_ease",
            "formulas.flesch_kincaid_grade",
            "formulas.smog_index",
            "formulas.automated_readability_index",
            "formulas.coleman_liau_index",
            "formulas.dale_chall_readability_score",
            "formulas.linsear_write_formula",
            "formulas.gunning_fog"
        ]
    }