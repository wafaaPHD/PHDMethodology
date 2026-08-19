"""
OpenIE Reference-Free Evaluation
=================================
Judges system-extracted triples (aggressiveAllMyMethod.txt) directly
against their SOURCE SENTENCES (cnn.txt) -- no gold/annotated triple
file is used at all.

Since there's no gold triple to match against, "Precision / Recall / F1"
are redefined in a reference-free way:

  - Correctness  (a.k.a. Faithfulness) : does every word in the triple
    actually occur in the source sentence? (catches hallucinated /
    invented content). Computed per-triple with fuzzy token-set
    similarity, 0-100.
  - Completeness (a.k.a. Coverage)     : how much of the sentence's
    content is captured by the UNION of all triples extracted from
    that sentence. Computed per-sentence.
  - Minimality   (a.k.a. Conciseness)  : is the triple an atomic,
    minimal fact, or does it just copy a huge chunk of the sentence?
    Computed per-triple as 1 - (triple_length / sentence_length).

  "Precision" (P)  = average Correctness score
  "Recall"    (R)  = average Completeness score
  "F1"             = harmonic mean of P and R

A triple counts as "high-fidelity" if its Correctness score >= SIM_THRESHOLD.

Outputs (written to /mnt/user-data/outputs):
  1. per_sentence_scores.csv   -> coverage / faithfulness / F1 per sentence
  2. overall_scores.csv        -> micro + macro aggregated P/R/F1
  3. quality_judgment.csv      -> per-triple minimality / completeness / correctness
  4. quality_summary.csv       -> aggregated accuracy % per criterion
"""

import re
import csv
import os
from collections import defaultdict
from rapidfuzz import fuzz
SIM_THRESHOLD = 90  # correctness/faithfulness threshold (0-100) for "high fidelity"
STOPWORDS = {
        "a", "an", "the", "of", "in", "on", "at", "to", "for", "and", "or",
        "is", "was", "were", "are", "be", "been", "being", "it", "its",
        "this", "that", "these", "those", "as", "by", "with", "he", "she",
        "his", "her", "him", "they", "them", "their", "who", "which", "'s",
        ".", ",", "``", "''", "'", "-", "--",
    }
# --------------------------------------------------------------------------
# HELPERS
# --------------------------------------------------------------------------
def clean_text(text: str) -> str:
    text = text.replace("[", "").replace("]", "")
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def content_words(text: str) -> set:
    """Tokenize and drop stopwords/punctuation -> set of meaningful words."""
    tokens = re.findall(r"[a-z0-9\.\-']+", text.lower())
    return {t for t in tokens if t not in STOPWORDS and len(t) > 1}


def triple_to_string(subj: str, rel: str, obj: str) -> str:
    return clean_text(f"{subj} {rel} {obj}")


def faithfulness(triple_str: str, sentence_str: str) -> float:
    """token_set_ratio ignores extra words in the LONGER string, so this
    measures whether the triple's words are (fuzzily) a subset of the
    sentence's words -- i.e. the triple isn't inventing content."""
    return fuzz.token_set_ratio(triple_str, sentence_str)


# --------------------------------------------------------------------------
# PARSE SENTENCES (no gold annotation used)
# --------------------------------------------------------------------------
def parse_sentences(path):
    """Plain text file, one sentence per line, line number = sent_id (1-indexed)."""
    sentences = {}
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            text = line.rstrip("\n")
            if text.strip():
                sentences[i] = text.strip()
    return sentences


# --------------------------------------------------------------------------
# PARSE SYSTEM TRIPLES
# --------------------------------------------------------------------------
def count_stats(sentences: dict, sys_triples: dict) -> dict:
    """Basic counts: how many sentences and how many triples are involved,
    plus how many sentences actually have at least one extracted triple."""
    n_sentences = len(sentences)
    n_triples = sum(len(v) for v in sys_triples.values())
    n_sentences_with_triples = sum(1 for sid in sentences if len(sys_triples.get(sid, [])) > 0)
    n_sentences_without_triples = n_sentences - n_sentences_with_triples
    avg_triples_per_sentence = round(n_triples / n_sentences, 3) if n_sentences else 0.0
    return {
        "n_sentences": n_sentences,
        "n_triples": n_triples,
        "n_sentences_with_triples": n_sentences_with_triples,
        "n_sentences_without_triples": n_sentences_without_triples,
        "avg_triples_per_sentence": avg_triples_per_sentence,
    }


def parse_system(path):
    """Tab-separated: sent_id \t subject \t relation \t object"""
    sys_triples = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            cols = line.split("\t")
            if len(cols) < 4:
                continue
            sent_id = int(cols[0].strip())
            subj, rel, obj = cols[1].strip(), cols[2].strip(), cols[3].strip()
            triple_str = triple_to_string(subj, rel, obj)
            sys_triples[sent_id].append((subj, rel, obj, triple_str))
    return sys_triples


# --------------------------------------------------------------------------
# MAIN EVALUATION
# --------------------------------------------------------------------------
def evaluate(SENTENCES_PATH,SYS_PATH,OUT_DIR):
    os.makedirs(OUT_DIR, exist_ok=True)

    sentences = parse_sentences(SENTENCES_PATH)
    sys_triples = parse_system(SYS_PATH)
    stats = count_stats(sentences, sys_triples)

    per_sentence_rows = []
    quality_rows = []

    micro_correctness_sum, micro_n_triples = 0.0, 0
    micro_completeness_sum, micro_n_sentences = 0.0, 0
    macro_precisions, macro_recalls, macro_f1s = [], [], []

    all_sent_ids = sorted(set(list(sentences.keys()) + list(sys_triples.keys())))

    for sent_id in all_sent_ids:
        sentence_text = sentences.get(sent_id, "")
        sentence_clean = clean_text(sentence_text)
        sent_content_words = content_words(sentence_text)
        s_triples = sys_triples.get(sent_id, [])

        # ---- completeness: union of all triple words vs sentence content words ----
        union_words = set()
        for (_, _, _, triple_str) in s_triples:
            union_words |= content_words(triple_str)
        completeness_score = (
            round(len(union_words & sent_content_words) / max(1, len(sent_content_words)), 3)
            if sent_content_words else 0.0
        )

        n_high_fidelity = 0
        correctness_scores_this_sentence = []

        for (subj, rel, obj, triple_str) in s_triples:
            correctness_raw = faithfulness(triple_str, sentence_clean)
            correctness_score = round(correctness_raw / 100.0, 3)
            correctness_scores_this_sentence.append(correctness_score)
            is_high_fidelity = correctness_raw >= SIM_THRESHOLD
            if is_high_fidelity:
                n_high_fidelity += 1

            # ---- minimality: shorter triple relative to sentence = more atomic ----
            sent_len = max(1, len(sentence_clean.split()))
            triple_len = max(1, len(triple_str.split()))
            minimality_score = round(max(0.0, 1 - (triple_len / sent_len)), 3)

            quality_rows.append({
                "sent_id": sent_id,
                "subject": subj,
                "relation": rel,
                "object": obj,
                "high_fidelity(sim>={})".format(SIM_THRESHOLD): is_high_fidelity,
                "correctness_%": round(correctness_raw, 1),
                "correctness": correctness_score,
                "completeness_of_sentence": completeness_score,
                "minimality": minimality_score,
                "accuracy_%": round(
                    (correctness_score + completeness_score + minimality_score) / 3 * 100, 1
                ),
            })

            micro_correctness_sum += correctness_score
            micro_n_triples += 1

        n_sys = len(s_triples)
        avg_correctness = (
            sum(correctness_scores_this_sentence) / n_sys if n_sys else 0.0
        )
        f1 = (
            2 * avg_correctness * completeness_score / (avg_correctness + completeness_score)
            if (avg_correctness + completeness_score)
            else 0.0
        )

        per_sentence_rows.append({
            "sent_id": sent_id,
            "sentence": sentence_text,
            "n_sys_triples": n_sys,
            "n_high_fidelity_triples": n_high_fidelity,
            "avg_correctness(precision)": round(avg_correctness, 3),
            "completeness(recall)": completeness_score,
            "f1": round(f1, 3),
        })

        micro_completeness_sum += completeness_score
        micro_n_sentences += 1
        if n_sys or sent_content_words:
            macro_precisions.append(avg_correctness)
            macro_recalls.append(completeness_score)
            macro_f1s.append(f1)

    # ---- aggregate scores ----
    micro_precision = micro_correctness_sum / micro_n_triples if micro_n_triples else 0.0
    micro_recall = micro_completeness_sum / micro_n_sentences if micro_n_sentences else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if (micro_precision + micro_recall)
        else 0.0
    )

    macro_precision = sum(macro_precisions) / len(macro_precisions) if macro_precisions else 0.0
    macro_recall = sum(macro_recalls) / len(macro_recalls) if macro_recalls else 0.0
    macro_f1 = sum(macro_f1s) / len(macro_f1s) if macro_f1s else 0.0

    overall_rows = [
        {"metric": "Precision (Correctness/Faithfulness)",
         "micro": round(micro_precision * 100, 2), "macro": round(macro_precision * 100, 2)},
        {"metric": "Recall (Completeness/Coverage)",
         "micro": round(micro_recall * 100, 2), "macro": round(macro_recall * 100, 2)},
        {"metric": "F1",
         "micro": round(micro_f1 * 100, 2), "macro": round(macro_f1 * 100, 2)},
    ]

    # ---- quality summary ----
    n_q = len(quality_rows)
    avg_correct = sum(r["correctness"] for r in quality_rows) / n_q if n_q else 0
    avg_complete = sum(r["completeness_of_sentence"] for r in quality_rows) / n_q if n_q else 0
    avg_minimal = sum(r["minimality"] for r in quality_rows) / n_q if n_q else 0
    avg_overall = sum(r["accuracy_%"] for r in quality_rows) / n_q if n_q else 0
    pct_high_fidelity = (
        sum(1 for r in quality_rows if r["high_fidelity(sim>={})".format(SIM_THRESHOLD)]) / n_q * 100
        if n_q else 0
    )

    quality_summary_rows = [
        {"criterion": "Correctness (faithfulness to sentence)", "avg_score_%": round(avg_correct * 100, 2)},
        {"criterion": "Completeness (sentence content covered)", "avg_score_%": round(avg_complete * 100, 2)},
        {"criterion": "Minimality (conciseness)", "avg_score_%": round(avg_minimal * 100, 2)},
        {"criterion": "Overall Judged Accuracy", "avg_score_%": round(avg_overall, 2)},
        {"criterion": f"% Triples High-Fidelity (sim>={SIM_THRESHOLD})", "avg_score_%": round(pct_high_fidelity, 2)},
    ]

    # -------------------- write CSVs --------------------
    def write_csv(filename, rows, fieldnames):
        path = os.path.join(OUT_DIR, filename)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return path

    write_csv(
        "per_sentence_scores_nogold.csv",
        per_sentence_rows,
        ["sent_id", "sentence", "n_sys_triples", "n_high_fidelity_triples",
         "avg_correctness(precision)", "completeness(recall)", "f1"],
    )
    write_csv("overall_scores_nogold.csv", overall_rows, ["metric", "micro", "macro"])
    write_csv(
        "quality_judgment_nogold.csv",
        quality_rows,
        ["sent_id", "subject", "relation", "object",
         "high_fidelity(sim>={})".format(SIM_THRESHOLD),
         "correctness_%", "correctness", "completeness_of_sentence", "minimality", "accuracy_%"],
    )
    write_csv("quality_summary_nogold.csv", quality_summary_rows, ["criterion", "avg_score_%"])
    write_csv(
        "counts_summary_nogold.csv",
        [{"stat": k, "value": v} for k, v in stats.items()],
        ["stat", "value"],
    )

    # -------------------- console report --------------------
    print("=" * 70)
    print(f"Number of sentences        : {stats['n_sentences']}")
    print(f"Number of triples          : {stats['n_triples']}")
    print(f"Sentences with >=1 triple  : {stats['n_sentences_with_triples']}")
    print(f"Sentences with 0 triples   : {stats['n_sentences_without_triples']}")
    print(f"Avg triples / sentence     : {stats['avg_triples_per_sentence']}")
    print(f"Fidelity threshold         : {SIM_THRESHOLD}%")
    print("=" * 70)
    print("\n--- Overall Precision / Recall / F1 (%) [reference-free] ---")
    for r in overall_rows:
        print(f"{r['metric']:<42} micro={r['micro']:>6.2f}%   macro={r['macro']:>6.2f}%")

    print("\n--- Quality Judgment Summary (%) ---")
    for r in quality_summary_rows:
        print(f"{r['criterion']:<45} {r['avg_score_%']:>6.2f}%")

    print("\nCSV files written to:", OUT_DIR)



if __name__ == "__main__":
    # --------------------------------------------------------------------------
    # CONFIG
    # --------------------------------------------------------------------------
    
    #SENTENCES_PATH="./Evaluation/data/cnn.txt"
    SENTENCES_PATH="./dataset/CDR/test0_cosine_similarity0/sentenses.txt"
    #SENTENCES_PATH="./Evaluation/data/wire57_sentences.txt"
    #SENTENCES_PATH="./Evaluation/data/wire57_sentences.txt"
    #systemName=["clausie_wire_benchie_form","compactie_wire_benchie_form","imojie_wire_benchie_form","m2oie_wire_benchie_form","minie_wire_benchie_form","openie6_wire_benchie_form","reverb_wire_benchie_form","GoldenTriples"]
    systemName=["aggressiveAllMyMethod"]
    for sys in systemName:
        SYS_PATH ="./dataset/CDR/test0_cosine_similarity0/"+sys+".txt"
        OUT_DIR = "./dataset/CDR/test0_cosine_similarity0/"+sys
        #SYS_PATH ="./Evaluation/data/wire/"+sys+".txt"
        #OUT_DIR = "./Evaluation/data/wire/"+sys
        os.makedirs(OUT_DIR, exist_ok=True)
        evaluate(SENTENCES_PATH,SYS_PATH,OUT_DIR)
        SYS_PATH ="./Evaluation/Ablation Study/_attributive_adj_triples0_cosine_similarity0/"+sys+".txt"
        OUT_DIR = "./Evaluation/Ablation Study/_attributive_adj_triples0_cosine_similarity0/"+sys
                #SYS_PATH ="./Evaluation/data/wire/"+sys+".txt"
                #OUT_DIR = "./Evaluation/data/wire/"+sys
        os.makedirs(OUT_DIR, exist_ok=True)
        evaluate(SENTENCES_PATH,SYS_PATH,OUT_DIR)
        SYS_PATH ="./Evaluation/Ablation Study/_extract_complex_appositions0_cosine_similarity0/"+sys+".txt"
        OUT_DIR = "./Evaluation/Ablation Study/_extract_complex_appositions0_cosine_similarity0/"+sys
                #SYS_PATH ="./Evaluation/data/wire/"+sys+".txt"
                #OUT_DIR = "./Evaluation/data/wire/"+sys
        os.makedirs(OUT_DIR, exist_ok=True)
        evaluate(SENTENCES_PATH,SYS_PATH,OUT_DIR)
        SYS_PATH ="./Evaluation/Ablation Study/_nmod_compound_triples0_cosine_similarity0/"+sys+".txt"
        OUT_DIR = "./Evaluation/Ablation Study/_nmod_compound_triples0_cosine_similarity0/"+sys
                #SYS_PATH ="./Evaluation/data/wire/"+sys+".txt"
                #OUT_DIR = "./Evaluation/data/wire/"+sys
        os.makedirs(OUT_DIR, exist_ok=True)
        evaluate(SENTENCES_PATH,SYS_PATH,OUT_DIR)
        SYS_PATH ="./Evaluation/Ablation Study/_possessive_triples0_cosine_similarity0/"+sys+".txt"
        OUT_DIR = "./Evaluation/Ablation Study/_possessive_triples0_cosine_similarity0/"+sys
                #SYS_PATH ="./Evaluation/data/wire/"+sys+".txt"
                #OUT_DIR = "./Evaluation/data/wire/"+sys
        os.makedirs(OUT_DIR, exist_ok=True)
        evaluate(SENTENCES_PATH,SYS_PATH,OUT_DIR)
