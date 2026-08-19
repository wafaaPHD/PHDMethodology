"""
OpenIE Evaluation Script
=========================
Compares system-extracted triples (aggressiveAllMyMethod.txt) against
gold/annotated clustered triples (benchie-annotated_300_.txt) in the
BenchIE style: a gold "cluster" groups several surface-form variants of
the SAME fact. A system triple counts as correct if it is similar
(>= SIM_THRESHOLD, default 90%) to ANY variant inside a cluster; a
cluster counts as recalled if AT LEAST ONE system triple matched it.

Outputs:
  1. per_sentence_scores.csv   -> P/R/F1 per sentence
  2. overall_scores.csv        -> micro + macro aggregated P/R/F1
  3. quality_judgment.csv      -> per-triple minimality / completeness / correctness
  4. quality_summary.csv       -> aggregated accuracy % per criterion
"""

import re
import csv
from collections import defaultdict
from rapidfuzz import fuzz
import os

SIM_THRESHOLD = 90
# --------------------------------------------------------------------------
# HELPERS
# --------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """Remove bracket markers [ ] but keep the words inside them, normalize
    whitespace, lowercase, strip punctuation spacing artifacts."""
    text = text.replace("[", "").replace("]", "")
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def triple_to_string(subj: str, rel: str, obj: str) -> str:
    return clean_text(f"{subj} {rel} {obj}")


def similarity(a: str, b: str) -> float:
    """Token-order-independent similarity score 0-100."""
    return fuzz.token_sort_ratio(a, b)


# --------------------------------------------------------------------------
# PARSE GOLD FILE
# --------------------------------------------------------------------------
def parse_gold(path):
    """
    Returns:
      gold_sentences: dict sent_id -> sentence text
      gold_clusters:  dict sent_id -> list of clusters
                       each cluster = list of triple strings (cleaned)
    """
    gold_sentences = {}
    gold_clusters = defaultdict(list)

    sent_id_re = re.compile(r"^sent_id:(\d+)\s*(.*)$")
    cluster_re = re.compile(r"^(\d+)-->\s*Cluster\s*(\d+):\s*$")

    current_sent = None
    current_cluster_variants = None

    with open(path, encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            if not line.strip():
                continue

            m_sent = sent_id_re.match(line)
            if m_sent:
                # flush previous cluster
                if current_sent is not None and current_cluster_variants:
                    gold_clusters[current_sent].append(current_cluster_variants)
                current_sent = int(m_sent.group(1))
                gold_sentences[current_sent] = m_sent.group(2).strip()
                current_cluster_variants = None
                continue

            m_cluster = cluster_re.match(line)
            if m_cluster:
                # flush previous cluster, start a new one
                if current_sent is not None and current_cluster_variants:
                    gold_clusters[current_sent].append(current_cluster_variants)
                current_cluster_variants = []
                continue

            # otherwise it's a triple line: "arg1 --> rel --> arg2" (or more arrows)
            if "-->" in line and current_sent is not None:
                parts = [p.strip() for p in line.split("-->")]
                if len(parts) >= 3:
                    subj = parts[0]
                    obj = parts[-1]
                    rel = " ".join(parts[1:-1])
                    triple_str = triple_to_string(subj, rel, obj)
                    if current_cluster_variants is None:
                        current_cluster_variants = []
                    current_cluster_variants.append(triple_str)

        # flush last cluster
        if current_sent is not None and current_cluster_variants:
            gold_clusters[current_sent].append(current_cluster_variants)

    return gold_sentences, gold_clusters


# --------------------------------------------------------------------------
# PARSE SYSTEM FILE
# --------------------------------------------------------------------------
def parse_system(path):
    """
    Tab-separated: sent_id \t subject \t relation \t object
    Returns dict sent_id -> list of (subj, rel, obj, triple_string)
    """
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
# MATCHING
# --------------------------------------------------------------------------
def count_stats(gold_sentences: dict, gold_clusters: dict, sys_triples: dict) -> dict:
    """Basic counts: number of sentences, number of gold clusters,
    and number of system triples involved in the evaluation."""
    n_sentences = len(gold_sentences)
    n_gold_clusters = sum(len(v) for v in gold_clusters.values())
    n_sys_triples = sum(len(v) for v in sys_triples.values())
    n_sentences_with_sys_triples = sum(
        1 for sid in gold_sentences if len(sys_triples.get(sid, [])) > 0
    )
    avg_triples_per_sentence = round(n_sys_triples / n_sentences, 3) if n_sentences else 0.0
    avg_gold_clusters_per_sentence = round(n_gold_clusters / n_sentences, 3) if n_sentences else 0.0
    return {
        "n_sentences": n_sentences,
        "n_gold_clusters": n_gold_clusters,
        "n_sys_triples": n_sys_triples,
        "n_sentences_with_sys_triples": n_sentences_with_sys_triples,
        "avg_gold_clusters_per_sentence": avg_gold_clusters_per_sentence,
        "avg_triples_per_sentence": avg_triples_per_sentence,
    }


def best_cluster_match(sys_triple_str, clusters):
    """Return (best_cluster_index, best_similarity, best_variant_matched) among clusters,
    or (None, best_similarity_seen, None) if nothing crosses the threshold."""
    best_idx, best_sim, best_variant = None, 0.0, None
    for idx, variants in enumerate(clusters):
        for variant in variants:
            sim = similarity(sys_triple_str, variant)
            if sim > best_sim:
                best_sim = sim
                best_variant = variant
                if sim >= SIM_THRESHOLD:
                    best_idx = idx
    return best_idx, best_sim, best_variant


# --------------------------------------------------------------------------
# MAIN EVALUATION
# --------------------------------------------------------------------------
def evaluate(GOLD_PATH,SYS_PATH,OUT_DIR):
    gold_sentences, gold_clusters = parse_gold(GOLD_PATH)
    sys_triples = parse_system(SYS_PATH)
    stats = count_stats(gold_sentences, gold_clusters, sys_triples)

    per_sentence_rows = []
    quality_rows = []

    micro_tp_p, micro_total_sys = 0, 0          # for precision (system side)
    micro_tp_r, micro_total_gold = 0, 0          # for recall (gold side)

    macro_precisions, macro_recalls, macro_f1s = [], [], []

    all_sent_ids = sorted(set(list(gold_clusters.keys()) + list(sys_triples.keys())))

    for sent_id in all_sent_ids:
        clusters = gold_clusters.get(sent_id, [])
        s_triples = sys_triples.get(sent_id, [])
        sentence_text = gold_sentences.get(sent_id, "")

        matched_cluster_idxs = set()
        correct_sys_count = 0

        for (subj, rel, obj, triple_str) in s_triples:
            cluster_idx, best_sim, best_variant = best_cluster_match(triple_str, clusters)
            is_correct = cluster_idx is not None
            if is_correct:
                correct_sys_count += 1
                matched_cluster_idxs.add(cluster_idx)

            # ---- quality judgment (minimality / completeness / correctness) ----
            correctness_score = round(best_sim / 100.0, 3)

            if is_correct:
                variants = clusters[cluster_idx]
                # completeness: overlap of system triple's words with the
                # longest (most complete) gold variant in the matched cluster
                longest_variant = max(variants, key=lambda v: len(v.split()))
                gold_words = set(longest_variant.split())
                sys_words = set(triple_str.split())
                completeness_score = round(
                    len(gold_words & sys_words) / max(1, len(gold_words)), 3
                )
                # minimality: ratio of shortest gold variant length to system
                # triple length, capped at 1.0 (system should not be longer
                # than the minimal gold expression of the same fact)
                shortest_variant = min(variants, key=lambda v: len(v.split()))
                shortest_len = len(shortest_variant.split())
                sys_len = len(triple_str.split())
                minimality_score = round(min(1.0, shortest_len / max(1, sys_len)), 3)
            else:
                completeness_score = 0.0
                minimality_score = 0.0

            quality_rows.append({
                "sent_id": sent_id,
                "subject": subj,
                "relation": rel,
                "object": obj,
                "matched": is_correct,
                "best_similarity_%": round(best_sim, 1),
                "correctness": correctness_score,
                "completeness": completeness_score,
                "minimality": minimality_score,
                "accuracy_%": round(
                    (correctness_score + completeness_score + minimality_score) / 3 * 100, 1
                ),
            })

        n_sys = len(s_triples)
        n_gold = len(clusters)
        n_recalled = len(matched_cluster_idxs)

        precision = correct_sys_count / n_sys if n_sys else 0.0
        recall = n_recalled / n_gold if n_gold else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        per_sentence_rows.append({
            "sent_id": sent_id,
            "sentence": sentence_text,
            "n_gold_clusters": n_gold,
            "n_sys_triples": n_sys,
            "n_correct_sys_triples": correct_sys_count,
            "n_recalled_clusters": n_recalled,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
        })

        micro_tp_p += correct_sys_count
        micro_total_sys += n_sys
        micro_tp_r += n_recalled
        micro_total_gold += n_gold

        if n_sys or n_gold:
            macro_precisions.append(precision)
            macro_recalls.append(recall)
            macro_f1s.append(f1)

    # ---- aggregate scores ----
    micro_precision = micro_tp_p / micro_total_sys if micro_total_sys else 0.0
    micro_recall = micro_tp_r / micro_total_gold if micro_total_gold else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if (micro_precision + micro_recall)
        else 0.0
    )

    macro_precision = sum(macro_precisions) / len(macro_precisions) if macro_precisions else 0.0
    macro_recall = sum(macro_recalls) / len(macro_recalls) if macro_recalls else 0.0
    macro_f1 = sum(macro_f1s) / len(macro_f1s) if macro_f1s else 0.0

    overall_rows = [
        {"metric": "Precision", "micro": round(micro_precision * 100, 2), "macro": round(macro_precision * 100, 2)},
        {"metric": "Recall", "micro": round(micro_recall * 100, 2), "macro": round(macro_recall * 100, 2)},
        {"metric": "F1", "micro": round(micro_f1 * 100, 2), "macro": round(macro_f1 * 100, 2)},
    ]

    # ---- quality summary ----
    n_q = len(quality_rows)
    avg_correct = sum(r["correctness"] for r in quality_rows) / n_q if n_q else 0
    avg_complete = sum(r["completeness"] for r in quality_rows) / n_q if n_q else 0
    avg_minimal = sum(r["minimality"] for r in quality_rows) / n_q if n_q else 0
    avg_overall = sum(r["accuracy_%"] for r in quality_rows) / n_q if n_q else 0
    pct_matched = sum(1 for r in quality_rows if r["matched"]) / n_q * 100 if n_q else 0

    quality_summary_rows = [
        {"criterion": "Correctness", "avg_score_%": round(avg_correct * 100, 2)},
        {"criterion": "Completeness", "avg_score_%": round(avg_complete * 100, 2)},
        {"criterion": "Minimality", "avg_score_%": round(avg_minimal * 100, 2)},
        {"criterion": "Overall Judged Accuracy", "avg_score_%": round(avg_overall, 2)},
        {"criterion": f"% System Triples Matched to Gold (sim>={SIM_THRESHOLD})",
         "avg_score_%": round(pct_matched, 2)},
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
        "per_sentence_scores.csv",
        per_sentence_rows,
        ["sent_id", "sentence", "n_gold_clusters", "n_sys_triples",
         "n_correct_sys_triples", "n_recalled_clusters", "precision", "recall", "f1"],
    )
    write_csv("overall_scores.csv", overall_rows, ["metric", "micro", "macro"])
    write_csv(
        "quality_judgment.csv",
        quality_rows,
        ["sent_id", "subject", "relation", "object", "matched",
         "best_similarity_%", "correctness", "completeness", "minimality", "accuracy_%"],
    )
    write_csv("quality_summary.csv", quality_summary_rows, ["criterion", "avg_score_%"])
    write_csv(
        "counts_summary_gold.csv",
        [{"stat": k, "value": v} for k, v in stats.items()],
        ["stat", "value"],
    )

    # -------------------- console report --------------------
    print("=" * 70)
    print(f"Number of sentences         : {stats['n_sentences']}")
    print(f"Number of gold clusters     : {stats['n_gold_clusters']}")
    print(f"Number of system triples    : {stats['n_sys_triples']}")
    print(f"Sentences with sys triples  : {stats['n_sentences_with_sys_triples']}")
    print(f"Avg gold clusters/sentence  : {stats['avg_gold_clusters_per_sentence']}")
    print(f"Avg sys triples/sentence    : {stats['avg_triples_per_sentence']}")
    print(f"Similarity threshold        : {SIM_THRESHOLD}%")
    print("=" * 70)
    print("\n--- Overall Precision / Recall / F1 (%) ---")
    for r in overall_rows:
        print(f"{r['metric']:<10} micro={r['micro']:>6.2f}%   macro={r['macro']:>6.2f}%")

    print("\n--- Quality Judgment Summary (%) ---")
    for r in quality_summary_rows:
        print(f"{r['criterion']:<55} {r['avg_score_%']:>6.2f}%")

    print("\nCSV files written to:", OUT_DIR)


if __name__ == "__main__":
    # --------------------------------------------------------------------------
    # CONFIG
    # --------------------------------------------------------------------------
    #GOLD_PATH = "./Evaluation/RE/re_oie2016_benchie_format.txt"
    GOLD_PATH = "./Evaluation/data/gold/benchie_gold_annotations_en.txt"
    #GOLD_PATH = "./Evaluation/data/gold/wire57-annotated(57).txt"
    systemName=["aggressiveAllMyMethod"]
    for sys in systemName:
            SYS_PATH ="./Evaluation/data/OIE/"+sys+".txt"
            OUT_DIR = "./Evaluation/data/OIE/"+sys
            #SYS_PATH ="./Evaluation/data/wire/"+sys+".txt"
            #OUT_DIR = "./Evaluation/data/wire/"+sys
    #systemName=["aggressiveAllMyMethod"]
    #for sys in systemName:
    #        SYS_PATH ="./Evaluation/RE/"+sys+".txt"
    #        OUT_DIR = "./Evaluation/RE/"+sys
            os.makedirs(OUT_DIR, exist_ok=True)
            evaluate(GOLD_PATH,SYS_PATH,OUT_DIR)
    GOLD_PATH = "./Evaluation/data/gold/2_annotators/OLD_wire57-annotated(57).txt"
    systemName=["clausie_wire_benchie_form","compactie_wire_benchie_form","imojie_wire_benchie_form","m2oie_wire_benchie_form","minie_wire_benchie_form","openie6_wire_benchie_form","reverb_wire_benchie_form"]
    for sys in systemName:
                SYS_PATH ="./Evaluation/data/wire/"+sys+".txt"
                OUT_DIR = "./Evaluation/wire/"+sys
                os.makedirs(OUT_DIR, exist_ok=True)
                evaluate(GOLD_PATH,SYS_PATH,OUT_DIR)
        
    
