import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
def extract_filter_stats(row, filter_name):
    if filter_name in row['filter_stats']:
        return row['filter_stats'][filter_name]
    return None
def mainMomery2(data,args):
    # Convert to DataFrame
    df = pd.DataFrame(data)
    # ============================================================
    # SECTION 1: DATA OVERVIEW AND BASIC STATISTICS
    # ============================================================

    print("="*60)
    print("DATA OVERVIEW")
    print("="*60)
    print(f"Total processing entries: {len(df)}")
    print(f"Columns available: {list(df.columns)}")
    print("\nFirst 3 entries:")
    print(df.head(3))

    # ============================================================
    # SECTION 2: TRAINING TIME ANALYSIS
    # ============================================================

    print("\n" + "="*60)
    print("TRAINING/PROCESSING TIME ANALYSIS")
    print("="*60)

    # Processing time statistics
    processing_times = df['processing_time_seconds']
    print(f"Processing Time - Mean: {processing_times.mean():.4f} seconds")
    print(f"Processing Time - Median: {processing_times.median():.4f} seconds")
    print(f"Processing Time - Min: {processing_times.min():.4f} seconds")
    print(f"Processing Time - Max: {processing_times.max():.4f} seconds")
    print(f"Processing Time - Std Dev: {processing_times.std():.4f} seconds")

    # ============================================================
    # SECTION 3: INFERENCE SPEED ANALYSIS
    # ============================================================

    print("\n" + "="*60)
    print("INFERENCE SPEED ANALYSIS")
    print("="*60)

    inference_times = df['inference_time_seconds']
    print(f"Inference Time - Mean: {inference_times.mean():.6f} seconds")
    print(f"Inference Time - Median: {inference_times.median():.6f} seconds")
    print(f"Inference Time - Min: {inference_times.min():.6f} seconds")
    print(f"Inference Time - Max: {inference_times.max():.6f} seconds")

    # Average inference time per triple
    df['inference_per_triple'] = df['inference_time_seconds'] / df['total_triples_before_filter']
    print(f"Inference Time per Triple - Mean: {df['inference_per_triple'].mean():.6f} seconds")
    print(f"Inference Time per Triple - Median: {df['inference_per_triple'].median():.6f} seconds")

    # ============================================================
    # SECTION 4: MEMORY USAGE ANALYSIS
    # ============================================================

    print("\n" + "="*60)
    print("MEMORY USAGE ANALYSIS")
    print("="*60)

    # Memory usage during processing
    memory_usage = df['memory_usage_mb']
    print(f"Memory Usage (delta) - Mean: {memory_usage.mean():.4f} MB")
    print(f"Memory Usage (delta) - Median: {memory_usage.median():.4f} MB")
    print(f"Memory Usage (delta) - Min: {memory_usage.min():.4f} MB")
    print(f"Memory Usage (delta) - Max: {memory_usage.max():.4f} MB")

    # Final memory after processing
    final_memory = df['final_memory_mb']
    print(f"\nFinal Memory Usage - Mean: {final_memory.mean():.4f} MB")
    print(f"Final Memory Usage - Median: {final_memory.median():.4f} MB")
    print(f"Final Memory Usage - Min: {final_memory.min():.4f} MB")
    print(f"Final Memory Usage - Max: {final_memory.max():.4f} MB")

    # Memory trend
    print(f"\nMemory Growth (final - initial): {final_memory.iloc[-1] - final_memory.iloc[0]:.4f} MB")

    # ============================================================
    # SECTION 5: SENTENCE LENGTH ANALYSIS
    # ============================================================

    print("\n" + "="*60)
    print("SENTENCE LENGTH ANALYSIS")
    print("="*60)

    # Raw text statistics
    raw_text_lengths = df['raw_text_length']
    raw_text_words = df['raw_text_words']
    total_sentences_length = df['total_sentences_length'].apply(lambda x: x[0] if x else 0)

    print(f"Raw Text Length (characters) - Mean: {raw_text_lengths.mean():.2f}")
    print(f"Raw Text Length (characters) - Median: {raw_text_lengths.median():.2f}")
    print(f"Raw Text Length (characters) - Min: {raw_text_lengths.min()}")
    print(f"Raw Text Length (characters) - Max: {raw_text_lengths.max()}")

    print(f"\nRaw Text Words - Mean: {raw_text_words.mean():.2f}")
    print(f"Raw Text Words - Median: {raw_text_words.median():.2f}")
    print(f"Raw Text Words - Min: {raw_text_words.min()}")
    print(f"Raw Text Words - Max: {raw_text_words.max()}")

    print(f"\nSentences Length (per entry) - Mean: {total_sentences_length.mean():.2f}")
    print(f"Sentences Length (per entry) - Median: {total_sentences_length.median():.2f}")

    # ============================================================
    # SECTION 6: FILTER STATISTICS ANALYSIS
    # ============================================================

    print("\n" + "="*60)
    print("FILTER STATISTICS ANALYSIS")
    print("="*60)
    # Extract DOIEUD stats
    df['doieud_total'] = df['filter_stats'].apply(lambda x: x.get('DOIEUD', {}).get('total', 0))
    df['doieud_kept'] = df['filter_stats'].apply(lambda x: x.get('DOIEUD', {}).get('kept', 0))
    df['doieud_filtered'] = df['filter_stats'].apply(lambda x: x.get('DOIEUD', {}).get('filtered', 0))
    df['doieud_precision'] = df['filter_stats'].apply(lambda x: x.get('DOIEUD', {}).get('precision', 0))

    # Extract DOIEUD2 stats
    df['doieud2_total'] = df['filter_stats'].apply(lambda x: x.get('DOIEUD2', {}).get('total', 0))
    df['doieud2_kept'] = df['filter_stats'].apply(lambda x: x.get('DOIEUD2', {}).get('kept', 0))
    df['doieud2_filtered'] = df['filter_stats'].apply(lambda x: x.get('DOIEUD2', {}).get('filtered', 0))
    df['doieud2_precision'] = df['filter_stats'].apply(lambda x: x.get('DOIEUD2', {}).get('precision', 0))

    print("DOIEUD Filter:")
    print(f"  Total triples processed: {df['doieud_total'].sum()}")
    print(f"  Total kept: {df['doieud_kept'].sum()}")
    print(f"  Total filtered: {df['doieud_filtered'].sum()}")
    print(f"  Overall Precision: {df['doieud_kept'].sum() / df['doieud_total'].sum():.4f}")

    print("\nDOIEUD2 Filter:")
    print(f"  Total triples processed: {df['doieud2_total'].sum()}")
    print(f"  Total kept: {df['doieud2_kept'].sum()}")
    print(f"  Total filtered: {df['doieud2_filtered'].sum()}")
    print(f"  Overall Precision: {df['doieud2_kept'].sum() / df['doieud2_total'].sum():.4f}")

    # ============================================================
    # SECTION 7: RELATIONS EXTRACTION ANALYSIS
    # ============================================================

    print("\n" + "="*60)
    print("RELATIONS EXTRACTION ANALYSIS")
    print("="*60)

    print(f"Total Relations Extracted - Mean: {df['total_relations_extracted'].mean():.2f}")
    print(f"Total Relations Extracted - Median: {df['total_relations_extracted'].median():.2f}")
    print(f"Total Relations Extracted - Min: {df['total_relations_extracted'].min()}")
    print(f"Total Relations Extracted - Max: {df['total_relations_extracted'].max()}")

    print(f"\nUnique Relations - Mean: {df['unique_relations'].mean():.2f}")
    print(f"Unique Relations - Median: {df['unique_relations'].median():.2f}")
    print(f"Unique Relations - Min: {df['unique_relations'].min()}")
    print(f"Unique Relations - Max: {df['unique_relations'].max()}")

    # ============================================================
    # SECTION 8: CORRELATION ANALYSIS
    # ============================================================

    print("\n" + "="*60)
    print("CORRELATION ANALYSIS")
    print("="*60)

    # Correlation matrix
    correlation_cols = [
        'raw_text_length', 'raw_text_words', 'total_triples_before_filter',
        'total_triples_after_filter', 'processing_time_seconds',
        'total_relations_extracted', 'unique_relations'
    ]
    correlation_matrix = df[correlation_cols].corr()

    print("\nCorrelation Matrix:")
    print(correlation_matrix.round(4))

    # ============================================================
    # SECTION 9: THEORETICAL COMPLEXITY ANALYSIS
    # ============================================================

    print("\n" + "="*60)
    print("THEORETICAL COMPLEXITY ANALYSIS")
    print("="*60)

    print("""
    ┌─────────────────────────────────────────────────────────────────────┐
    │                    COMPLEXITY ANALYSIS                             │
    ├─────────────────────────────────────────────────────────────────────┤
    │                                                                    │
    │  1. INPUT SIZE (n)                                                │
    │     - n = number of sentences (S)                                 │
    │     - Each sentence has length L_i                                │
    │     - Total characters = C = Σ L_i                               │
    │                                                                    │
    │  2. PREPROCESSING                                                 │
    │     - Tokenization: O(C)                                          │
    │     - Sentence splitting: O(C)                                   │
    │     - Total: O(C)                                                │
    │                                                                    │
    │  3. TRIPLE EXTRACTION                                             │
    │     - For each sentence: O(L_i²) due to pair-wise relation        │
    │       extraction between entities                                │
    │     - Total: O(Σ L_i²) = O(S * L_avg²)                          │
    │     - Worst case: O(S * L_max²)                                 │
    │                                                                    │
    │  4. FILTERING (DOIEUD & DOIEUD2)                                  │
    │     - Each triple processed once: O(T)                           │
    │     - Where T = total triples before filter                      │
    │     - T ≈ O(S * L_avg²)                                         │
    │     - Filtering with embeddings: O(T * d)                        │
    │       where d = embedding dimension                              │
    │                                                                    │
    │  5. INFERENCE                                                    │
    │     - Model inference per triple: O(T * f(d))                    │
    │     - f(d) depends on model architecture                         │
    │     - For transformer models: O(d²) per inference                │
    │                                                                    │
    │  6. OVERALL COMPLEXITY                                            │
    │     - O(S * L_max² * d)                                         │
    │     - or O(C * L_max * d)                                      │
    │                                                                    │
    │  7. MEMORY COMPLEXITY                                             │
    │     - O(S * L_max²) for storing all triples                     │
    │     - O(d) for model parameters                                 │
    │     - O(T * d) for embedding storage                            │
    │                                                                    │
    └─────────────────────────────────────────────────────────────────────┘
    """)

    # ============================================================
    # SECTION 10: EMPIRICAL COMPLEXITY VALIDATION
    # ============================================================

    print("\n" + "="*60)
    print("EMPIRICAL COMPLEXITY VALIDATION")
    print("="*60)

    # Group by text length ranges
    df['text_length_bin'] = pd.cut(df['raw_text_length'], bins=[0, 50, 100, 150, 200, 250, 300, 350])
    length_bins = df.groupby('text_length_bin').agg({
        'processing_time_seconds': ['mean', 'std'],
        'total_triples_before_filter': ['mean', 'std'],
        'total_triples_after_filter': ['mean', 'std']
    }).round(4)

    print("\nProcessing Time by Text Length:")
    print(length_bins)

    # ============================================================
    # SECTION 11: VISUALIZATION
    # ============================================================

    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. Processing Time vs Text Length
    ax1 = axes[0, 0]
    ax1.scatter(df['raw_text_length'], df['processing_time_seconds'], alpha=0.5)
    z = np.polyfit(df['raw_text_length'], df['processing_time_seconds'], 1)
    p = np.poly1d(z)
    ax1.plot(df['raw_text_length'].sort_values(), p(df['raw_text_length'].sort_values()), 'r-', label=f'Linear fit (slope={z[0]:.4f})')
    ax1.set_xlabel('Raw Text Length (chars)')
    ax1.set_ylabel('Processing Time (seconds)')
    ax1.set_title('Processing Time vs Text Length')
    ax1.legend()

    # 2. Processing Time vs Triples
    ax2 = axes[0, 1]
    ax2.scatter(df['total_triples_before_filter'], df['processing_time_seconds'], alpha=0.5)
    z2 = np.polyfit(df['total_triples_before_filter'], df['processing_time_seconds'], 1)
    p2 = np.poly1d(z2)
    ax2.plot(df['total_triples_before_filter'].sort_values(), p2(df['total_triples_before_filter'].sort_values()), 'r-', label=f'Linear fit (slope={z2[0]:.4f})')
    ax2.set_xlabel('Total Triples Before Filter')
    ax2.set_ylabel('Processing Time (seconds)')
    ax2.set_title('Processing Time vs Number of Triples')
    ax2.legend()

    # 3. Memory Usage Over Time
    ax3 = axes[0, 2]
    ax3.plot(range(len(df)), df['final_memory_mb'], 'b-', alpha=0.5)
    ax3.fill_between(range(len(df)), df['final_memory_mb'].min(), df['final_memory_mb'], alpha=0.2)
    ax3.set_xlabel('Processing Order')
    ax3.set_ylabel('Final Memory (MB)')
    ax3.set_title('Memory Usage Over Processing')
    ax3.axhline(y=df['final_memory_mb'].mean(), color='r', linestyle='--', label=f'Mean: {df["final_memory_mb"].mean():.2f} MB')
    ax3.legend()

    # 4. Filter Precision Comparison
    ax4 = axes[1, 0]
    doieud_prec = df[df['doieud_total'] > 0]['doieud_precision'].dropna()
    doieud2_prec = df[df['doieud2_total'] > 0]['doieud2_precision'].dropna()
    ax4.boxplot([doieud_prec, doieud2_prec])
    ax4.set_ylabel('Precision')
    ax4.set_title('Filter Precision Comparison')
    ax4.axhline(y=0.5, color='r', linestyle='--', label='Random baseline')
    ax4.legend()

    # 5. Relations Distribution
    ax5 = axes[1, 1]
    ax5.scatter(df['total_relations_extracted'], df['unique_relations'], alpha=0.5)
    ax5.plot([0, max(df['total_relations_extracted'])], [0, max(df['total_relations_extracted'])], 'r--', label='y=x')
    ax5.set_xlabel('Total Relations Extracted')
    ax5.set_ylabel('Unique Relations')
    ax5.set_title('Total vs Unique Relations')
    ax5.legend()

    # 6. Words vs Processing Time
    ax6 = axes[1, 2]
    ax6.scatter(df['raw_text_words'], df['processing_time_seconds'], alpha=0.5)
    z6 = np.polyfit(df['raw_text_words'], df['processing_time_seconds'], 1)
    p6 = np.poly1d(z6)
    ax6.plot(df['raw_text_words'].sort_values(), p6(df['raw_text_words'].sort_values()), 'r-', label=f'Linear fit (slope={z6[0]:.4f})')
    ax6.set_xlabel('Number of Words')
    ax6.set_ylabel('Processing Time (seconds)')
    ax6.set_title('Processing Time vs Word Count')
    ax6.legend()

    plt.tight_layout()
    plt.savefig(args.PathDataset+'processing_analysis_plots.png', dpi=300, bbox_inches='tight')
    print("\nVisualization saved as 'processing_analysis_plots.png'")
    #plt.show()
    plt.close(fig)

    # ============================================================
    # SECTION 12: DETAILED COMPLEXITY ANALYSIS WITH EMPIRICAL DATA
    # ============================================================

    print("\n" + "="*60)
    print("DETAILED COMPLEXITY ANALYSIS")
    print("="*60)

    # Calculate empirical complexity exponents
    # Using log-log regression to estimate exponents

    # For processing time vs text length
    log_len = np.log(df['raw_text_length'])
    log_time = np.log(df['processing_time_seconds'])
    slope_len, intercept_len, r_value_len, p_value_len, std_err_len = stats.linregress(log_len, log_time)

    print(f"\nProcessing Time ~ (Text Length)^{slope_len:.3f}")
    print(f"  R² = {r_value_len**2:.4f}")
    print(f"  p-value = {p_value_len:.6f}")

    # For processing time vs number of triples
    log_triples = np.log(df['total_triples_before_filter'])
    slope_triples, intercept_triples, r_value_triples, p_value_triples, std_err_triples = stats.linregress(log_triples, log_time)

    print(f"\nProcessing Time ~ (Number of Triples)^{slope_triples:.3f}")
    print(f"  R² = {r_value_triples**2:.4f}")
    print(f"  p-value = {p_value_triples:.6f}")

    # ============================================================
    # SECTION 13: SUMMARY STATISTICS TABLE
    # ============================================================

    print("\n" + "="*60)
    print("SUMMARY STATISTICS TABLE")
    print("="*60)

    summary_table = pd.DataFrame({
        'Metric': [
            'Total Processing Entries',
            'Avg Processing Time (s)',
            'Total Processing Time (s)',
            'Avg Inference Time (s)',
            'Avg Inference per Triple (s)',
            'Avg Memory Delta (MB)',
            'Avg Final Memory (MB)',
            'Avg Raw Text Length (chars)',
            'Avg Raw Text Words',
            'Avg Triples Before Filter',
            'Avg Triples After Filter',
            'Filter Keep Rate',
            'Avg Relations Extracted',
            'Avg Unique Relations'
        ],
        'Value': [
            len(df),
            f"{processing_times.mean():.4f}",
            f"{processing_times.sum():.2f}",
            f"{inference_times.mean():.6f}",
            f"{df['inference_per_triple'].mean():.6f}",
            f"{memory_usage.mean():.4f}",
            f"{final_memory.mean():.4f}",
            f"{raw_text_lengths.mean():.2f}",
            f"{raw_text_words.mean():.2f}",
            f"{df['total_triples_before_filter'].mean():.2f}",
            f"{df['total_triples_after_filter'].mean():.2f}",
            f"{(df['total_triples_after_filter'].sum() / df['total_triples_before_filter'].sum() * 100):.2f}%",
            f"{df['total_relations_extracted'].mean():.2f}",
            f"{df['unique_relations'].mean():.2f}"
        ]
    })

    print(summary_table.to_string(index=False))

    # ============================================================
    # SECTION 14: SAVE RESULTS TO CSV
    # ============================================================

    # Save analyzed data
    df.to_csv(args.PathDataset+'analyzed_processing_stats.csv', index=False)
    summary_table.to_csv('summary_statistics.csv', index=False)

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print("Files generated:")
    print("  - analyzed_processing_stats.csv")
    print("  - summary_statistics.csv")
    print("  - processing_analysis_plots.png")

    # ============================================================
    # SECTION 15: ADDITIONAL COMPLEXITY FORMULAS
    # ============================================================

    print("\n" + "="*60)
    print("THEORETICAL COMPLEXITY FORMULAS")
    print("="*60)

    print("""
    ┌─────────────────────────────────────────────────────────────────────┐
    │                    COMPLEXITY FORMULAS                             │
    ├─────────────────────────────────────────────────────────────────────┤
    │                                                                    │
    │  Let:                                                             │
    │    - S = number of sentences                                      │
    │    - L_i = length of sentence i                                  │
    │    - L_avg = average sentence length                             │
    │    - L_max = maximum sentence length                             │
    │    - T = total triples                                           │
    │    - d = embedding dimension                                     │
    │    - C = total characters = Σ L_i                              │
    │                                                                    │
    │  1. TRIPLE EXTRACTION                                             │
    │     T = O(Σ L_i²)                                               │
    │     T = O(S * L_avg²)                                          │
    │                                                                    │
    │  2. FILTERING                                                     │
    │     O(T * d)                                                     │
    │     O(S * L_avg² * d)                                          │
    │                                                                    │
    │  3. INFERENCE                                                     │
    │     O(T * f(d))                                                  │
    │     where f(d) is inference cost per triple                      │
    │     For transformer: f(d) = O(d²)                               │
    │     Total: O(T * d²) = O(S * L_avg² * d²)                     │
    │                                                                    │
    │  4. MEMORY                                                        │
    │     - Input storage: O(C)                                        │
    │     - Triple storage: O(T)                                       │
    │     - Embeddings: O(T * d)                                      │
    │     - Model: O(d²)                                              │
    │     Total: O(S * L_avg² * d)                                  │
    │                                                                    │
    │  5. EMPIRICAL OBSERVATIONS                                        │
    │     From log-log regression:                                     │
    │     - Processing Time ~ Text Length^{%.3f}                      │
    │     - Processing Time ~ Triples^{%.3f}                          │
    │     - Processing Time ~ Words^{%.3f}                            │
    │                                                                    │
    └─────────────────────────────────────────────────────────────────────┘
    """ % (slope_len, slope_triples, slope_len))