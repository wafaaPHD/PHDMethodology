from ast import arg
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import os
import warnings
warnings.filterwarnings('ignore')

# Parse all sentences and their triple data
def parse_all_sentences(data):
    """Extract all sentences and their triple information from the nested JSON structure."""
    sentences_data = []
    
    def process_sentence(sentence_entry):
        """Process a single sentence entry containing the sentence and its triple data."""
        if len(sentence_entry) >= 4:
            # The structure is: [cosine_sim, z_score, sentence_text, extraction1, extraction2, extraction3]
            sentence_text = sentence_entry[2]
            extraction_results = sentence_entry[3:]  # All the extraction objects
            
            for i, result in enumerate(extraction_results):
                if isinstance(result, dict):
                    result['sentence'] = sentence_text
                    result['extraction_version'] = i + 1
                    sentences_data.append(result)
    
    # Navigate through the nested structure
    for main_entry in data:
        if isinstance(main_entry, list):
            for sub_entry in main_entry:
                if isinstance(sub_entry, list):
                    process_sentence(sub_entry)
    return sentences_data

def analyze_extractions(data):
    """Analyze the extraction results and compute key metrics."""
    sentences_data = parse_all_sentences(data)
    
    # Create DataFrames for analysis
    results_df = pd.DataFrame(sentences_data)
    
    # Flatten per_triple_scores into separate rows
    triple_rows = []
    results_df['per_triple_scores'] = results_df['per_triple_scores'].fillna(0)
    for idx, row in results_df.iterrows():
        if 'per_triple_scores' in row and row['per_triple_scores']:
             for triple_score in row['per_triple_scores']:
                    triple_rows.append({
                        'sentence': row['sentence'],
                        'extraction_version': row['extraction_version'],
                        'total_triples': row.get('total_triples', 0),
                        'kept_triples': row.get('kept_triples', 0),
                        'filtered_out': row.get('filtered_out', 0),
                        'precision': row.get('precision', 0),
                        'triple': triple_score.get('triple', []),
                        'relevance': triple_score.get('relevance', 0),
                        'consistency': triple_score.get('consistency', 0),
                        'combined': triple_score.get('combined', 0),
                        'z_score': triple_score.get('z_score', 0),
                        'kept': triple_score.get('kept', False),
                        'error_type': triple_score.get('error_type', None)
                    })
    
    triples_df = pd.DataFrame(triple_rows)
    
    # Calculate total relations
    total_relations = 0
    total_kept = 0
    total_filtered = 0
    
    for result in sentences_data:
        total_relations += result.get('total_triples', 0)
        total_kept += result.get('kept_triples', 0)
        total_filtered += result.get('filtered_out', 0)
    
    return {
        'sentences_data': sentences_data,
        'results_df': results_df,
        'triples_df': triples_df,
        'summary': {
            'total_sentences': len(sentences_data),
            'total_relations': total_relations,
            'total_kept': total_kept,
            'total_filtered': total_filtered,
            'overall_precision': total_kept / total_relations if total_relations > 0 else 0,
            'avg_precision': results_df['precision'].mean() if 'precision' in results_df else 0,
            'avg_total_triples': results_df['total_triples'].mean() if 'total_triples' in results_df else 0,
            'avg_kept_triples': results_df['kept_triples'].mean() if 'kept_triples' in results_df else 0
        }
    }



# Create visualizations
def create_analysis_plots(analysis):
    """Create figures for the analysis."""
    triples_df = analysis['triples_df']
    results_df = analysis['results_df']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Distribution of combined scores by kept/filtered status
    ax1 = axes[0, 0]
    kept_scores = triples_df[triples_df['kept']]['combined']
    filtered_scores = triples_df[~triples_df['kept']]['combined']
    
    ax1.hist(kept_scores, bins=20, alpha=0.7, label='Kept', color='green', edgecolor='black')
    ax1.hist(filtered_scores, bins=20, alpha=0.7, label='Filtered', color='red', edgecolor='black')
    ax1.set_xlabel('Combined Score')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Combined Scores')
    ax1.legend()
    
    # 2. Z-score distribution by kept/filtered status
    ax2 = axes[0, 1]
    ax2.hist(triples_df[triples_df['kept']]['z_score'], bins=20, alpha=0.7, label='Kept', color='green', edgecolor='black')
    ax2.hist(triples_df[~triples_df['kept']]['z_score'], bins=20, alpha=0.7, label='Filtered', color='red', edgecolor='black')
    ax2.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Z-score')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Z-scores')
    ax2.legend()
    
    # 3. Precision per extraction version
    ax3 = axes[0, 2]
    version_precision = results_df.groupby('extraction_version')['precision'].mean()
    version_precision.plot(kind='bar', ax=ax3, color='steelblue', edgecolor='black')
    ax3.set_xlabel('Extraction Version')
    ax3.set_ylabel('Precision')
    ax3.set_title('Precision by Extraction Version')
    ax3.set_ylim(0, 1)
    
    # 4. Scatter plot: Relevance vs Consistency colored by kept/filtered
    ax4 = axes[1, 0]
    colors = triples_df['kept'].map({True: 'green', False: 'red'})
    scatter = ax4.scatter(triples_df['relevance'], triples_df['consistency'], 
                          c=colors, alpha=0.6, s=30)
    ax4.set_xlabel('Relevance')
    ax4.set_ylabel('Consistency')
    ax4.set_title('Relevance vs Consistency')
    # Add legend manually
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='green', label='Kept'),
                       Patch(facecolor='red', label='Filtered')]
    ax4.legend(handles=legend_elements)
    
    # 5. Number of filtered triples by error type
    ax5 = axes[1, 1]
    error_counts = triples_df[~triples_df['kept']]['error_type'].value_counts()
    if not error_counts.empty:
        error_counts.plot(kind='bar', ax=ax5, color='coral', edgecolor='black')
        ax5.set_xlabel('Error Type')
        ax5.set_ylabel('Count')
        ax5.set_title('Filtered Triples by Error Type')
    else:
        ax5.text(0.5, 0.5, 'No filtered triples', transform=ax5.transAxes, ha='center', va='center')
        ax5.set_title('Filtered Triples by Error Type')
    
    # 6. Distribution of triples per sentence
    ax6 = axes[1, 2]
    triples_per_sentence = results_df['total_triples'].value_counts().sort_index()
    triples_per_sentence.plot(kind='bar', ax=ax6, color='teal', edgecolor='black')
    ax6.set_xlabel('Number of Triples per Sentence')
    ax6.set_ylabel('Number of Sentences')
    ax6.set_title('Distribution of Triples per Sentence')
    
    plt.tight_layout()
    return fig
# Additional sensitivity analysis simulation
def simulate_threshold_analysis(triples_df, thresholds):
    """Simulate the effect of different threshold settings on extraction precision and recall."""
    results = []
    
    for threshold in thresholds:
        # Simulate different threshold levels for keeping triples
        # This is a simplified simulation based on the combined score and z-score
        kept_indices = triples_df['combined'] >= threshold
        
        # Calculate metrics
        total = len(triples_df)
        kept = kept_indices.sum()
        filtered = total - kept
        
        # Determine if kept triples are "correct" (based on original kept status)
        # This is a simulation - assuming the original kept status represents ground truth
        true_positives = ((kept_indices) & (triples_df['kept'])).sum()
        false_positives = ((kept_indices) & (~triples_df['kept'])).sum()
        false_negatives = ((~kept_indices) & (triples_df['kept'])).sum()
        true_negatives = ((~kept_indices) & (~triples_df['kept'])).sum()
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        results.append({
            'threshold': threshold,
            'kept': kept,
            'filtered': filtered,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
    
    return pd.DataFrame(results)
# Plot sensitivity analysis
def plot_sensitivity_analysis(sensitivity_results):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(sensitivity_results['threshold'], sensitivity_results['precision'], 
            marker='o', label='Precision', linewidth=2)
    ax.plot(sensitivity_results['threshold'], sensitivity_results['recall'], 
            marker='s', label='Recall', linewidth=2)
    ax.plot(sensitivity_results['threshold'], sensitivity_results['f1'], 
            marker='^', label='F1 Score', linewidth=2)
    
    # Find optimal threshold
    optimal_idx = sensitivity_results['f1'].idxmax()
    optimal_threshold = sensitivity_results.loc[optimal_idx, 'threshold']
    optimal_f1 = sensitivity_results.loc[optimal_idx, 'f1']
    
    ax.axvline(x=optimal_threshold, color='red', linestyle='--', alpha=0.7)
    ax.text(optimal_threshold + 0.02, 0.5, f'Optimal: {optimal_threshold:.2f}', 
            rotation=90, fontsize=10)
    
    ax.set_xlabel('Threshold', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Sensitivity Analysis: Effect of Threshold on Extraction Performance', fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    
    plt.tight_layout()
    return fig
def save_figure(fig,args, filename, dpi=300, bbox_inches='tight'):
    """Helper function to save figures with consistent settings."""
    fig.savefig(args.PathDataset+filename,
        dpi=dpi,
        bbox_inches=bbox_inches,
        facecolor='white',
        edgecolor='none'
    )
    print(f"Saved: {filename}")
# Figure 1: Distribution of Combined Scores
def create_figure_1(analysis):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    triples_df = analysis['triples_df']
    kept_scores = triples_df[triples_df['kept']]['combined']
    filtered_scores = triples_df[~triples_df['kept']]['combined']
    
    ax.hist(kept_scores, bins=20, alpha=0.7, label='Kept', color='green', edgecolor='black')
    ax.hist(filtered_scores, bins=20, alpha=0.7, label='Filtered', color='red', edgecolor='black')
    ax.set_xlabel('Combined Score', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Figure 1: Distribution of Combined Scores by Keep/Filter Status', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig

# Figure 2: Distribution of Z-scores
def create_figure_2(analysis):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    triples_df = analysis['triples_df']
    ax.hist(triples_df[triples_df['kept']]['z_score'], bins=20, alpha=0.7, 
            label='Kept', color='green', edgecolor='black')
    ax.hist(triples_df[~triples_df['kept']]['z_score'], bins=20, alpha=0.7, 
            label='Filtered', color='red', edgecolor='black')
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax.axvline(x=1, color='blue', linestyle='--', alpha=0.5, label='Z-score = 1 (current threshold)')
    ax.axvline(x=-1, color='blue', linestyle='--', alpha=0.5)
    ax.set_xlabel('Z-score', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Figure 2: Distribution of Z-scores by Keep/Filter Status', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig

# Figure 3: Precision by Extraction Version
def create_figure_3(analysis):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    results_df = analysis['results_df']
    version_precision = results_df.groupby('extraction_version')['precision'].mean()
    version_precision_filled = version_precision.fillna(0)
    bars = version_precision_filled.plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
    ax.set_xlabel('Extraction Version', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Figure 3: Precision by Extraction Version', fontsize=14)
    ax.set_ylim(0, 1)
    ax.axhline(y=analysis['summary']['overall_precision'], color='red', 
               linestyle='--', alpha=0.7, label=f'Overall Precision: {analysis["summary"]["overall_precision"]:.3f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars.patches:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    return fig

# Figure 4: Relevance vs Consistency Scatter Plot
def create_figure_4(analysis):
    fig, ax = plt.subplots(figsize=(10, 8))
    
    triples_df = analysis['triples_df']
    colors = triples_df['kept'].map({True: 'green', False: 'red'})
    scatter = ax.scatter(triples_df['relevance'], triples_df['consistency'], 
                         c=colors, alpha=0.6, s=50)
    ax.set_xlabel('Relevance', fontsize=12)
    ax.set_ylabel('Consistency', fontsize=12)
    ax.set_title('Figure 4: Relevance vs Consistency (Colored by Keep/Filter Status)', fontsize=14)
    
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='green', label='Kept'),
                       Patch(facecolor='red', label='Filtered')]
    ax.legend(handles=legend_elements)
    ax.grid(True, alpha=0.3)
    
    return fig

# Figure 5: Filtered Triples by Error Type
def create_figure_5(analysis):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    triples_df = analysis['triples_df']
    filtered_df = triples_df[~triples_df['kept']]
    
    if not filtered_df.empty:
        error_counts = filtered_df['error_type'].value_counts()
        bars = error_counts.plot(kind='bar', ax=ax, color='coral', edgecolor='black')
        ax.set_xlabel('Error Type', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Figure 5: Filtered Triples by Error Type', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for bar in bars.patches:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{int(height)}', ha='center', va='bottom', fontsize=10)
    else:
        ax.text(0.5, 0.5, 'No filtered triples', transform=ax.transAxes, 
                ha='center', va='center', fontsize=14)
        ax.set_title('Figure 5: No Filtered Triples Found', fontsize=14)
    
    return fig

# Figure 6: Distribution of Triples per Sentence
def create_figure_6(analysis):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    results_df = analysis['results_df']
    triples_per_sentence = results_df['total_triples'].value_counts().sort_index()
    
    bars = triples_per_sentence.plot(kind='bar', ax=ax, color='teal', edgecolor='black')
    ax.set_xlabel('Number of Triples per Sentence', fontsize=12)
    ax.set_ylabel('Number of Sentences', fontsize=12)
    ax.set_title('Figure 6: Distribution of Triples per Sentence', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Add value labels
    for bar in bars.patches:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{int(height)}', ha='center', va='bottom', fontsize=10)
    
    return fig

# Figure 7: Sensitivity Analysis - Effect of Threshold
def create_figure_7(sensitivity_results, current_threshold=0.5):
    fig, ax = plt.subplots(figsize=(12, 7))
    
    ax.plot(sensitivity_results['threshold'], sensitivity_results['precision'], 
            marker='o', label='Precision', linewidth=2, markersize=8)
    ax.plot(sensitivity_results['threshold'], sensitivity_results['recall'], 
            marker='s', label='Recall', linewidth=2, markersize=8)
    ax.plot(sensitivity_results['threshold'], sensitivity_results['f1'], 
            marker='^', label='F1 Score', linewidth=2, markersize=8)
    
    # Find optimal threshold
    optimal_idx = sensitivity_results['f1'].idxmax()
    optimal_threshold = sensitivity_results.loc[optimal_idx, 'threshold']
    optimal_f1 = sensitivity_results.loc[optimal_idx, 'f1']
    
    # Mark current threshold
    ax.axvline(x=current_threshold, color='blue', linestyle='--', alpha=0.7, 
               label=f'Current threshold: {current_threshold}')
    
    # Mark optimal threshold
    ax.axvline(x=optimal_threshold, color='red', linestyle='--', alpha=0.7, 
               label=f'Optimal threshold: {optimal_threshold:.2f}')
    
    # Add annotation for optimal point
    ax.plot(optimal_threshold, optimal_f1, 'ro', markersize=12)
    ax.annotate(f'F1 = {optimal_f1:.3f}\nThreshold = {optimal_threshold:.2f}',
                xy=(optimal_threshold, optimal_f1),
                xytext=(optimal_threshold + 0.05, optimal_f1 - 0.05),
                fontsize=10, bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
    
    ax.set_xlabel('Threshold', fontsize=13)
    ax.set_ylabel('Score', fontsize=13)
    ax.set_title('Figure 7: Sensitivity Analysis - Effect of Threshold on Performance', fontsize=15)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=11)
    
    return fig

# Figure 8: Combined Summary Dashboard
def create_figure_8(analysis, sensitivity_results):
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('Extraction Performance Dashboard', fontsize=16, fontweight='bold')
    
    # Create subplot grid
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Subplot 1: Overall metrics
    ax1 = fig.add_subplot(gs[0, 0])
    triples_df = analysis['triples_df']
    metrics = {
        'Total Triples': len(triples_df),
        'Kept': triples_df['kept'].sum(),
        'Filtered': (~triples_df['kept']).sum(),
        'Precision': analysis['summary']['overall_precision']
    }
    bars = ax1.bar(metrics.keys(), metrics.values(), 
                   color=['#2E86AB', '#2ECC71', '#E74C3C', '#F39C12'])
    ax1.set_title('Overall Metrics', fontsize=12)
    ax1.set_ylabel('Count', fontsize=10)
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height:.3f}' if isinstance(height, float) else f'{int(height)}', 
                ha='center', va='bottom', fontsize=9)
    
    # Subplot 2: Score distributions
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.boxplot([triples_df['relevance'].values, 
                 triples_df['consistency'].values, 
                 triples_df['combined'].values],
                )
    ax2.set_title('Score Distributions', fontsize=12)
    ax2.set_ylabel('Score', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Subplot 3: Kept vs Filtered counts by extraction version
    ax3 = fig.add_subplot(gs[0, 2])
    version_stats = triples_df.groupby('extraction_version')['kept'].agg(['sum', lambda x: (~x).sum()])
    version_stats.columns = ['Kept', 'Filtered']
    version_stats.plot(kind='bar', ax=ax3, color=['#2ECC71', '#E74C3C'], edgecolor='black')
    ax3.set_title('Keep/Filter by Extraction Version', fontsize=12)
    ax3.set_xlabel('Extraction Version', fontsize=10)
    ax3.set_ylabel('Count', fontsize=10)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Subplot 4: Optimal threshold highlight
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(sensitivity_results['threshold'], sensitivity_results['f1'], 
             'g-', linewidth=2, label='F1 Score')
    ax4.fill_between(sensitivity_results['threshold'], 0, sensitivity_results['f1'], 
                     alpha=0.3, color='green')
    optimal_idx = sensitivity_results['f1'].idxmax()
    optimal_threshold = sensitivity_results.loc[optimal_idx, 'threshold']
    optimal_f1 = sensitivity_results.loc[optimal_idx, 'f1']
    ax4.plot(optimal_threshold, optimal_f1, 'ro', markersize=10)
    ax4.annotate(f'Optimal: {optimal_threshold:.2f}', 
                xy=(optimal_threshold, optimal_f1),
                xytext=(optimal_threshold + 0.02, optimal_f1 - 0.1),
                fontsize=10, arrowprops=dict(arrowstyle='->', color='black'))
    ax4.set_xlabel('Threshold', fontsize=10)
    ax4.set_ylabel('F1 Score', fontsize=10)
    ax4.set_title('F1 Score vs Threshold', fontsize=12)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 1)
    
    # Subplot 5: Error type distribution (pie chart)
    ax5 = fig.add_subplot(gs[1, 1])
    filtered_df = triples_df[~triples_df['kept']]
    if not filtered_df.empty:
        error_counts = filtered_df['error_type'].value_counts()
        ax5.pie(error_counts.values, autopct='%1.1f%%',
                colors=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        ax5.set_title('Error Type Distribution', fontsize=12)
    else:
        ax5.text(0.5, 0.5, 'No errors', transform=ax5.transAxes, ha='center', va='center')
        ax5.set_title('No errors to display', fontsize=12)
    
    
    plt.tight_layout()
    return fig
# Comprehensive solution with error handling
def calculate_improvement(sensitivity_results, optimal_f1, target=0.5):
    # Find nearest threshold
    nearest_idx = (sensitivity_results['threshold'] - target).abs().idxmin()
    nearest_threshold = sensitivity_results.loc[nearest_idx, 'threshold']
    baseline_f1 = sensitivity_results.loc[nearest_idx, 'f1']
    
    # Calculate improvement
    improvement = ((optimal_f1 - baseline_f1) / baseline_f1 * 100)
    
    # Print with context
    print(f"Baseline F1 (threshold={nearest_threshold:.3f}): {baseline_f1:.4f}")
    print(f"Optimal F1: {optimal_f1:.4f}")
    print(f"Improvement: {improvement:+.1f}%")
    
    return improvement
def mainanalysis2(data,args):
    # Run the analysis
    analysis = analyze_extractions(data)

    print("=" * 60)
    print("EXTRACTION ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"Total sentences processed: {analysis['summary']['total_sentences']}")
    print(f"Total relations identified: {analysis['summary']['total_relations']}")
    print(f"Total relations kept: {analysis['summary']['total_kept']}")
    print(f"Total relations filtered out: {analysis['summary']['total_filtered']}")
    print(f"Overall precision: {analysis['summary']['overall_precision']:.3f}")
    print(f"Average precision per extraction: {analysis['summary']['avg_precision']:.3f}")
    print(f"Average triples per sentence: {analysis['summary']['avg_total_triples']:.2f}")
    print(f"Average kept triples per sentence: {analysis['summary']['avg_kept_triples']:.2f}")

    # 1. TOTAL NUMBER OF RELATIONS THAT SHOULD HAVE BEEN IDENTIFIED
    print("\n" + "=" * 60)
    print("TOTAL RELATIONS ANALYSIS")
    print("=" * 60)
    print(f"Total number of relations that should have been identified: {analysis['summary']['total_relations']}")
    print(f"Number of relations successfully kept: {analysis['summary']['total_kept']}")
    print(f"Number of relations incorrectly filtered: {analysis['summary']['total_filtered']}")
    print(f"Overall recall of relations: {analysis['summary']['overall_precision']:.3f}")

    # Error breakdown analysis
    error_counts = defaultdict(int)
    for result in analysis['sentences_data']:
        error_breakdown = result.get('error_breakdown', {})
        for error_type, count in error_breakdown.items():
            error_counts[error_type] += count

    print("\nError Breakdown:")
    for error_type, count in error_counts.items():
        print(f"  {error_type}: {count}")

    # Analyze the per-triple scores distribution
    triples_df = analysis['triples_df']

    print("\n" + "=" * 60)
    print("PER-TRIPLE SCORE STATISTICS")
    print("=" * 60)
    print(f"Total triples analyzed: {len(triples_df)}")
    print(f"Triples kept: {triples_df['kept'].sum()}")
    print(f"Triples filtered: {(~triples_df['kept']).sum()}")
    print(f"\nScore statistics:")
    print(f"  Relevance - mean: {triples_df['relevance'].mean():.3f}, std: {triples_df['relevance'].std():.3f}")
    print(f"  Consistency - mean: {triples_df['consistency'].mean():.3f}, std: {triples_df['consistency'].std():.3f}")
    print(f"  Combined - mean: {triples_df['combined'].mean():.3f}, std: {triples_df['combined'].std():.3f}")
    print(f"  Z-score - mean: {triples_df['z_score'].mean():.3f}, std: {triples_df['z_score'].std():.3f}")
    # =============================================
    # SAVE ALL FIGURES
    # =============================================

    print("\n" + "=" * 60)
    print("SAVING FIGURES...")
    print("=" * 60)

    # Create and save each figure
    figures = [
        (create_figure_1, 'figure_1_combined_scores_distribution.png'),
        (create_figure_2, 'figure_2_zscore_distribution.png'),
        (create_figure_3, 'figure_3_precision_by_version.png'),
        (create_figure_4, 'figure_4_relevance_vs_consistency.png'),
        (create_figure_5, 'figure_5_error_types.png'),
        (create_figure_6, 'figure_6_triples_per_sentence.png'),
        (create_figure_7, 'figure_7_sensitivity_analysis.png'),
    ]
    # Sensitivity analysis: varying threshold levels
    thresholds = np.arange(0.3, 0.95, 0.05)
    sensitivity_results = simulate_threshold_analysis(triples_df, thresholds)

    # Print sensitivity analysis results
    print("\n" + "=" * 60)
    print("SENSITIVITY ANALYSIS: EFFECT OF THRESHOLD SETTINGS")
    print("=" * 60)
    print("\nThreshold | Kept | Filtered | Precision | Recall | F1")
    print("-" * 60)
    for _, row in sensitivity_results.iterrows():
        print(f"{row['threshold']:.2f}     | {row['kept']:4f} | {row['filtered']:6f} | {row['precision']:.3f}    | {row['recall']:.3f} | {row['f1']:.3f}")

    # Create and save the dashboard
    print("\nCreating dashboard figure...")
    dashboard_fig = create_figure_8(analysis, sensitivity_results)
    save_figure(dashboard_fig,args,'figure_8_dashboard.png', dpi=150)
    plt.close(dashboard_fig)
    # Create the plots
    fig = create_analysis_plots(analysis)
    #plt.show()
    save_figure(fig,args,'create_analysis_plots.png', dpi=150)
    plt.close(fig)
    # Plot sensitivity analysis
    sensitivity_fig = plot_sensitivity_analysis(sensitivity_results)
    #plt.show()
    save_figure(sensitivity_fig,args,'plot_sensitivity_analysis.png', dpi=150)
    plt.close(sensitivity_fig)

    # Additional analysis: optimal threshold
    optimal_threshold = sensitivity_results.loc[sensitivity_results['f1'].idxmax(), 'threshold']
    optimal_precision = sensitivity_results.loc[sensitivity_results['f1'].idxmax(), 'precision']
    optimal_recall = sensitivity_results.loc[sensitivity_results['f1'].idxmax(), 'recall']
    optimal_f1 = sensitivity_results.loc[sensitivity_results['f1'].idxmax(), 'f1']

    print("\n" + "=" * 60)
    print("OPTIMAL THRESHOLD ANALYSIS")
    print("=" * 60)
    print(f"Optimal threshold (maximizing F1): {optimal_threshold:.2f}")
    print(f"Precision at optimal threshold: {optimal_precision:.3f}")
    print(f"Recall at optimal threshold: {optimal_recall:.3f}")
    print(f"F1 Score at optimal threshold: {optimal_f1:.3f}")

    # Summary statistics for report
    print("\n" + "=" * 60)
    print("FINAL SUMMARY FOR REPORT")
    print("=" * 60)
    print(f"""
    1. Total number of relations identified: {analysis['summary']['total_relations']}
    2. Number of relations successfully extracted (kept): {analysis['summary']['total_kept']}
    3. Number of relations filtered out: {analysis['summary']['total_filtered']}
    4. Overall precision: {analysis['summary']['overall_precision']:.3f} ({analysis['summary']['overall_precision']*100:.1f}%)
    5. Average precision per extraction: {analysis['summary']['avg_precision']:.3f}

    Threshold Sensitivity Analysis:
    - Current threshold used: 0.5 (based on Z-score = 1)
    - Optimal threshold: {optimal_threshold:.2f} (would maximize F1 score)
    """)
    improvement = calculate_improvement(sensitivity_results, optimal_f1)

    # Output to CSV for further analysis
    analysis['triples_df'].to_csv(args.PathDataset+'triples_analysis.csv', index=False)
    analysis['results_df'].to_csv(args.PathDataset+'extraction_results.csv', index=False)
    sensitivity_results.to_csv(args.PathDataset+'sensitivity_analysis.csv', index=False)

    print("\nAnalysis results saved to CSV files:")
    print("  - triples_analysis.csv: Individual triple-level data")
    print("  - extraction_results.csv: Sentence-level extraction results")
    print("  - sensitivity_analysis.csv: Threshold sensitivity analysis")