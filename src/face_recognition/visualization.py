import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# Try importing TSNE
try:
    from sklearn.manifold import TSNE
    TSNE_AVAILABLE = True
except ImportError:
    TSNE_AVAILABLE = False

def plot_tsne(embeddings, labels, target_names, max_samples=200):
    """
    Applies t-SNE reduction and visualizes 128-D face embeddings in a 2D scatter plot.
    """
    if not TSNE_AVAILABLE:
        print("Scikit-learn is required for t-SNE plotting.")
        return

    print("Extracting embeddings for t-SNE...")

    # Select top 10 people with most images
    top_10_ids = [pid for pid, _ in Counter(labels).most_common(10)]
    mask = np.isin(labels, top_10_ids)

    y_subset = labels[mask][:max_samples]
    embeddings_subset = embeddings[mask][:max_samples]

    print("Running t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(y_subset)-1))
    emb_2d = tsne.fit_transform(embeddings_subset)

    unique_ids = np.unique(y_subset)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_ids)))

    plt.figure(figsize=(10, 7))
    for i, pid in enumerate(unique_ids):
        idx = y_subset == pid
        plt.scatter(emb_2d[idx, 0], emb_2d[idx, 1],
                    c=[colors[i]], label=target_names[pid], s=60, alpha=0.8)

    plt.title("t-SNE of 128-D Face Embeddings\n(Tight clusters represent robust identity discriminability)",
              fontsize=12, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_1n_results(results_log, accuracy_1n):
    """
    Plots horizontal bars showing per-person accuracy and distance histograms of correct vs incorrect matches.
    """
    # 1:N Log accuracy statistics
    per_person_correct = {}
    for r in results_log:
        if r['true'] not in per_person_correct:
            per_person_correct[r['true']] = [0, 0]  # [correct_count, total_count]
        per_person_correct[r['true']][1] += 1
        if r['correct']:
            per_person_correct[r['true']][0] += 1

    names = list(per_person_correct.keys())
    accs  = [per_person_correct[n][0] / per_person_correct[n][1] * 100 for n in names]
    colors_bar = ['steelblue' if a >= 50 else 'tomato' for a in accs]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'1:N Closed-Set Identification  |  Overall Rank-1 Accuracy: {accuracy_1n:.1f}%',
                 fontsize=12, fontweight='bold')

    # Horizontal Bar Plot
    axes[0].barh(names, accs, color=colors_bar, edgecolor='white')
    axes[0].axvline(50, color='gray', linestyle='--', linewidth=1, label='50% line')
    axes[0].set_xlabel('Identification Accuracy (%)')
    axes[0].set_title('Per-Person Rank-1 Accuracy')
    axes[0].set_xlim(0, 105)
    for i, v in enumerate(accs):
        axes[0].text(v + 1, i, f'{v:.0f}%', va='center', fontsize=9)
    axes[0].legend()

    # Distance Distribution Plot
    correct_dists   = [r['dist'] for r in results_log if r['correct']]
    incorrect_dists = [r['dist'] for r in results_log if not r['correct']]

    axes[1].hist(correct_dists,   bins=15, alpha=0.7, color='steelblue', label=f'Correct ({len(correct_dists)})')
    if incorrect_dists:
        axes[1].hist(incorrect_dists, bins=15, alpha=0.7, color='tomato',    label=f'Wrong   ({len(incorrect_dists)})')

    axes[1].set_xlabel('Cosine Distance to Nearest Gallery Embedding')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Distance Distribution')
    axes[1].legend()

    plt.tight_layout()
    plt.show()
