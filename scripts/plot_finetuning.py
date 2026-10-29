"""Plot saved training evidence without running the model or revisiting test data."""
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import ROOT, setup_cache
setup_cache()
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    directory = ROOT / 'outputs' / 'finetuning'
    history = json.loads((directory / 'training_history.json').read_text())
    manifest = json.loads((directory / 'training_manifest.json').read_text())
    losses = [row for row in history if 'loss' in row]
    epochs = {}
    for row in history:
        if 'eval_macro_f1' in row:
            # The final row re-evaluates restored epoch-2 weights at step 636;
            # keep the original epoch-3 result, not that duplicate evaluation.
            epochs.setdefault(int(row['step']), row)
    records = list(epochs.values())
    selected_step = int(manifest['selected_checkpoint'].rsplit('-', 1)[1])
    selected = epochs[selected_step]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), layout='constrained')
    axes[0].plot([row['step'] for row in losses], [row['loss'] for row in losses], color='#028090', linewidth=2)
    axes[0].set(title='Training loss', xlabel='Optimizer step', ylabel='Logged interval mean loss')
    axes[1].plot([row['epoch'] for row in records], [100 * row['eval_macro_f1'] for row in records], color='#0B3C49', marker='o', linewidth=2)
    axes[1].scatter([selected['epoch']], [100 * selected['eval_macro_f1']], color='#F2A541', s=95, zorder=3, label='Selected checkpoint')
    for row in records:
        axes[1].annotate(f"{100 * row['eval_macro_f1']:.2f}%", (row['epoch'], 100 * row['eval_macro_f1']), xytext=(0, 9), textcoords='offset points', ha='center')
    axes[1].set(title='Validation checkpoint selection', xlabel='Epoch', ylabel='Validation macro-F1 (%)', xticks=[1, 2, 3], ylim=(80.8, 82.4))
    axes[1].legend(loc='lower right', frameon=False)
    for axis in axes:
        axis.spines[['top', 'right']].set_visible(False)
        axis.grid(axis='y', alpha=0.2)
    path = directory / 'training_curves.png'
    fig.savefig(path, dpi=180, facecolor='white')
    plt.close(fig)
    print(path)


if __name__ == '__main__':
    main()
