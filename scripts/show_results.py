"""Display saved controls and plot a compact comparison. No training occurs."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def main():
    ensembles = pd.read_csv(ROOT / 'results/controls/ensemble_scores.csv')
    print('Saved pair-score means; these are not scores of pooled predictions.')
    print(ensembles[['region', 'model', 'sequence_R2', 'lead24_R2']].to_string(index=False))
    scale = pd.read_csv(ROOT / 'results/controls/scale_scores_by_seed.csv')
    scale = scale.groupby(['region', 'scale'])[['CRPS', 'NLL', 'coverage_95', 'width_95']].mean()
    print('\nSame C4 mean; saved five-run probability scores:')
    print(scale.to_string())
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    names = ['Eta+Eta', 'Multi+Multi', 'Eta+Multi_distinct']
    for ax, region in zip(axes, ['historical_7', 'external_10']):
        part = ensembles[ensembles.region == region].set_index('model').loc[names]
        ax.bar(['Eta + Eta', 'Multi + Multi', 'Eta + Multi'], part.sequence_R2, color=['#63859f','#337364','#bd8352'])
        ax.set_ylabel('Sequence R² (mean pair score)')
        ax.set_title('Seven stations' if region == 'historical_7' else 'Ten stations')
        ax.set_ylim(0, 1)
        for i, value in enumerate(part.sequence_R2):
            ax.text(i, value + .015, f'{value:.4f}', ha='center')
    out = ROOT / 'output'
    out.mkdir(exist_ok=True)
    fig.savefig(out / 'ensemble_controls.png', dpi=160)
    plt.close(fig)
    print('\nSaved figure: output/ensemble_controls.png')

if __name__ == '__main__':
    main()
