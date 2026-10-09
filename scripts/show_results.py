"""Display saved HS-DT and C4 findings. No training occurs."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def main():
    for title, filename in [
        ('Original historical expert comparison', 'expert_comparison.csv'),
        ('Matched historical C4 component comparison', 'c4_components.csv'),
        ('Paired C4 gains and reported confidence intervals', 'c4_paired_sequence.csv'),
    ]:
        print('\n' + title)
        print(pd.read_csv(ROOT / 'results/main_findings' / filename).to_string(index=False))
    scale = pd.read_csv(ROOT / 'results/controls/scale_scores_by_seed.csv')
    scale = scale.groupby(['region', 'scale'])[['CRPS', 'NLL', 'coverage_95', 'width_95']].mean()
    print('\nSame C4 mean; saved five-run probability scores:')
    print(scale.to_string())
    print('\nOrdinary ensemble controls remain available under results/controls/.')

if __name__ == '__main__':
    main()
