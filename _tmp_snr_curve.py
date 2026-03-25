import csv
from pathlib import Path
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

root = Path('outputs/phase6/controlled_snr_test31/snr_10_15')
clean_dir = root / 'clean'
noisy_dir = root / 'noisy'
pred_dir = Path('outputs/phase9/nhfae_e1_interact_tune1/wav')
out_dir = Path('outputs/phase9/nhfae_e1_interact_tune1')
out_dir.mkdir(parents=True, exist_ok=True)

rows = []
for c in sorted(clean_dir.glob('*.wav')):
    n = noisy_dir / c.name
    p = pred_dir / c.name
    if not n.exists() or not p.exists():
        continue
    xc, _ = sf.read(c, dtype='float32')
    xn, _ = sf.read(n, dtype='float32')
    xp, _ = sf.read(p, dtype='float32')
    L = min(len(xc), len(xn), len(xp))
    xc = xc[:L]
    xn = xn[:L]
    xp = xp[:L]
    eps = 1e-8
    snr_in = 10.0 * np.log10((np.sum(xc**2)+eps)/(np.sum((xn-xc)**2)+eps))
    sdr_noisy = 10.0 * np.log10((np.sum(xc**2)+eps)/(np.sum((xn-xc)**2)+eps))
    sdr_pred = 10.0 * np.log10((np.sum(xc**2)+eps)/(np.sum((xp-xc)**2)+eps))
    d_sdr = sdr_pred - sdr_noisy
    rows.append((c.stem, snr_in, d_sdr))

csv_path = out_dir / 'snr_vs_delta_sdr.csv'
with csv_path.open('w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['name','snr_in_db','delta_sdr_db'])
    w.writerows(rows)

x = np.array([r[1] for r in rows], dtype=np.float64)
y = np.array([r[2] for r in rows], dtype=np.float64)
order = np.argsort(x)
xs = x[order]
ys = y[order]

win = max(3, len(xs)//8)
ys_smooth = np.convolve(ys, np.ones(win)/win, mode='same')

plt.figure(figsize=(7,5))
plt.scatter(x, y, s=12, alpha=0.65, label='samples')
plt.plot(xs, ys_smooth, linewidth=2.0, label='moving avg')
plt.axhline(0.0, linestyle='--', linewidth=1)
plt.xlabel('SNR_in (dB)')
plt.ylabel('delta SDR (dB)')
plt.title('E1+ Interact tune1: SNR vs delta SDR')
plt.legend()
plt.tight_layout()
plt.savefig(out_dir / 'snr_vs_delta_sdr.png', dpi=160)
print('Saved', csv_path)
print('Saved', out_dir / 'snr_vs_delta_sdr.png')
