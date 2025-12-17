import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

STATS_PATH = "data/V3/gzip_stats.tsv"

df = pd.read_csv(STATS_PATH, sep="\t")
print("Rows:", len(df))
print("Kept:", int(df["kept"].sum()), f"({df['kept'].mean()*100:.2f}%)")


qs = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
print("\nRatio quantiles:")
print(df["ratio"].quantile(qs).to_string())


plt.figure()
plt.hist(df["ratio"].values, bins=80)
plt.xlabel("gzip_ratio = gz_bytes / raw_bytes")
plt.ylabel("count")
plt.title("Distribution of gzip_ratio")
plt.tight_layout()
plt.savefig("gzip_ratio_hist.png", dpi=200)


thresholds = np.arange(0.10, 0.401, 0.005)
kept_frac = np.array([(df["ratio"] >= t).mean() for t in thresholds])

chosen_t = 0.24
chosen_keep = (df["ratio"] >= chosen_t).mean()

plt.figure()
plt.plot(thresholds, kept_frac)
plt.xlabel("MIN_RATIO threshold")
plt.ylabel("fraction kept")
plt.title("Kept fraction vs gzip_ratio threshold")


plt.axvline(chosen_t, linestyle="--")
plt.axhline(chosen_keep, linestyle="--")
plt.scatter([chosen_t], [chosen_keep], zorder=5)

plt.text(
    chosen_t, chosen_keep,
    f"  t={chosen_t:.2f}, kept={chosen_keep*100:.1f}%",
    va="bottom", ha="left"
)

plt.tight_layout()
plt.savefig("gzip_threshold_curve.png", dpi=200)
