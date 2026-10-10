"""Step 1: fit the LSA topic model on the filtered tweets and write its diagnostics.

    python fit_topics.py

Outputs: outputs/tables/lsa_poles.{csv,md}, outputs/tables/lsa_spectrum.csv,
outputs/figures/lsa_spectrum.png, outputs/lsa_model.pkl
"""
import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import config, nlp_svd_topics as lsa, nlp_topics as nt  # noqa: E402
from src.table_generator import to_markdown  # noqa: E402


def main() -> None:
    tweets, _ = nt.quality_filter(nt.load_cached_tweets())
    tweets = tweets.reset_index(drop=True)
    texts = tweets["text"]
    m = lsa.fit_lsa(texts, k=config.LSA_K)
    stab = lsa.label_stability(texts, m, n_boot=5)
    poles = m.poles.merge(stab[["component", "sign", "stability"]],
                          on=["component", "sign"], how="left")
    poles.to_csv(config.TABLE_DIR / "lsa_poles.csv", index=False)
    shown = poles[~poles["label"].isin(["Other"])].copy()
    shown["top_terms"] = shown["top_terms"].str.split(", ").str[:8].str.join(", ")
    (config.TABLE_DIR / "lsa_poles.md").write_text(
        to_markdown(shown.set_index(["component", "sign"]), 2), encoding="utf-8")

    X, _ = lsa.build_tfidf(texts)
    full = lsa.svd_check(X, 60)
    spec = pd.DataFrame({"component": np.arange(1, 61), "singular_value": full})
    spec.to_csv(config.TABLE_DIR / "lsa_spectrum.csv", index=False)
    fig, ax = plt.subplots(1, 2, figsize=(8, 3))
    ax[0].plot(spec["component"], spec["singular_value"], marker=".", lw=1)
    ax[0].axvline(config.LSA_K, color="grey", ls=":")
    ax[0].set(xlabel="component", ylabel="singular value", title="Singular value spectrum")
    energy = np.cumsum(full ** 2) / (X.multiply(X).sum())
    ax[1].plot(spec["component"], 100 * energy, lw=1)
    ax[1].axvline(config.LSA_K, color="grey", ls=":")
    ax[1].set(xlabel="k", ylabel="% of TF-IDF energy", title="Cumulative energy")
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "lsa_spectrum.png", dpi=150)

    W = lsa.topic_weights(m)
    tweets["topic"] = lsa.dominant_topic(W).values
    with open(config.OUTPUT_DIR / "lsa_model.pkl", "wb") as fh:
        pickle.dump({"model": m, "docs": tweets[["id", "created_at", "text", "stratum", "topic"]]},
                    fh)
    print(f"docs {len(texts):,}; TF-IDF {X.shape}; k={config.LSA_K}; "
          f"energy captured {100 * energy[config.LSA_K - 1]:.1f}%")
    print(tweets.groupby("stratum")["topic"].value_counts().unstack(0).fillna(0).astype(int))


if __name__ == "__main__":
    main()
