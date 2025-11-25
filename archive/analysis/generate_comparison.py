import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
RESULT_DIR = ROOT_DIR / "data" / "results"
DIFF_LOG = RESULT_DIR / "lbracket_diff_fea_log.csv"
MMC_LOGS = [
    RESULT_DIR / "mmc_log_lbracket.csv",
    RESULT_DIR / "mmc_log_vertical.csv",
]
OUTPUT_TABLE = RESULT_DIR / "method_comparison.csv"
COMPLIANCE_PLOT = RESULT_DIR / "comparison_compliance.png"
TIME_PLOT = RESULT_DIR / "comparison_time.png"


def summarize_diff():
    df = pd.read_csv(DIFF_LOG)
    comp = df["compliance"].iloc[-1]
    time_s = df["wall_time_ms"].iloc[1:].mean() / 1000.0
    return {
        "method": "diff_fea",
        "tag": "lbracket",
        "compliance": comp,
        "time_per_iter": time_s,
        "source": DIFF_LOG.name,
    }


def summarize_mmc(path):
    df = pd.read_csv(path)
    comp = df["compliance"].iloc[-1]
    time_s = df["wall_time_ms"].mean() / 1000.0
    return {
        "method": "mmc",
        "tag": path.stem.replace("mmc_log_", ""),
        "compliance": comp,
        "time_per_iter": time_s,
        "source": path.name,
    }


def plot_bar(df, column, path, ylabel, title):
    plt.figure(figsize=(6, 4))
    plt.bar(df["label"], df[column], color=["#6baed6", "#fd8d3c", "#74c476"])
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def main():
    records = [summarize_diff()]
    for path in MMC_LOGS:
        if path.exists():
            records.append(summarize_mmc(path))

    df = pd.DataFrame(records)
    df["label"] = df["method"] + " (" + df["tag"] + ")"
    df.to_csv(OUTPUT_TABLE, index=False)
    print(f"Wrote comparison table to {OUTPUT_TABLE}")

    plot_bar(df, "compliance", COMPLIANCE_PLOT, "Compliance (J)", "Final Compliance per Method")
    plot_bar(df, "time_per_iter", TIME_PLOT, "Time per Iteration (s)", "Runtime per Method")
    print(f"Saved plots to {COMPLIANCE_PLOT} and {TIME_PLOT}")


if __name__ == "__main__":
    main()

