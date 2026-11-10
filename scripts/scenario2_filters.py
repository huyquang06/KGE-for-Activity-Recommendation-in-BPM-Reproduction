import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
META_FILE = os.path.join(BASE_DIR, "results", "test_meta.csv")
CONTEXT_FILE = os.path.join(
    BASE_DIR,
    "local",
    "KGE-ActivityRecommendation",
    "evaluation_test_context",
    "test_context.txt",
)
OUTPUT_CSV = os.path.join(BASE_DIR, "results", "scenario2_ablation.csv")
ERROR_CSV = os.path.join(BASE_DIR, "results", "scenario2_error_cases.csv")
FIG_DIR = os.path.join(BASE_DIR, "figures")
FIG_FILE = os.path.join(FIG_DIR, "scenario2_ablation.png")


def find_prediction_file():
    candidates = [
        os.path.join(
            BASE_DIR,
            "local",
            "new results",
            "transe_onlyAfter_main-TEST.txt",
        ),
    ]
    candidates.extend(
        glob.glob(os.path.join(BASE_DIR, "local", "new results", "*transe*-TEST.txt"))
    )
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def parse_contexts():
    contexts = {}
    with open(CONTEXT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            tokens = line.strip().split("\t")
            for t in tokens[1:]:
                if len(t.split(",")) == 1 and "(" in t:
                    label = t.split("(")[0]
                    node = t.split("(")[1].replace(")", "")
                    proc = "_".join(node.split("_")[1:])
                    if proc not in contexts:
                        contexts[proc] = set()
                    contexts[proc].add(label)
    return contexts


def evaluate_filter_config(df_meta, tails_dict, contexts, config_name):
    ranks = []
    top5_list = []

    for _, row in df_meta.iterrows():
        proc = str(row["process_id"])
        true_lbl = str(row["true_label"])

        matching_preds = []
        for node, preds in tails_dict.items():
            if node.endswith(proc):
                matching_preds = preds
                break

        proc_known_labels = contexts.get(proc, set())
        filtered = []

        for p in matching_preds:
            # Cấu hình 1: Không lọc gì
            if config_name == "1. Raw (Không lọc)":
                filtered.append(p)
            # Cấu hình 2: Chỉ lọc Node (chứa dấu _)
            elif config_name == "2. Only Node Filter":
                if "_" in p:
                    continue
                filtered.append(p)
            # Cấu hình 3: Chỉ lọc nhãn trùng
            elif config_name == "3. Only Dup Filter":
                if p in proc_known_labels:
                    continue
                filtered.append(p)
            # Cấu hình 4: Lọc cả hai (chuẩn tác giả)
            elif config_name == "4. Full Filter (Both)":
                if "_" in p or p in proc_known_labels:
                    continue
                filtered.append(p)

        if true_lbl in filtered:
            r = filtered.index(true_lbl) + 1
        else:
            r = 999999

        ranks.append(r)
        top5_list.append("|".join(filtered[:5]))

    df_res = df_meta.copy()
    df_res["rank"] = ranks
    df_res["top5_preds"] = top5_list

    hits10 = (df_res["rank"] <= 10).mean() * 100.0
    mrr10 = df_res["rank"].apply(lambda r: 1.0 / r if r <= 10 else 0.0).mean()
    mrr_full = df_res["rank"].apply(lambda r: 1.0 / r if r < 999999 else 0.0).mean()

    return (
        {
            "Config": config_name,
            "Hits@10 (%)": hits10,
            "MRR@10": mrr10,
            "Full_MRR": mrr_full,
        },
        df_res,
    )


def run_scenario2():
    pred_file = find_prediction_file()
    if not pred_file:
        print(
            "[-] Không tìm thấy file raw predictions của TransE trong 'local/new"
            " results/'"
        )
        print("    B hãy kiểm tra lại đường dẫn file *-TEST.txt của TransE.")
        return

    print(f"[+] Đang nạp dữ liệu từ: {pred_file}")
    df_meta = pd.read_csv(META_FILE)
    contexts = parse_contexts()

    # Đọc file raw predictions
    tails_dict = {}
    current_node = None
    with open(pred_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if (
                "hasLabel" in line
                and not line.startswith("Tails:")
                and not line.startswith("Heads:")
            ):
                current_node = line.split()[0]
            elif line.startswith("Tails:"):
                preds = line.replace("Tails:", "").strip().split("\t")
                tails_dict[current_node] = preds[0::2]

    configs = [
        "1. Raw (Không lọc)",
        "2. Only Node Filter",
        "3. Only Dup Filter",
        "4. Full Filter (Both)",
    ]

    summary_rows = []
    detailed_dfs = {}

    for cfg in configs:
        metrics, df_detail = evaluate_filter_config(df_meta, tails_dict, contexts, cfg)
        summary_rows.append(metrics)
        detailed_dfs[cfg] = df_detail

    df_summary = pd.DataFrame(summary_rows)
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df_summary.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

    print("\n" + "=" * 70)
    print("KẾT QUẢ ABLATION BỘ LỌC TRANSE (KỊCH BẢN 2)")
    print("=" * 70)
    print(
        f"{'Cấu hình bộ lọc':<25} | {'Hits@10 (%)':<15} | {'MRR@10':<12} |"
        f" {'Full MRR':<10}"
    )
    print("-" * 70)
    for _, r in df_summary.iterrows():
        print(
            f"{r['Config']:<25} | {r['Hits@10 (%)']:>10.2f}%   | {r['MRR@10']:>10.4f} |"
            f" {r['Full_MRR']:>8.4f}"
        )
    print("=" * 70)

    # Trích xuất 5 ca lỗi tiêu biểu (Error Analysis) từ cấu hình Full Filter
    df_full = detailed_dfs["4. Full Filter (Both)"]
    # Lọc các ca đoán sai (rank > 10 hoặc không tìm thấy)
    error_cases = df_full[df_full["rank"] > 10].copy()
    # Chọn đa dạng các ca có độ dài ngữ cảnh khác nhau
    sample_errors = error_cases.sample(n=min(5, len(error_cases)), random_state=42)
    sample_errors.to_csv(ERROR_CSV, index=False, encoding="utf-8")
    print(f"\n[+] Đã xuất 5 case study lỗi phục vụ Error Analysis: {ERROR_CSV}")

    # Vẽ biểu đồ Ablation
    os.makedirs(FIG_DIR, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    cfg_labels = [
        "1. Raw\n(Không lọc)",
        "2. Chỉ lọc Node\n(Type Filter)",
        "3. Chỉ lọc Trùng\n(Dup Filter)",
        "4. Lọc cả hai\n(Full Filter)",
    ]
    colors = ["#7f7f7f", "#3182bd", "#fd8d3c", "#2ca02c"]

    # Cột Hits@10
    bars1 = ax1.bar(
        cfg_labels,
        df_summary["Hits@10 (%)"],
        color=colors,
        edgecolor="black",
        width=0.55,
    )
    ax1.set_ylabel("Hits@10 (%)", fontweight="bold")
    ax1.set_title("Mức độ đóng góp của từng bộ lọc vào Hits@10", fontweight="bold")
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    for bar in bars1:
        h = bar.get_height()
        ax1.annotate(
            f"{h:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # Cột MRR@10
    bars2 = ax2.bar(
        cfg_labels,
        df_summary["MRR@10"],
        color=colors,
        edgecolor="black",
        width=0.55,
    )
    ax2.set_ylabel("MRR@10", fontweight="bold")
    ax2.set_title("Mức độ đóng góp của từng bộ lọc vào MRR@10", fontweight="bold")
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    for bar in bars2:
        h = bar.get_height()
        ax2.annotate(
            f"{h:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(FIG_FILE, dpi=300)
    print(f"[+] Đã xuất biểu đồ bóc tách bộ lọc: {FIG_FILE}")


if __name__ == "__main__":
    run_scenario2()
