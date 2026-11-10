import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 1. Xác định đường dẫn thư mục gốc
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANYBURL_FILE = os.path.join(BASE_DIR, "results", "anyburl", "ranks.csv")
TRANSE_FILE = os.path.join(BASE_DIR, "results", "transe", "ranks.csv")
OUTPUT_CSV = os.path.join(BASE_DIR, "results", "scenario1_context.csv")
FIG_DIR = os.path.join(BASE_DIR, "figures")
FIG_FILE = os.path.join(FIG_DIR, "scenario1_context.png")


def assign_group(length):
    """Chia độ dài ngữ cảnh thành 3 nhóm cân bằng mẫu"""
    if length <= 3:
        return "1. Ngắn (L <= 3)"
    elif length <= 8:
        return "2. Trung bình (4 <= L <= 8)"
    else:
        return "3. Dài (L >= 9)"


def compute_metrics(df):
    """Tính Hits@10 (%) và MRR theo từng nhóm"""
    n = len(df)
    hits10 = (df["rank"] <= 10).mean() * 100.0
    mrr10 = df["rank"].apply(lambda r: 1.0 / r if r <= 10 else 0.0).mean()
    mrr_full = df["rank"].apply(lambda r: 1.0 / r if r < 999999 else 0.0).mean()
    return pd.Series(
        {"N": n, "Hits@10 (%)": hits10, "MRR@10": mrr10, "Full_MRR": mrr_full}
    )


def run_scenario1():
    # Kiểm tra file đầu vào
    if not os.path.exists(ANYBURL_FILE) or not os.path.exists(TRANSE_FILE):
        print("[-] Lỗi: Không tìm thấy file ranks.csv của AnyBURL hoặc TransE.")
        return

    df_a = pd.read_csv(ANYBURL_FILE)
    df_t = pd.read_csv(TRANSE_FILE)

    # Gán nhóm ngữ cảnh
    df_a["group"] = df_a["context_length"].apply(assign_group)
    df_t["group"] = df_t["context_length"].apply(assign_group)

    # Thống kê theo nhóm
    stats_a = (
        df_a.groupby("group", observed=False).apply(compute_metrics).reset_index()
    )
    stats_a["Model"] = "AnyBURL"

    stats_t = (
        df_t.groupby("group", observed=False).apply(compute_metrics).reset_index()
    )
    stats_t["Model"] = "TransE"

    # Gộp bảng kết quả
    combined = pd.concat([stats_t, stats_a], ignore_index=True)
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    combined.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"[+] Đã lưu bảng thống kê kịch bản 1: {OUTPUT_CSV}\n")

    # In ra màn hình console dạng bảng
    print("=" * 75)
    print(
        f"{'Mô hình':<10} | {'Nhóm ngữ cảnh':<26} | {'N':<4} | "
        f"{'Hits@10':<10} | {'MRR@10':<8}"
    )
    print("-" * 75)
    for _, row in combined.iterrows():
        print(
            f"{row['Model']:<10} | {row['group']:<26} | {int(row['N']):<4} |"
            f" {row['Hits@10 (%)']:>6.2f}%   | {row['MRR@10']:>6.4f}"
        )
    print("=" * 75)

    # 2. Vẽ biểu đồ so sánh trực quan
    os.makedirs(FIG_DIR, exist_ok=True)
    groups = ["Ngắn\n(L ≤ 3)", "Trung bình\n(4 ≤ L ≤ 8)", "Dài\n(L ≥ 9)"]
    x = np.arange(len(groups))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Biểu đồ 1: Hits@10 (%)
    t_hits = stats_t["Hits@10 (%)"].values
    a_hits = stats_a["Hits@10 (%)"].values

    rects1 = ax1.bar(
        x - width / 2,
        t_hits,
        width,
        label="TransE",
        color="#2b5c8f",
        edgecolor="black",
    )
    rects2 = ax1.bar(
        x + width / 2,
        a_hits,
        width,
        label="AnyBURL",
        color="#d95f02",
        edgecolor="black",
    )

    ax1.set_ylabel("Hits@10 (%)", fontsize=11, fontweight="bold")
    ax1.set_title(
        "Độ chính xác Hits@10 theo độ dài ngữ cảnh",
        fontsize=12,
        fontweight="bold",
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(groups, fontsize=10)
    ax1.legend(loc="upper left")
    ax1.grid(axis="y", linestyle="--", alpha=0.6)

    # Ghi nhãn số liệu trên đầu cột
    for rect in rects1:
        h = rect.get_height()
        ax1.annotate(
            f"{h:.1f}%",
            xy=(rect.get_x() + rect.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    for rect in rects2:
        h = rect.get_height()
        ax1.annotate(
            f"{h:.1f}%",
            xy=(rect.get_x() + rect.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Biểu đồ 2: MRR@10
    t_mrr = stats_t["MRR@10"].values
    a_mrr = stats_a["MRR@10"].values

    rects3 = ax2.bar(
        x - width / 2,
        t_mrr,
        width,
        label="TransE",
        color="#2b5c8f",
        edgecolor="black",
    )
    rects4 = ax2.bar(
        x + width / 2,
        a_mrr,
        width,
        label="AnyBURL",
        color="#d95f02",
        edgecolor="black",
    )

    ax2.set_ylabel("MRR@10", fontsize=11, fontweight="bold")
    ax2.set_title(
        "Thứ hạng nghịch đảo MRR@10 theo độ dài ngữ cảnh",
        fontsize=12,
        fontweight="bold",
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels(groups, fontsize=10)
    ax2.legend(loc="upper left")
    ax2.grid(axis="y", linestyle="--", alpha=0.6)

    for rect in rects3:
        h = rect.get_height()
        ax2.annotate(
            f"{h:.3f}",
            xy=(rect.get_x() + rect.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    for rect in rects4:
        h = rect.get_height()
        ax2.annotate(
            f"{h:.3f}",
            xy=(rect.get_x() + rect.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(FIG_FILE, dpi=300)
    print(f"[+] Đã xuất biểu đồ chất lượng cao: {FIG_FILE}")


if __name__ == "__main__":
    run_scenario1()