import os
import pandas as pd

    # Đường dẫn (chạy được cả trên ổ K: hoặc đường dẫn gốc)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PRED_FILE = os.path.join(BASE_DIR, "results", "anyburl_A", "predictions-100")
META_FILE = os.path.join(BASE_DIR, "results", "test_meta.csv")
CONTEXT_FILE = os.path.join(
    BASE_DIR, "local", "KGE-ActivityRecommendation", "evaluation_test_context", "test_context.txt"
)
OUTPUT_DIR = os.path.join(BASE_DIR, "results", "anyburl")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "ranks.csv")

def parse_contexts():
    """Đọc tập nhãn đã có trong từng process để lọc nhãn trùng"""
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

def convert_anyburl():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df_meta = pd.read_csv(META_FILE)
    contexts = parse_contexts()

        # Đọc file raw predictions của AnyBURL
    with open(PRED_FILE, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

        # Trong AnyBURL apply output: 1 dòng triples, 1 dòng "Tails: ...", 1 dòng "Heads: ..."
    tails_dict = {}
    current_node = None
    for line in lines:
        if "hasLabel" in line and not line.startswith("Tails:") and not line.startswith("Heads:"):
            parts = line.split()
            current_node = parts[0]
        elif line.startswith("Tails:"):
            preds = line.replace("Tails:", "").strip().split("\t")
                # AnyBURL ghi xen kẽ: nhãn \t điểm \t nhãn \t điểm
            labels_only = preds[0::2]
            tails_dict[current_node] = labels_only

    ranks = []
    top10_list = []

    for _, row in df_meta.iterrows():
        proc = str(row["process_id"])
        true_lbl = str(row["true_label"])

            # Tìm danh sách dự đoán ứng với test query
            # Tìm theo node chứa process_id
        matching_preds = []
        for node, preds in tails_dict.items():
            if node.endswith(proc):
                matching_preds = preds
                break

            # Áp dụng bộ lọc hậu xử lý (Filter của tác giả)
        proc_known_labels = contexts.get(proc, set())
        filtered = []
        for p in matching_preds:
            if "_" in p:  # Lọc entity không phải nhãn hoạt động (node/process)
                continue
            if p in proc_known_labels:  # Lọc nhãn đã xuất hiện trong cùng process
                continue
            filtered.append(p)

            # Tính rank của nhãn đúng
        if true_lbl in filtered:
            r = filtered.index(true_lbl) + 1
        else:
            r = 999999  # Không nằm trong danh sách dự đoán

        ranks.append(r)
        top10_list.append("|".join(filtered[:10]))

    df_meta["rank"] = ranks
    df_meta["top10_predictions"] = top10_list

        # Lưu ra results/anyburl/ranks.csv
    df_meta.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"-> Đã ghi file kết quả chuẩn: {OUTPUT_CSV}")

        # Đánh giá chỉ số
    hits10 = (df_meta["rank"] <= 10).mean()
        # Tính MRR@10 theo chuẩn bài báo
    mrr_10 = df_meta["rank"].apply(lambda r: 1.0 / r if r <= 10 else 0.0).mean()
        # Tính Full MRR
    mrr_full = df_meta["rank"].apply(lambda r: 1.0 / r if r < 999999 else 0.0).mean()

    print("\n================ KẾT QUẢ BASELINE ANYBURL CỦA BẠN (A) ================")
    print(f"Tổng số queries: {len(df_meta)}")
    print(f"Hits@10        : {hits10:.4f} ({hits10*100:.2f}%)")
    print(f"MRR@10 (bài báo): {mrr_10:.4f}")
    print(f"Full MRR       : {mrr_full:.4f}")
    print("=======================================================================")

if __name__ == "__main__":
    convert_anyburl()