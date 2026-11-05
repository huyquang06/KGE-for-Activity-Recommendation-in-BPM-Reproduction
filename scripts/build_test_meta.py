import os
import csv

    # Đường dẫn thư mục gốc kge (điều chỉnh nếu bạn để script ở nơi khác)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TEST_FILE = os.path.join(BASE_DIR, "data", "bpmai_lastrev_caise_onlyAfter", "test.txt")
CONTEXT_FILE = os.path.join(
    BASE_DIR, "local", "KGE-ActivityRecommendation", "evaluation_test_context", "test_context.txt"
)
OUTPUT_CSV = os.path.join(BASE_DIR, "results", "test_meta.csv")

def extract_meta():
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    records = []
    with open(TEST_FILE, "r", encoding="utf-8") as tf, open(CONTEXT_FILE, "r", encoding="utf-8") as cf:
        test_lines = [line.strip() for line in tf if line.strip()]
        context_lines = [line.strip() for line in cf if line.strip()]

        assert len(test_lines) == len(context_lines), f"Lệch số dòng: test={len(test_lines)}, context={len(context_lines)}"

        for query_id, (t_line, c_line) in enumerate(zip(test_lines, context_lines)):
            parts = t_line.split("\t")
            if len(parts) != 3:
                continue
            node, relation, true_label = parts[0], parts[1], parts[2]

                # Tách process_id từ node (ví dụ: days_2044556727_rev13 -> 2044556727_rev13)
            process_id = "_".join(node.split("_")[1:])

                # Đếm số nhãn hoạt động trong context (bỏ qua phần tử đầu tiên vì là target)
            c_tokens = c_line.split("\t")
                # Các atom nhãn có dạng: label(node) -> không chứa dấu phẩy
            context_atoms = [tok for tok in c_tokens[1:] if len(tok.split(",")) == 1 and "(" in tok]
            context_length = len(context_atoms)

            records.append({
                "query_id": query_id,
                "process_id": process_id,
                "true_label": true_label,
                "context_length": context_length
            })

        # Ghi ra file CSV
        with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as out_f:
            writer = csv.DictWriter(out_f, fieldnames=["query_id", "process_id", "true_label", "context_length"])
            writer.writeheader()
            writer.writerows(records)

        print(f"Đã tạo thành công {len(records)} bản ghi tại: {OUTPUT_CSV}")

if __name__ == "__main__":
        extract_meta()