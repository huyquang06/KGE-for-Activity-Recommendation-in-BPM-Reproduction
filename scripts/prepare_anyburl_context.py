import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONTEXT_FILE = os.path.join(
    BASE_DIR, "local", "KGE-ActivityRecommendation", "evaluation_test_context", "test_context.txt"
    )
TRAIN_FILE = os.path.join(BASE_DIR, "data", "bpmai_lastrev_caise_onlyAfter", "train.txt")
OUTPUT_FILE = os.path.join(BASE_DIR, "data", "bpmai_lastrev_caise_onlyAfter", "train_with_context.txt")

def build_context_graph():
    print(f"[*] Đang đọc test context từ: {CONTEXT_FILE}")
    context_triples = set()
    target_triples = set()

    with open(CONTEXT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            tokens = line.strip().split("\t")
            if not tokens or not tokens[0]:
                continue

                # Target triple (cần đoán) -> TUYỆT ĐỐI KHÔNG đưa vào đồ thị nền để tránh Data Leakage
            target = tokens[0]
            t_lbl = target.split("(")[0]
            t_node = target.split("(")[1].replace(")", "")
            target_triples.add((t_node, "hasLabel", t_lbl))

                # Trích xuất các cạnh ngữ cảnh (after và hasLabel của các bước trước)
            for t in tokens[1:]:
                if not t:
                    continue
                if t.startswith("after("):
                    args = t[6:-1].split(",")
                    context_triples.add((args[0], "after", args[1]))
                elif "(" in t and t.endswith(")") and not t.startswith("inSameProcess("):
                    lbl = t.split("(")[0]
                    node = t.split("(")[1].replace(")", "")
                    if not (node == t_node and lbl == t_lbl):
                        context_triples.add((node, "hasLabel", lbl))

    print(f"[+] Trích xuất được {len(context_triples)} bộ ba ngữ cảnh quy trình.")
    print(f"[+] Kiểm tra rò rỉ dữ liệu (Data Leakage): {len(context_triples.intersection(target_triples))} trùng lặp (Chuẩn: 0).")

    # Đọc train.txt gốc
    with open(TRAIN_FILE, "r", encoding="utf-8") as f:
        train_lines = [l.strip() for l in f if l.strip()]

    print(f"[*] Số bộ ba tập train gốc: {len(train_lines)}")

    # Gộp lại thành train_with_context.txt
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for l in train_lines:
            f.write(l + "\n")
        for h, r, t in context_triples:
            f.write(f"{h}\t{r}\t{t}\n")

    total_triples = len(train_lines) + len(context_triples)
    print(f"[✓] Đã tạo thành công đồ thị nền hoàn chỉnh: {OUTPUT_FILE} (Tổng {total_triples} bộ ba).")

if __name__ == "__main__":
    build_context_graph()