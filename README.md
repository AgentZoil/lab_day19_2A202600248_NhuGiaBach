# LAB DAY 19: GraphRAG với Tech Company Corpus

## Thông tin sinh viên

| Thông tin | Chi tiết |
|-----------|----------|
| Họ và tên | Nhu Gia Bach |
| MSSV | 2A202600248 |
| Môn học | LAB DAY 19 - GraphRAG |
| Công cụ | NetworkX + Python |

## Mục tiêu

- Trích xuất thực thể và quan hệ từ văn bản thô
- Xây dựng đồ thị tri thức bằng NetworkX
- Truy vấn multi-hop trên đồ thị
- Đánh giá GraphRAG so với Flat RAG trên benchmark 20 câu hỏi

## Cấu trúc chính

```text
lab_day19_2A2026248_NhuGiaBach/
├── data/raw/tech_companies.txt
├── data/processed/triples.json
├── src/
│   ├── entity_extraction.py
│   ├── flat_rag.py
│   ├── graph_builder.py
│   └── query_engine.py
├── outputs/
│   ├── graphs/
│   ├── results/
│   └── reports/
├── notebooks/graphrag_demo.ipynb
├── run.py
├── requirements.txt
└── README.md
```

## Yêu cầu hệ thống

- Python 3.9+
- RAM tối thiểu 4GB
- 500MB dung lượng trống

## Cài đặt và chạy

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python run.py
```

Nếu dùng Windows:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python run.py
```

Notebook tùy chọn:

```bash
jupyter notebook notebooks/graphrag_demo.ipynb
```

Chạy notebook dạng test end-to-end:

```bash
jupyter nbconvert --to notebook --execute notebooks/graphrag_demo.ipynb --output graphrag_demo.executed.ipynb
```

## Mô tả nhanh các file

- `data/raw/tech_companies.txt`: dữ liệu đầu vào
- `data/processed/triples.json`: triples đã trích xuất từ raw text
- `src/entity_extraction.py`: trích xuất triples từ corpus
- `src/flat_rag.py`: baseline retrieval cho so sánh với GraphRAG
- `src/graph_builder.py`: xây dựng và trực quan hóa đồ thị
- `src/query_engine.py`: xử lý truy vấn trên đồ thị
- `run.py`: chạy toàn bộ pipeline
- `notebooks/graphrag_demo.ipynb`: demo từng bước
- `outputs/reports/report.md`: report ngắn về cost/time/accuracy

## Benchmark

Project includes 20 benchmark questions spanning direct facts, multi-hop cases, and out-of-corpus abstention tests.

## Kết quả hiện tại

- GraphRAG accuracy: 100%
- Flat RAG accuracy: 75%
- Graph size: 52 nodes, 46 edges
- Extracted triples: 47
- Benchmark includes 18 answerable questions and 2 out-of-corpus questions
- Corpus hiện tại được chia nhỏ thành 16 paragraph để benchmark khó hơn và bộc lộ khác biệt giữa retrieval baseline và graph reasoning.

## Kết quả benchmark

- `outputs/results/benchmark_results.json`
- `outputs/results/comparison_table.md`

## Kết quả đầu ra

- Ảnh đồ thị: `outputs/graphs/knowledge_graph.png`
- Interactive graph: `outputs/graphs/knowledge_graph_interactive.html`
- Kết quả benchmark: `outputs/results/benchmark_results.json`
- Bảng so sánh: `outputs/results/comparison_table.md`

## Ghi chú

- Dự án ưu tiên OpenAI extraction khi có API key và kết nối, nhưng có rule-based fallback để pipeline vẫn chạy được khi offline
- Nếu muốn tránh gọi API khi test, đặt `USE_OPENAI_EXTRACTOR=0`
- `outputs/` là dữ liệu sinh ra khi chạy, có thể không cần commit

## Tài liệu tham khảo

- NetworkX Documentation
- Matplotlib Documentation
- GraphRAG Paper

## Cách test nhanh

```bash
source venv/bin/activate
python run.py
jupyter nbconvert --to notebook --execute notebooks/graphrag_demo.ipynb --output graphrag_demo.executed.ipynb
```

## Liên hệ

- Nhu Gia Bach
- 2A202600248
