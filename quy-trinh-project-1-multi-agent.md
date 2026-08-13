# Quy trình chuẩn để làm Project 1 — Multi-agent LLM System

**Hạn: 21/08/2026** (roadmap ghi: ôn nền LangGraph xong 15/08 → làm project 15/08–21/08, ~7 ngày, full-time 7-8h/ngày)

Đây là quy trình theo chuẩn hiện đại (2026), tức là có kèm cả phần "làm việc với AI CLI" (Claude Code hay công cụ tương tự) chứ không chỉ quy trình code truyền thống. Ý tưởng chung: **định hình rõ trước khi code → dựng khung chạy được tối thiểu → deploy sớm → thêm độ phức tạp dần → test/CI song song → polish cuối**.

---

## Bước 0 — Setup & Planning (nửa ngày đầu, 15/08)

1. **Viết spec 1 trang** trước khi code — trả lời rõ: bài toán gì, input/output, phạm vi (multi-agent làm gì, RAG assistant trả lời loại câu hỏi nào), tiêu chí "xong" là gì. Việc này giúp không bị lan man khi AI CLI gợi ý thêm tính năng.
2. **Cấu trúc repo chuẩn**:
   ```
   project/
     src/            # code chính (agents/, rag/, api/)
     tests/           # pytest
     docs/            # architecture.md, notes
     .github/workflows/  # CI
     Dockerfile
     AGENTS.md         # xem Bước 1
     README.md
     .env.example
     pyproject.toml
   ```
3. **Môi trường**: dùng `uv` hoặc `poetry`, khóa version. `.env` cho API key, **không commit** (`.gitignore`).
4. **Git**: commit theo Conventional Commits (`feat:`, `fix:`, `chore:`...), commit nhỏ sau mỗi bước chạy được — không gộp cả ngày vào 1 commit.

---

## Bước 1 — Tạo file `AGENTS.md` (quy tắc cho AI CLI)

Đây đúng là phần bạn hỏi. `AGENTS.md` là chuẩn chung hiện nay cho các AI coding agent (Claude Code, Cursor, Copilot...) — thay vì mỗi tool một file riêng (`.cursorrules`, `CLAUDE.md`...), giờ có 1 file chung đặt ở gốc repo mà hầu hết CLI đọc được. Cuối 2025 chuẩn này được đưa vào Linux Foundation nên khá chắc sẽ còn dùng lâu.

**Lưu ý quan trọng**: nghiên cứu cho thấy file do AI tự sinh ra thực ra làm agent làm việc *tệ hơn* (giảm ~1-2% tỉ lệ thành công, tốn thêm ~20% token), còn file do người tự viết tay giúp tăng ~4%. Nên **tự viết**, đừng bảo AI viết hộ AGENTS.md.

Nội dung nên có (giữ dưới 150-200 dòng, đừng lặp lại thứ đã có trong README):

- **Tech stack + version cụ thể** — ví dụ Python 3.12, LangGraph 0.x, FastAPI, embedding model dùng, vector store dùng.
- **Lệnh hay dùng** — `pytest`, `ruff check .`, lệnh chạy dev server, lệnh build docker — ghi rõ để AI không đoán sai.
- **Code style** — dán 1 đoạn code mẫu thay vì mô tả chung chung.
- **Quy tắc test** — chạy test bằng lệnh gì, có mock LLM call không.
- **3 mức quyền rõ ràng**:
  - ✅ Luôn được tự làm: sửa code, chạy test, thêm log
  - ⚠️ Phải hỏi trước: đổi schema state, đổi cấu trúc thư mục, thêm dependency mới
  - 🚫 Không bao giờ: commit `.env`/secret, force push, xoá dữ liệu
- **Pattern "phi trực giác"** — chỗ nào code trông "sai" nhưng thực ra cố ý (ví dụ dùng `Annotated[list, add]` cho state để tránh bị ghi đè) thì ghi rõ lý do, không AI sẽ "sửa" nhầm.

---

## Bước 2 — Thiết kế trước khi code (nửa ngày, vẫn 15/08)

Đừng mở editor code ngay. Viết ra `docs/architecture.md` ngắn:

- **Graph LangGraph**: liệt kê node (supervisor, retriever, responder...), state schema (dùng `TypedDict` + `Annotated[..., operator.add]` cho các field bị nhiều node ghi vào, tránh bug ghi đè state — đây là lỗi hay gặp nhất khi mới làm multi-agent).
- **RAG pipeline**: nguồn dữ liệu → chunking strategy → embedding model → vector store → có rerank không.
- **Phân vai agent**: agent nào là supervisor/router, agent nào chỉ được đọc (query), **tách riêng agent/tool có quyền ghi/xoá** — không gộp chung với tool chỉ đọc, để sau này dễ thêm bước duyệt (human-in-the-loop) cho hành động nguy hiểm.

---

## Bước 3 — Build theo "walking skeleton", không build từ trong ra ngoài

Thứ tự khuyến nghị (khác với thói quen code hết logic rồi mới deploy — cách đó rủi ro deploy fail sát hạn):

1. **Ngày 2 (16/08)** — Dựng khung tối thiểu chạy được: 1 node LangGraph trả lời cứng → bọc FastAPI → **deploy lên Render/HF Spaces ngay**, có link sống. Mục tiêu: chứng minh pipeline deploy hoạt động trước, còn thông minh hay không tính sau.
2. **Ngày 3-4 (17-18/08)** — Thêm RAG thật: nạp dữ liệu, embedding, retrieval, nối vào 1 agent, test bằng vài câu hỏi thật.
3. **Ngày 5 (19/08)** — Thêm multi-agent thật (supervisor điều phối nhiều agent con), thêm guardrail cơ bản: giới hạn số vòng lặp tối đa (tránh agent lặp vô hạn), giới hạn token/cost.
4. **Ngày 6 (20/08)** — Test + CI + Docker (chi tiết ở Bước 4-5).
5. **Ngày 7 (21/08)** — Deploy bản cuối, hoàn thiện README, quay GIF demo, polish.

---

## Bước 4 — Độ tin cậy (áp dụng khi build multi-agent, không phải optional)

Đây là khác biệt giữa "demo cho vui" và "trông như production", ban giám khảo/nhà tuyển dụng nhìn vào sẽ thấy ngay:

- **Lỗi tạm thời** (API rate limit, timeout): retry với exponential backoff.
- **Lỗi do LLM** (tool gọi sai tham số): validate input, trả lỗi rõ ràng để agent tự sửa lại (self-correct), không crash cả pipeline.
- **Lỗi logic** (agent lặp vô hạn): đặt hard cap số vòng lặp + phát hiện trạng thái "kẹt" (agent trả lời giống hệt vòng trước).
- **Checkpointing**: dùng SQLite/Postgres checkpointer của LangGraph để lưu state — phần này bạn đã học trong "AI Agents in LangGraph" (persistence), áp dụng lại luôn.
- **Đừng để agent hoàn toàn tự trị** ở bước quan trọng — thêm 1 điểm dừng xác nhận (dù chỉ là log ra để bạn tự kiểm tra) thể hiện tư duy production.

---

## Bước 5 — Test & CI (song song lúc build, không dồn cuối)

- `pytest` cho từng tool/function riêng lẻ — mock LLM call để test chạy nhanh và ổn định (deterministic).
- 1 test tích hợp gọi thẳng endpoint FastAPI với câu hỏi cố định, kiểm tra response đúng schema.
- GitHub Actions: 1 workflow chạy `ruff check` + `pytest` mỗi lần push — vài chục dòng YAML, nhưng có cái này trong CV/README ghi điểm rõ với recruiter kỹ thuật.

---

## Bước 6 — Docker & Deploy

- Viết `Dockerfile` (và `docker-compose.yml` nếu vector store chạy như service riêng).
- Secret (API key) để trong biến môi trường của platform deploy (Render/HF Spaces), **không bao giờ** trong code hay Dockerfile.
- Sau khi deploy, **tự tay test lại toàn bộ flow qua link sống** — nhiều người chỉ test local rồi quên verify bản deploy thật.

---

## Bước 7 — Tài liệu & polish cho hồ sơ

README nên có: mô tả bài toán 2-3 câu, sơ đồ kiến trúc (ảnh đơn giản cũng được), link demo sống, GIF/screenshot, tech stack, hướng dẫn chạy local, và mục "hạn chế / nếu có thêm thời gian sẽ làm gì" — mục cuối này thể hiện tư duy senior (đúng phần "Giao tiếp kỹ thuật" trong roadmap của bạn), rất hay được hỏi lúc phỏng vấn.

---

## Bước 8 — Cách làm việc với AI CLI hằng ngày (Claude Code hoặc tương tự)

- **Lập kế hoạch trước khi để AI sửa nhiều file** — dùng plan mode / yêu cầu AI trình bày kế hoạch trước, đừng để nó tự chạy một mạch cho thay đổi lớn.
- **Commit nhỏ, review từng bước** — sau mỗi bước chạy được thì commit, không để AI viết một lượt cả nghìn dòng rồi mới review.
- **Luôn chạy test trước khi chấp nhận code AI viết** — đừng tin bằng mắt.
- **Kiểm tra diff trước khi push**, đặc biệt xem có secret/API key nào lọt vào không.
- **AGENTS.md là nguồn sự thật dùng chung** — mỗi khi có quyết định kiến trúc mới (ví dụ đổi vector store), cập nhật lại AGENTS.md để các phiên làm việc sau (kể cả bạn tự đọc lại) không bị lệch ngữ cảnh.

---

## Tóm tắt lịch 7 ngày (15/08 → 21/08)

| Ngày | Việc chính |
|---|---|
| 15/08 | Spec + AGENTS.md + repo skeleton + architecture.md |
| 16/08 | Walking skeleton: 1 node → API → deploy link sống |
| 17-18/08 | RAG thật (chunking, embedding, retrieval) + 1 agent hoạt động |
| 19/08 | Multi-agent thật (supervisor) + guardrail (loop cap, error handling) |
| 20/08 | Test + CI (GitHub Actions) + Dockerize |
| 21/08 | Deploy bản cuối + README + demo GIF + polish |

---

### Nguồn tham khảo
- [How to Build Your AGENTS.md (2026) — Augment Code](https://www.augmentcode.com/guides/how-to-build-agents-md)
- [Building Production AI Agents with LangGraph: Beyond the Toy Examples (2026) — DEV Community](https://dev.to/young_gao/building-production-ai-agents-with-langgraph-beyond-the-toy-examples-2idm)
