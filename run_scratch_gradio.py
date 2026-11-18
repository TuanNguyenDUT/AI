import os
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "True"
import json
from pathlib import Path
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
import gradio as gr

# =====================================================================
# 1. KIẾN TRÚC MODEL TỰ HUẤN LUYỆN TỪ ĐẦU (110K VIDEO)
# =====================================================================
T, CROP, MAXL = 16, 112, 48
NTOK = (T // 2) * (CROP // 16) * (CROP // 16)
MEAN = torch.tensor([0.45, 0.45, 0.45]).view(3, 1, 1, 1)
STD = torch.tensor([0.225, 0.225, 0.225]).view(3, 1, 1, 1)

class Patch3D(nn.Module):
    def __init__(self, D):
        super().__init__()
        self.proj = nn.Conv3d(3, D, (2, 16, 16), (2, 16, 16))
    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)

def blocks(D, heads, depth):
    layer = nn.TransformerEncoderLayer(D, heads, 4 * D, 0.0, activation='gelu', batch_first=True, norm_first=True)
    return nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)

class Encoder(nn.Module):
    def __init__(self, D=384, heads=6, depth=8):
        super().__init__()
        self.D = D
        self.patch = Patch3D(D)
        self.pos = nn.Parameter(torch.zeros(1, NTOK, D))
        self.blk = blocks(D, heads, depth)
        self.norm = nn.LayerNorm(D)
    def forward(self, x):
        return self.norm(self.blk(self.patch(x) + self.pos))

class Captioner(nn.Module):
    def __init__(self, V, D=384, heads=6, depth=4):
        super().__init__()
        self.enc = Encoder(D)
        self.emb = nn.Embedding(V, D)
        self.pos = nn.Parameter(torch.zeros(1, MAXL, D))
        layer = nn.TransformerDecoderLayer(D, heads, 4 * D, 0.1, activation='gelu', batch_first=True, norm_first=True)
        self.dec = nn.TransformerDecoder(layer, depth)
        self.norm = nn.LayerNorm(D)
        self.out = nn.Linear(D, V)

    def decode(self, mem, ids):
        L = ids.shape[1]
        causal = torch.triu(torch.ones(L, L, device=ids.device, dtype=torch.bool), 1)
        h = self.emb(ids) + self.pos[:, :L]
        return self.out(self.norm(self.dec(h, mem, tgt_mask=causal)))

    @torch.no_grad()
    def generate_clean(self, x, itos, maxlen=20, rep_penalty=1.35):
        mem = self.enc(x)
        B = x.shape[0]
        ids = torch.ones(B, 1, dtype=torch.long, device=x.device)
        confs = []
        for _ in range(maxlen - 1):
            logits = self.decode(mem, ids)[:, -1, :].float()
            logits[:, 0] = -1e9
            logits[:, 3] = -1e9
            for b in range(B):
                for t in set(ids[b, 1:].tolist()):
                    logits[b, t] = logits[b, t] / rep_penalty if logits[b, t] > 0 else logits[b, t] * rep_penalty
                if ids.shape[1] >= 2:
                    prev_tok = ids[b, -1].item()
                    seq = ids[b, 1:].tolist()
                    for idx in range(len(seq) - 1):
                        if seq[idx] == prev_tok:
                            logits[b, seq[idx + 1]] = -1e9
            probs = torch.softmax(logits, dim=-1)
            nxt_prob, nxt = probs.max(dim=-1, keepdim=True)
            confs.append(nxt_prob.item())
            ids = torch.cat([ids, nxt], dim=1)
            if (ids == 2).any(1).all():
                break
        words = []
        for t in ids[0, 1:].tolist():
            if t == 2:
                break
            words.append(itos[t])
        avg_conf = round(float(np.mean(confs)) if confs else 0.5, 4)
        return " ".join(words), avg_conf

# =====================================================================
# 2. NẠP TRỌNG SỐ MÔ HÌNH (ƯU TIÊN BẢN CAP_110K_BEST.PT)
# =====================================================================
BASE_DIR = Path(__file__).resolve().parent
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_scratch_model():
    candidates = [
        (BASE_DIR / "member3-handoff" / "cap_110k_best.pt", BASE_DIR / "member3-handoff" / "vocab_40k.json"),
        (BASE_DIR / "cap_110k_best.pt", BASE_DIR / "vocab_40k.json"),
        (BASE_DIR / "member3-handoff" / "cap_80k_best.pt", BASE_DIR / "member3-handoff" / "vocab_40k.json"),
        (BASE_DIR / "cap_80k_best.pt", BASE_DIR / "vocab_40k.json"),
        (BASE_DIR / "cap_40k_best.pt", BASE_DIR / "vocab_40k.json"),
        (BASE_DIR / "cap_best.pt", BASE_DIR / "vocab.json"),
        (BASE_DIR / "ckpt" / "cap_110k_best.pt", BASE_DIR / "ckpt" / "vocab_40k.json"),
        (BASE_DIR / "ckpt" / "cap_best.pt", BASE_DIR / "ckpt" / "vocab.json"),
    ]
    for ckpt_p, vocab_p in candidates:
        if ckpt_p.exists() and vocab_p.exists():
            itos = json.load(open(vocab_p, "r", encoding="utf-8"))
            model = Captioner(len(itos)).to(DEVICE)
            state = torch.load(ckpt_p, map_location=DEVICE, weights_only=False)
            model.load_state_dict(state["m"] if "m" in state else state, strict=True)
            model.eval()
            return model, itos, f"Đã nạp model: {ckpt_p.name} ({len(itos)} từ vựng)"
    return None, None, "Chưa tìm thấy cap_110k_best.pt và vocab_40k.json trong thư mục!"

SCRATCH_MODEL, ITOS, MODEL_STATUS = load_scratch_model()

# =====================================================================
# 3. HÀM XỬ LÝ VIDEO & HỖ TRỢ PHỤ ĐỀ NHIỀU NGƯỜI (MULTI-PERSON DENSE)
# =====================================================================
# Bộ phân loại người dựa trên OpenCV HOG (với cơ chế dự phòng an toàn)
if hasattr(cv2, 'HOGDescriptor'):
    try:
        HOG_DETECTOR = cv2.HOGDescriptor()
        HOG_DETECTOR.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    except Exception:
        HOG_DETECTOR = None
else:
    HOG_DETECTOR = None

def detect_person_boxes(mid_frame):
    if HOG_DETECTOR is not None:
        try:
            boxes, _ = HOG_DETECTOR.detectMultiScale(mid_frame, winStride=(8, 8), padding=(8, 8), scale=1.05)
            if len(boxes) > 0:
                return sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)[:4]
        except Exception:
            pass
    # Dự phòng thông minh: Phân vùng không gian (Trái / Phải / Trung tâm)
    H, W, _ = mid_frame.shape
    return [
        [0, 0, int(W * 0.5), H],          # Nửa bên trái
        [int(W * 0.5), 0, int(W * 0.5), H] # Nửa bên phải
    ]

def extract_tensor_from_frames(frames, crop_box=None):
    idx = np.linspace(0, len(frames) - 1, T).astype(int)
    sampled = []
    for i in idx:
        f = frames[i]
        if crop_box is not None:
            x, y, w, h = crop_box
            H, W, _ = f.shape
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(W, x + w), min(H, y + h)
            cropped = f[y1:y2, x1:x2]
            if cropped.size > 0:
                f = cropped
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (CROP, CROP), interpolation=cv2.INTER_AREA)
        sampled.append(resized)
    arr = np.stack(sampled, axis=0)
    x_t = torch.from_numpy(arr).permute(3, 0, 1, 2).float() / 255.0
    return ((x_t - MEAN) / STD).unsqueeze(0).to(DEVICE)

def run_causal_caption_step(frames, mode="single"):
    """
    Sinh phụ đề Causal cho một chunk dữ liệu (chỉ dùng quá khứ và hiện tại).
    Tập trung nhận diện hành động của một người chính trong khung hình.
    """
    last_frame = frames[-1].copy()
    H, W, _ = last_frame.shape
    mid_frame = frames[len(frames) // 2]
    boxes = detect_person_boxes(mid_frame)

    if mode == "single":
        # CHẾ ĐỘ ƯU TIÊN: Tập trung nhận diện 1 người duy nhất đang làm gì
        # Chọn người có diện tích lớn nhất / rõ ràng nhất trong khung hình
        target_box = None
        if len(boxes) > 0:
            target_box = max(boxes, key=lambda b: b[2] * b[3])
            x, y, w, h = target_box
            # Mở rộng nhẹ viền 10% để bắt trọn động tác tay/chân
            pad_w = int(w * 0.1)
            pad_h = int(h * 0.1)
            x1, y1 = max(0, x - pad_w), max(0, y - pad_h)
            x2, y2 = min(W, x + w + pad_w), min(H, y + h + pad_h)
            crop_box = [x1, y1, x2 - x1, y2 - y1]

            # Vẽ khung theo dõi người mục tiêu
            cv2.rectangle(last_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(last_frame, "Target Person", (x1 + 5, max(25, y1 - 8)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            person_tensor = extract_tensor_from_frames(frames, crop_box=crop_box)
        else:
            # Nếu không tìm thấy box rõ rệt, lấy toàn cảnh trung tâm
            person_tensor = extract_tensor_from_frames(frames)

        caption, score = SCRATCH_MODEL.generate_clean(person_tensor, ITOS)
        
        output_text = (
            f"👤 [Hành động]: {caption}\n"
            f"🎯 [Độ tự tin]: {score * 100:.1f}%"
        )
        return output_text, last_frame

    # CHẾ ĐỘ PHỤ: Nhận diện nhiều người nếu người dùng bật
    global_x = extract_tensor_from_frames(frames)
    scene_caption, scene_score = SCRATCH_MODEL.generate_clean(global_x, ITOS)
    output_lines = [f"🌐 [Toàn cảnh]: {scene_caption} (Conf: {scene_score:.2f})"]
    colors = [(0, 255, 0), (255, 120, 0), (0, 180, 255), (200, 0, 255)]

    if len(boxes) >= 2:
        output_lines.append(f"👥 Phát hiện {len(boxes)} chủ thể đang hoạt động đồng thời:")
        for idx, box in enumerate(boxes):
            x, y, w, h = box
            color = colors[idx % len(colors)]
            cv2.rectangle(last_frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(last_frame, f"P{idx + 1}", (x + 5, max(25, y - 5)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

            person_x = extract_tensor_from_frames(frames, crop_box=box)
            p_cap, p_score = SCRATCH_MODEL.generate_clean(person_x, ITOS)
            output_lines.append(f"   👤 [Người {idx + 1}]: {p_cap} (Conf: {p_score:.2f})")
    else:
        output_lines.append(f"   👤 [Chủ thể chính]: {scene_caption}")

    return "\n".join(output_lines), last_frame

# =====================================================================
# 4. HÀM XỬ LÝ CAUSAL STREAMING THỜI GIAN THỰC (FILE HOẶC WEBCAM)
# =====================================================================
def sync_causal_stream(source_type, video_file, mode_choice, chunk_sec=4.0):
    global SCRATCH_MODEL, ITOS, MODEL_STATUS
    if SCRATCH_MODEL is None:
        SCRATCH_MODEL, ITOS, MODEL_STATUS = load_scratch_model()
        if SCRATCH_MODEL is None:
            yield None, f"❌ Lỗi: {MODEL_STATUS}"
            return

    # Xác định nguồn phát stream
    if source_type == "webcam":
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            yield None, "⚠️ Không thể kết nối với Webcam trên máy. Vui lòng chuyển sang chọn File Video (.mp4)."
            return
        is_live = True
        fps = 20.0
    else:
        if not video_file:
            yield None, "⚠️ Vui lòng tải lên một file video .mp4."
            return
        video_path = video_file if isinstance(video_file, str) else getattr(video_file, "name", str(video_file))
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            yield None, f"⚠️ Không thể đọc file video: {video_path}"
            return
        is_live = False
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        if fps <= 0 or fps > 60:
            fps = 25.0

    frames_per_chunk = max(6, int(fps * chunk_sec))
    rolling_buffer = []
    transcript_history = []
    chunk_idx = 0
    frame_count = 0

    yield None, f"🚀 Đang khởi động luồng Causal Stream ({'Webcam Trực Tiếp' if is_live else 'File Video'})...\n* Chế độ: {mode_choice.upper()} | Chunk: {chunk_sec}s | FPS: {fps:.1f}"

    while True:
        ret, frame = cap.read()
        if not ret:
            # Xử lý phần dữ liệu còn lại ở cuối luồng nếu có
            if len(rolling_buffer) >= 6:
                chunk_start = chunk_idx * chunk_sec
                chunk_end = chunk_start + (len(rolling_buffer) / fps)
                cap_text, annotated_frame = run_causal_caption_step(rolling_buffer, mode=mode_choice)
                time_stamp = f"⏱️ [{chunk_start:.1f}s ➔ {chunk_end:.1f}s]"
                transcript_history.append(f"{time_stamp}\n{cap_text}")
                rgb_annotated = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                yield rgb_annotated, "\n\n" + ("\n\n".join(transcript_history[-6:]))
            break

        frame_count += 1
        rolling_buffer.append(frame)

        # Cập nhật khung hình xem trước mượt mà
        if frame_count % 8 == 0 and len(rolling_buffer) < frames_per_chunk:
            rgb_preview = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            yield rgb_preview, "\n\n" + ("\n\n".join(transcript_history[-6:])) if transcript_history else "⏳ Đang tích lũy khung hình cho chunk đầu tiên..."

        # Khi gom đủ 1 chunk nhân quả (Causal Window)
        if len(rolling_buffer) >= frames_per_chunk:
            chunk_start = chunk_idx * chunk_sec
            chunk_end = chunk_start + chunk_sec
            time_stamp = f"⏱️ Mốc [{chunk_start:.1f}s ➔ {chunk_end:.1f}s]"

            # Xử lý mô hình Causal (chỉ dựa vào quá khứ + hiện tại)
            cap_text, annotated_frame = run_causal_caption_step(rolling_buffer, mode=mode_choice)
            transcript_history.append(f"{time_stamp}\n{cap_text}")

            rgb_annotated = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            yield rgb_annotated, "\n\n" + ("\n\n".join(transcript_history[-6:]))

            # Giữ lại 25% số frames gần nhất làm Causal Memory Overlap cho chunk kế tiếp
            keep_overlap = int(frames_per_chunk * 0.25)
            rolling_buffer = rolling_buffer[-keep_overlap:] if keep_overlap > 0 else []
            chunk_idx += 1

            if not is_live:
                # Điều tiết độ trễ phát video giống thời gian thực ngoài đời
                time.sleep(0.08)

    cap.release()
    final_output = f"✅ Hoàn tất phát luồng Causal Stream!\n\n" + ("\n\n".join(transcript_history))
    yield None, final_output

# =====================================================================
# 5. GIAO DIỆN WEB DEMO CAUSAL STREAMING
# =====================================================================
def build_demo():
    with gr.Blocks(title="Causal Streaming Video Captioning - 110k Model", analytics_enabled=False) as demo:
        gr.Markdown(
            f"""
            # 🎬 Hệ Thống Causal Stream: Nhận Diện Hành Động Người Theo Thời Gian Thực
            * **Mô hình hiện tại:** `{MODEL_STATUS}` | **Thiết bị chạy:** `{DEVICE.upper()}`
            * **Đặc tính Causal Stream:** Mô hình **hoàn toàn không nhìn trước tương lai**, xử lý dữ liệu liên tục theo dòng chảy (Webcam hoặc File Video).
            * **Mục tiêu:** Tự động bắt vị trí người và **trả lời chính xác người đó đang làm gì** theo từng chunk thời gian thực.
            """
        )
        with gr.Row():
            with gr.Column(scale=1):
                source_type = gr.Radio(
                    choices=[("📁 Tải file Video (.mp4)", "file"), ("📹 Trực tiếp từ Webcam thiết bị", "webcam")],
                    value="file",
                    label="1. Nguồn Dữ Liệu Stream"
                )
                video_input = gr.File(label="Tải file video (.mp4)", file_types=[".mp4"])
                mode_choice = gr.Radio(
                    choices=[
                        ("👤 Nhận diện 1 người đang làm gì (Single-Person Action)", "single"), 
                        ("👥 Nhận diện nhiều người cùng lúc (Multi-Person)", "multi")
                    ],
                    value="single",
                    label="2. Chế độ phân tích"
                )
                chunk_slider = gr.Slider(minimum=2.0, maximum=6.0, value=4.0, step=0.5, label="3. Độ dài Causal Chunk (giây)")
                with gr.Row():
                    submit_btn = gr.Button("▶️ BẮT ĐẦU CAUSAL STREAM", variant="primary")
                    stop_btn = gr.Button("⏹️ DỪNG LẠI", variant="stop")
            with gr.Column(scale=1):
                live_view = gr.Image(label="Khung hình Stream Trực Tiếp (Live Causal View)", interactive=False)
                transcript_box = gr.Textbox(label="Dòng Phụ Đề Trực Tiếp (Live Causal Transcript)", lines=12, placeholder="Chờ kích hoạt stream...")

        watcher = submit_btn.click(
            fn=sync_causal_stream, 
            inputs=[source_type, video_input, mode_choice, chunk_slider], 
            outputs=[live_view, transcript_box]
        )
        stop_btn.click(
            fn=lambda: (None, "⏹️ Luồng Causal Stream đã được dừng lại."), 
            inputs=None, 
            outputs=[live_view, transcript_box], 
            cancels=[watcher]
        )
    return demo

if __name__ == "__main__":
    demo = build_demo()
    demo.launch(server_name="127.0.0.1", server_port=7861, share=False)
