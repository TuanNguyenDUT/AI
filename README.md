# Causal Streaming Video Captioning (From Scratch - 110k Videos)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1se5iJd79eCw4sxF-LWyen5g50H0Nucuo?usp=sharing)

Hệ thống sinh phụ đề và nhận diện hành động dạng dòng chảy nhân quả thời gian thực (**Causal Streaming Video Captioning**) hỗ trợ người khiếm thính, được huấn luyện hoàn toàn từ đầu (**From Scratch**) trên hơn 100.000 video.

* **Notebook huấn luyện trực tiếp trên Colab:** [Mở Google Colab](https://colab.research.google.com/drive/1se5iJd79eCw4sxF-LWyen5g50H0Nucuo?usp=sharing)

---

## 📁 Cấu trúc Repository

* `run_scratch_gradio.py`: Ứng dụng giao diện Web Gradio chạy suy luận thời gian thực (hỗ trợ cả **Webcam trực tiếp** và **File Video**).
* `vocab_40k.json`: Bộ từ vựng chuẩn hóa gồm 8.004 từ.
* `requirements.txt`: Danh sách các thư viện cần cài đặt.

---

## 🚀 Hướng Dẫn Cài Đặt & Chạy Thử Nghiệm

### 1. Cài đặt môi trường
Khuyến nghị sử dụng Python 3.10 - 3.12:

```bash
git clone https://github.com/TuanNguyenDUT/AI.git
cd AI
pip install -r requirements.txt
```

### 2. Tải file trọng số mô hình (`cap_110k_best.pt`)
Do GitHub giới hạn kích thước file tải lên tối đa là 100MB, file trọng số mô hình `cap_110k_best.pt` (122MB) được lưu trữ trên Google Drive:

* **Link tải model checkpoint:** *[Dán link Google Drive chia sẻ file cap_110k_best.pt vào đây]*
* Sau khi tải về, hãy đặt file `cap_110k_best.pt` vào cùng thư mục với file `run_scratch_gradio.py`.

### 3. Khởi chạy ứng dụng Web Demo
Chạy lệnh sau trong terminal:

```bash
python run_scratch_gradio.py
```

Truy cập giao diện Web tại địa chỉ: **`http://127.0.0.1:7861`**

* **Chế độ Webcam:** Chọn *Trực tiếp từ Webcam thiết bị* $\to$ Bấm **Bắt đầu Causal Stream** để nhận diện hành động trực tiếp từ camera.
* **Chế độ File Video:** Tải file `.mp4` lên $\to$ Bấm **Bắt đầu Causal Stream** để xem mô hình phân tích theo từng chunk thời gian thực.
