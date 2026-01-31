<!-- Banner -->
<p align="center">
  <a href="https://www.uit.edu.vn/" title="Trường Đại học Công nghệ Thông tin" style="border: none;">
    <img src="https://i.imgur.com/WmMnSRt.png" alt="Trường Đại học Công nghệ Thông tin | University of Information Technology">
  </a>
</p>

<h1 align="center"><b>COMPUTATIONAL THINKING</b></h1>

---

# THÀNH VIÊN NHÓM

| STT | MSSV     | Họ và Tên                 | Chức vụ     | Email                    |
|-----|----------|---------------------------|-------------|--------------------------|
| 1   | 23520732 | Đặng Anh Khoa             | Nhóm trưởng | 23520732@gm.uit.edu.vn   |
| 2   | 23520986 | Phan Công Nam             | Thành viên  | 23520986@gm.uit.edu.vn   |
| 3   | 23520705 | Phạm Minh Bảo Khang       | Thành viên  | 23520705@gm.uit.edu.vn   |
| 4   | 23520746 | Nguyễn Đăng Khoa          | Thành viên  | 23520746@gm.uit.edu.vn   |
| 5   | 23520899 | Nguyễn Thế Luân           | Thành viên  | 23520899@gm.uit.edu.vn   |

---


# GIỚI THIỆU MÔN HỌC

- **Tên môn học:** Tư duy tính toán
- **Mã môn học:** CS117
- **Mã lớp:** CS117.Q11
- **Năm học:** Học kỳ 1 (2025 - 2026)
- **Giảng viên:** TS. Ngô Đức Thành

---

## GIỚI THIỆU ĐỒ ÁN

Tại các đô thị lớn ở Việt Nam như Thành phố Hồ Chí Minh và Hà Nội, tình trạng ngập lụt do
mưa lớn đã trở thành vấn đề thường xuyên. Tuy nhiên, khó khăn lớn không chỉ nằm ở việc xảy
ra ngập mà còn ở sự thiếu hụt thông tin định lượng về quá trình thoát nước, đặc biệt là thời điểm
nước rút sau mưa. Hiện nay, các cảnh báo ngập lụt chủ yếu mang tính định tính, chưa cung cấp
được các ước lượng cụ thể về thời gian nước rút tại từng khu vực, gây khó khăn cho người
dân trong việc di chuyển, sinh hoạt và làm gia tăng rủi ro thiệt hại tài sản cũng như ùn tắc giao
thông. Đồng thời, việc thiếu dự báo định lượng cũng làm giảm hiệu quả trong công tác điều tiết
và ứng phó của các cơ quan quản lý. Xuất phát từ thực tế đó, đề tài “Dự đoán thời gian nước
rút tại một khu vực ngập do mưa” được thực hiện nhằm xây dựng một phương pháp dự đoán
mang tính định lượng, dựa trên việc mô hình hóa các yếu tố địa hình, hạ tầng thoát nước và đặc
trưng mưa.

---

## NGUỒN DỮ LIỆU

Dữ liệu dùng để huấn luyện và kiểm thử mô hình trong đồ án này **là dữ liệu nhân tạo (synthetic data)**, được sinh tự động bằng chương trình để phục vụ mục đích học tập và mô phỏng, **không phải dữ liệu thực tế**.

---

## CẤU TRÚC THƯ MỤC

```text
.
├── data/                        # Dữ liệu nhân tạo dùng cho huấn luyện
│   ├── dem_train.npy            # DEM (Digital Elevation Model)
│   ├── building_mask_train.npy  # Mặt nạ công trình xây dựng
│   └── boundary_polygon.json    # Đa giác xác định khu vực nghiên cứu
│
├── model_utils.py               # Các hàm tiền xử lý, tạo đặc trưng và huấn luyện
├── train_pipeline.py            # Pipeline tạo dữ liệu, huấn luyện và lưu mô hình
├── model.pkl                    # Mô hình LightGBM đã huấn luyện (tạo sau khi train)
├── app.py                       # Ứng dụng Streamlit minh họa kết quả
└── README.md                    # Tài liệu mô tả đồ án
```

---

## CÁCH CHẠY CHƯƠNG TRÌNH

### 1. Cài đặt thư viện cần thiết
```bash
pip install numpy scipy scikit-learn lightgbm streamlit matplotlib plotly joblib
```

### 2. Huấn luyện mô hình
```bash
python train_pipeline.py
```

Sau khi chạy xong, file model.pkl sẽ được tạo ra

### 3. Chạy ứng dụng demo
```bash
streamlit run app.py
```

### Mở trình duyệt tại địa chỉ Streamlit cung cấp (thường là http://localhost:8501)
