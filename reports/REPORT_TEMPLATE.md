# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

Ngày chạy:

Runtime Colab: CPU/GPU

Python / PyTorch / Ultralytics:

Checkpoint: yolo11n-cls.pt, yolo11n.pt, yolo11n-seg.pt

Thay đổi so với notebook nguồn: Không

> ZIP do notebook tạo có tên <KHOA>-DAY01-report.zip (ví dụ: K4-DAY01-report.zip). Giải nén rồi đặt trực tiếp REPORT.md và
> day1_lab_outputs/ vào thư mục report/ của repository tạo từ template. Không ghi họ tên, MSSV,
> email, số điện thoại hoặc dữ liệu cá nhân khác. Nộp link repository trên VLearn; tài khoản VLearn xác
> định người nộp.

## 1. Phân loại ảnh – prediction cấp ảnh

Nguồn evidence: classification_predictions.json, sample traffic.

- Record hạng 1 (class_id, class_name, rank, score, taxonomy_name):
{
  "class_id": 468,
  "class_name": "cab",
  "rank": 1,
  "score": 0.510915,
  "taxonomy_name": "ImageNet-1K"
}

- Record này mô tả toàn ảnh như thế nào?
  Mô hình dự đoán chủ thể nổi bật nhất của khung cảnh giao thông traffic là xe "cab", class ID 468 với score là 0.510915 và xếp ở vị trí cao nhất trong các class. 

- Ai định nghĩa class list mà checkpoint có thể dự đoán?
  Class list được định nghĩa bởi taxonomy của tập dữ liệu huấn luyện ImageNet-1K được các chuyên gia tham gia xây dựng.

- Vì sao cần giữ cả ID, tên lớp và tên taxonomy?
  - class_id (468): Cung cấp mã định danh duy nhất giúp hệ thống máy tính xử lý, lập chỉ mục nhất quán mà không bị lỗi encoding hay khoảng trắng.
  - class_name ("cab"): Cung cấp nhãn ngữ nghĩa ngôn ngữ tự nhiên giúp con người đọc hiểu trực quan.
  - taxonomy_name ("ImageNet-1K"): Xác định phạm vi và ngữ cảnh phân loại chuẩn. Cùng một đối tượng nhưng trong taxonomy khác có thể mang ID khác hoặc được gộp thành tên class khác. Việc giữ tên taxonomy đảm bảo truy xuất nguồn gốc và tránh nhập nhằng khi tích hợp đa hệ thống.

- Nếu ảnh có nhiều chủ thể, guideline cần quy định điều gì?
  Guideline cần quy định rõ:
  1. Ưu tiên chủ thể: Xác định nhãn dựa trên chủ thể chiếm diện tích lớn nhất ở tiền cảnh hoặc nằm ở trọng tâm bức ảnh.
  2. Xác định loại bài toán: chọn đối tượng chính để gán nhãn, hoặc gán nhiều nhãn cho nhiều đối tượng khác nhau

- Vì sao model score không phải ground truth?
  Ground Truth là "chân lý thực tế" đã được chuyên gia thẩm định và xác nhận đạt chuẩn 100% theo guideline. Còn model score là giá trị dự đoán của mô hình, có thể sai sót, bị ảnh hưởng bởi nhiều yếu tố. Vì vậy, model score chỉ mang tính tham khảo/lọc ngưỡng, tuyệt đối không được coi là ground truth.

## 2. Phát hiện vật thể – lớp và box cho từng object

Nguồn evidence: detection_predictions.json và visuals/detection_predictions.png, sample kitchen.

- Một record (class_name, score, bbox_xyxy, bbox_width, bbox_height):
{
  "class_name": "person",
  "score": 0.912625,
  "bbox_xyxy": [
    385.33,
    69.24,
    498.92,
    348.92
  ],
  "bbox_width": 113.58,
  "bbox_height": 279.68
}


- Diễn giải vị trí box bằng lời:
  Box của đối tượng "person" (score 0.912625) có tọa độ góc trên-trái (x_min = 385.33 px, y_min = 69.24 px) và góc dưới-phải (x_max = 498.92 px, y_max = 348.92 px), với kích thước chiều rộng bbox_width = 113.58 px và chiều cao bbox_height = 279.68 px. Box này đóng khung trọn vẹn người đầu bếp mặc áo trắng.

- So sánh số prediction ở hai threshold:
  - Ngưỡng threshold = 0.35: Mô hình phát hiện 11 vật thể (person: 2, bowl: 5, oven: 2, cup: 2).
  - Ngưỡng threshold = 0.60: Mô hình chỉ giữ lại 6 vật thể có độ tin cậy cao nhất (person: 2, bowl: 2, oven: 2).
  - So sánh: Khi tăng threshold từ 0.35 lên 0.60, số prediction giảm từ 11 xuống 6, loại bỏ 5 đối tượng có độ tin cậy thấp hơn, chỉ giữ lại các đối tượng rõ ràng nhất.


- Điều gì thay đổi đối với độ bao phủ và khối lượng reviewer cần xem?
  - Khi hạ threshold (0.20): Recall tăng lên, bắt được nhiều vật thể nhỏ và mờ hơn, nhưng tỷ lệ FP cao. Khối lượng công việc của reviewer tăng mạnh do phải rà soát qua nhiều box nhiễu và tốn thời gian bấm xóa các box sai.
  - Khi nâng threshold (0.60): Precision của các box được giữ lại tăng lên, giảm tải việc dọn rác cho reviewer, nhưng độ bao phủ giảm mạnh (bỏ sót nhiều vật thể thực tế - FN). Khi đó, reviewer hoặc annotator phải tốn nhiều công sức để tự tìm và vẽ bổ sung thủ công các box bị thiếu.

- Đề xuất một quy tắc box chặt:
  "Bounding box phải bao bọc sát nhất có thể toàn bộ phần nhìn thấy được  của đối tượng; 4 cạnh phải chạm đúng vào 4 điểm cực biên 4 góc của vật thể, không bao gồm khoảng trống thừa của nền"

- Với object bị che khuất/cắt mép, điều gì cần guideline hoặc escalation quyết định?
  - Guideline cần quy định rõ:
    1. Những đối tượng nào bị che khuất quá nhiều hay bị cắt mép thì không được gắn nhãn.
    2. Khi đối tượng bị vật cản lớn chia cắt thành 2 phần tách biệt thì cần vẽ 1 box to bao trùm hay tách làm 2 box riêng lẻ.

## 3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance

Nguồn evidence: segmentation_predictions.json và visuals/segmentation_prediction.png, sample kitchen.

- Một record (instance_id, class_name, score, số điểm và một phần polygon_xy):
{
  "instance_id": "kitchen-001",
  "class_name": "person",
  "score": 0.899318,
  "polygon_point_count": 348,
  "polygon_xy": [
    [446.0, 70.0],
    [445.0, 71.0],
    [444.0, 71.0]
  ]
}


- Polygon bổ sung chi tiết gì so với box?
  Trong khi bounding box chỉ là một khung chữ nhật bao ngoài chứa nhiều diện tích nền hoặc các vật thể lân cận, Polygon cung cấp đường biên phân đoạn chính xác theo đúng hình dạng thực tế của đối tượng. Polygon phân biệt rõ đâu thuộc về cơ thể người và đâu là bếp xung quanh, giúp tính toán chính xác diện tích thực, chu vi và hình dáng của đối tượng.

- instance_id dùng để làm gì và không phải loại ID nào?
  - instance_id  dùng để định danh duy nhất từng đối tượng độc lập trong cùng một bức ảnh, giúp phân biệt giữa các đối tượng cùng một class.
  - instance_id KHÔNG PHẢI là:
    1. class_id: Mã định danh chung của danh mục class.
    2. tracking_id: Mã định danh đối tượng duy trì nhất quán qua chuỗi khung hình video theo thời gian.

- Đề xuất một quy tắc biên mask:
  Đường biên polygon phải bám khít sát ranh giới thực của đối tượng với độ sai lệch không quá nhiều. Không được cắt lẹm vào đối tượng và dính phần ngoài.

- Với vùng mờ/tiếp xúc/che khuất, điều gì cần guideline hoặc escalation quyết định?
  - Guideline cần quy định:
    1. Ranh giới tiếp xúc (contact boundary): Khi hai đối tượng cùng class đặt sát/chồng đè lên nhau (như các bát xếp chồng), quy định đường phân cách dựa trên độ tương phản màu sắc hoặc đường bóng đổ.
    2. Vùng biên mờ do chuyển động (motion blur) hoặc out-of-focus: Quy định viền mask đi theo ranh giới trung vị của dải chuyển tiếp (50% alpha/contrast threshold).
  - Escalation: Khi một instance bị che khuất tạo thành các mảng pixel rời rạc (ví dụ cánh tay giơ qua kệ bị thanh gỗ chắn ngang), annotator cần escalation để xin chỉ đạo gộp thành một multi-polygon cho 1 instance hay đánh nhãn thành các instance độc lập.

## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

- Phân loại ảnh:
  - Đơn vị/định dạng ground truth: Nhãn cấp ảnh (Image-level label): Single-label hoặc Multi-label set.
  - Lỗi hoặc điểm mơ hồ quan sát được: Ảnh traffic có nhiều loại phương tiện (taxi, bus, xe cảnh sát) cùng xuất hiện, khó xác định 1 nhãn đại diện duy nhất.
  - Annotator làm gì?: Đọc kỹ thứ tự ưu tiên trong guideline; chọn nhãn cho xe ở tiền cảnh chiếm diện tích lớn nhất; nếu ngang hàng thì gửi escalation.
  - Reviewer xem gì?: Kiểm tra nhãn có thuộc đúng taxonomy ImageNet-1K không; đối chiếu quy tắc ưu tiên trong guideline; tránh nhầm giữa các class tương đồng.

- Phát hiện vật thể:
  - Đơn vị/định dạng ground truth: Danh sách bounding box 2D kèm class_id cho từng vật thể.
  - Lỗi hoặc điểm mơ hồ quan sát được: Ảnh kitchen có bát đĩa nhỏ bị che khuất hoặc xếp chồng; đồ kim loại phản chiếu dễ gán nhầm.
  - Annotator làm gì?: Đóng box khít 4 điểm biên của phần nhìn thấy; không bỏ sót vật thể rõ nét; tuyệt đối không đóng box lên bóng.
  - Reviewer xem gì?: Kiểm tra độ khít (tightness) của bounding box; kiểm tra trùng lặp box; đối chiếu các ca bị che khuất hoặc cắt mép ảnh.

- Instance segmentation:
  - Đơn vị/định dạng ground truth: Tập hợp đa giác khép kín kèm class_id và instance_id.
  - Lỗi hoặc điểm mơ hồ quan sát được: Ranh giới giữa áo người và nền bếp sáng bị lẫn màu; dining table bị che khuất phức tạp.
  - Annotator làm gì?: Chấm polygon bám sát ranh giới pixel thực; không lẹm vào thân, không tràn ra nền; gán instance_id riêng cho từng cá thể.
  - Reviewer xem gì?: Phóng to 200–400% kiểm tra biên mask (không under/over-segmentation); tránh nhầm lẫn instance ID giữa các vật cùng class.


## 5. An toàn dữ liệu

- Một quy tắc bảo vệ dữ liệu:
  - Tuân thủ nghiêm ngặt nguyên tắc Bảo mật và Tối thiểu hóa Dữ liệu:
  - Tuyệt đối không tải lên, lưu trữ hoặc chia sẻ dữ liệu hình ảnh cá nhân, thông tin định danh (PII), tài liệu mật của doanh nghiệp lên các dịch vụ Cloud Storage hay nền tảng công cộng.
  - Chỉ thao tác trên các tập dữ liệu mẫu mã nguồn mở đã được cấp phép minh bạch.

- Nếu thấy ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng và báo cho:
  - Mentor
  - Lab Coach
  - Trưởng ban tổ chức
  - Các phòng ban của nhà trường

## 6. Danh sách bằng chứng

- [x] classification_predictions.json
- [x] detection_predictions.json
- [x] segmentation_predictions.json
- [x] IMAGE_ATTRIBUTION.md
- [x] visuals/classification_top5.png
- [x] visuals/detection_predictions.png
- [x] visuals/segmentation_prediction.png
- [x] Ô validation cuối notebook báo PASS.
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.
