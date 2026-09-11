# Báo cáo bài thực hành Ngày 1 – Đọc nhãn từ đầu ra YOLO11

**Ngày chạy:** 11/09/2026

**Runtime Colab:** CPU (Local Python 3.11 Environment)

**Python / PyTorch / Ultralytics:** Python 3.11.11 / PyTorch 2.5.1 / Ultralytics 8.4.145

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Thay đổi so với notebook nguồn:** Không

> ZIP do notebook tạo có tên `<KHOA>-DAY01-report.zip` (ví dụ: `K4-DAY01-report.zip`). Giải nén rồi đặt trực tiếp `REPORT.md` và
> `day1_lab_outputs/` vào thư mục `report/` của repository tạo từ template. Không ghi họ tên, MSSV,
> email, số điện thoại hoặc dữ liệu cá nhân khác. Nộp link repository trên VLearn; tài khoản VLearn xác
> định người nộp.

## 1. Phân loại ảnh – prediction cấp ảnh

Nguồn evidence: `classification_predictions.json`, sample `traffic`.

- Record hạng 1 (`class_id`, `class_name`, `rank`, `score`, `taxonomy_name`):
```json
{
  "class_id": 468,
  "class_name": "cab",
  "rank": 1,
  "score": 0.510915,
  "taxonomy_name": "ImageNet-1K"
}
```

- Record này mô tả toàn ảnh như thế nào?
  Record này gán một nhãn phân loại duy nhất đại diện cho toàn bộ nội dung bức ảnh (`image-level classification`). Cụ thể, mô hình dự đoán chủ thể nổi bật nhất của khung cảnh giao thông `traffic` là xe taxi ("cab", class ID 468) với độ tin cậy (`model score`) là 0.510915 (~51.09%) và xếp ở vị trí cao nhất (rank 1) trong phân phối xác suất các lớp. Record này không chỉ định tọa độ vị trí, không có bounding box và không phân biệt có bao nhiêu phương tiện riêng lẻ trong ảnh.

- Ai định nghĩa class list mà checkpoint có thể dự đoán?
  Class list được định nghĩa bởi taxonomy của tập dữ liệu huấn luyện – cụ thể ở đây là ImageNet-1K (gồm 1.000 danh mục lớp do các chuyên gia, nhà nghiên cứu xây dựng bộ dữ liệu ImageNet và ban tổ chức ILSVRC chuẩn hóa). Checkpoint `yolo11n-cls.pt` bị ràng buộc cố định trong không gian nhãn này và chỉ có thể dự đoán một trong 1.000 lớp đó.

- Vì sao cần giữ cả ID, tên lớp và tên taxonomy?
  - `class_id` (468): Cung cấp mã định danh số học duy nhất (unique numeric identifier), giúp hệ thống máy tính, pipeline lập trình và cơ sở dữ liệu xử lý, lập chỉ mục nhất quán mà không bị lỗi encoding hay khoảng trắng.
  - `class_name` ("cab"): Cung cấp nhãn ngữ nghĩa ngôn ngữ tự nhiên (human-readable semantic label), giúp con người (annotator, reviewer, kỹ sư) đọc hiểu trực quan.
  - `taxonomy_name` ("ImageNet-1K"): Xác định phạm vi và ngữ cảnh phân loại chuẩn. Cùng một đối tượng nhưng trong taxonomy khác (như COCO-80 hay OpenImages) có thể mang ID khác hoặc được gộp thành tên lớp khác (ví dụ: COCO chỉ có class "car" thay vì tách riêng "cab"). Việc giữ tên taxonomy đảm bảo truy xuất nguồn gốc (provenance) và tránh nhập nhằng khi tích hợp đa hệ thống.

- Nếu ảnh có nhiều chủ thể, guideline cần quy định điều gì?
  Khi khung hình xuất hiện nhiều chủ thể hỗn hợp (như ảnh `traffic` có cả xe taxi, minibus, xe cảnh sát, người đi đường), guideline cần quy định rõ:
  1. **Quy tắc ưu tiên chủ thể (Priority rule)**: Xác định nhãn dựa trên chủ thể chiếm diện tích pixel lớn nhất ở tiền cảnh (dominant object) hoặc nằm ở trọng tâm bức ảnh, hoặc chủ thể mang mục đích nghiệp vụ quan trọng nhất.
  2. **Ranh giới bài toán**: Quy định rõ đây là bài toán Single-label (bắt buộc chọn duy nhất 1 nhãn chính theo thứ tự ưu tiên) hay Multi-label classification (cho phép gán nhiều nhãn đồng thời cho các chủ thể cùng xuất hiện).
  3. **Quy tắc phân xử và escalation**: Quy định nhãn mặc định/background khi các chủ thể có độ nổi bật ngang nhau, hoặc quy trình chuyển lên cấp trên (reviewer/lead) để đưa ra quyết định thống nhất.

- Vì sao model score không phải ground truth?
  Model score (confidence score: 0.510915) chỉ là giá trị xác suất toán học do mạng nơ-ron tính toán dựa trên trọng số đã học từ dữ liệu quá khứ. Điểm số này có thể bị sai lệch (overconfidence/underconfidence), bị ảnh hưởng bởi nhiễu ảnh hoặc ảo giác của mô hình. Trong khi đó, Ground Truth là "chân lý thực tế" đã được chuyên gia/con người thẩm định, kiểm tra chéo và xác nhận đạt chuẩn 100% theo guideline. Vì vậy, model score chỉ mang tính tham khảo/lọc ngưỡng, tuyệt đối không được coi là ground truth.

## 2. Phát hiện vật thể – lớp và box cho từng object

Nguồn evidence: `detection_predictions.json` và `visuals/detection_predictions.png`, sample `kitchen`.

- Một record (`class_name`, `score`, `bbox_xyxy`, `bbox_width`, `bbox_height`):
```json
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
```

- Diễn giải vị trí box bằng lời:
  Box của đối tượng "person" (score 0.912625) có định dạng tọa độ pixel `xyxy` với gốc (0, 0) ở góc trên-trái bức ảnh: góc trên-trái của box tại (x_min = 385.33 px, y_min = 69.24 px) và góc dưới-phải tại (x_max = 498.92 px, y_max = 348.92 px), với kích thước chiều rộng bbox_width = 113.58 px và chiều cao bbox_height = 279.68 px. Box này đóng khung trọn vẹn người đầu bếp mặc áo trắng đang đứng quay lưng ở khu vực trung tâm lệch phải của căn bếp, kéo dài từ đỉnh đầu/mũ xuống đến gót chân.

- So sánh số prediction ở hai threshold:
  - Ngưỡng threshold = 0.35 (mặc định): Mô hình phát hiện 11 vật thể (`person`: 2, `bowl`: 5, `oven`: 2, `cup`: 2).
  - Ngưỡng threshold = 0.60 (ngưỡng cao): Mô hình chỉ giữ lại 6 vật thể có độ tin cậy cao nhất (`person`: 2, `bowl`: 2, `oven`: 2).
  - So sánh: Khi tăng threshold từ 0.35 lên 0.60, số prediction giảm gần một nửa (từ 11 xuống 6 vật thể), loại bỏ 5 đối tượng có độ tin cậy thấp hơn (gồm 3 bát và 2 cốc phụ), chỉ giữ lại các đối tượng rõ ràng nhất.

- Điều gì thay đổi đối với độ bao phủ và khối lượng reviewer cần xem?
  - **Khi hạ threshold (0.20)**: Độ bao phủ (recall) tăng lên, bắt được nhiều vật thể nhỏ và mờ hơn, nhưng tỷ lệ dương tính giả (false positives) cao. Khối lượng công việc của reviewer tăng mạnh do phải rà soát qua nhiều box nhiễu và tốn thời gian bấm xóa các box sai.
  - **Khi nâng threshold (0.60)**: Độ chính xác (precision) của các box được giữ lại tăng lên, giảm tải việc dọn rác cho reviewer, nhưng độ bao phủ giảm mạnh (bỏ sót nhiều vật thể thực tế - false negatives). Khi đó, reviewer hoặc annotator phải tốn nhiều công sức để tự tìm và vẽ bổ sung thủ công các box bị thiếu.

- Đề xuất một quy tắc box chặt:
  "Bounding box phải bao bọc sát nhất có thể toàn bộ phần nhìn thấy được (visible pixels) của đối tượng; 4 cạnh của hộp chữ nhật phải chạm đúng vào 4 điểm cực biên (trên cùng, dưới cùng, xa nhất bên trái, xa nhất bên phải) của vật thể. Dung sai padding không vượt quá 2 pixel, không cắt lẹm vào đối tượng và tuyệt đối không bao gồm khoảng trống thừa của nền trừ khi góc hình học bắt buộc."

- Với object bị che khuất/cắt mép, điều gì cần guideline hoặc escalation quyết định?
  - **Guideline cần quy định rõ**:
    1. Ngưỡng diện tích tối thiểu được gắn nhãn (ví dụ: vật thể bị che khuất > 80% hoặc nhỏ hơn 10x10 px thì bỏ qua).
    2. Quy định đóng box theo phần nhìn thấy (`visible bounding box`) hay ước lượng toàn thể hình học gốc (`amodal bounding box`).
    3. Quy tắc cắt mép (truncated): Bounding box được chạm sát mép ảnh (x=0 hoặc y=0), và đối tượng phải có tối thiểu bao nhiêu % hiển thị trong khung hình để được đánh nhãn.
  - **Escalation**: Khi một đối tượng bị vật cản lớn chia cắt thành 2 phần tách biệt (ví dụ người đứng sau cột hoặc tủ bếp che ngang người), annotator phải gửi escalation để xin chỉ đạo vẽ 1 box to bao trùm hay tách làm 2 box riêng lẻ.

## 3. Phân đoạn theo từng đối tượng – polygon cho mỗi instance

Nguồn evidence: `segmentation_predictions.json` và `visuals/segmentation_prediction.png`, sample `kitchen`.

- Một record (`instance_id`, `class_name`, `score`, số điểm và một phần `polygon_xy`):
```json
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
```

- Polygon bổ sung chi tiết gì so với box?
  Trong khi bounding box chỉ là một khung chữ nhật bao ngoài chứa nhiều diện tích nền hoặc các vật thể lân cận đan xen, Polygon cung cấp đường biên phân đoạn chính xác đến cấp độ từng pixel (pixel-level mask contour) theo đúng hình dạng thực tế của đối tượng. Polygon phân biệt rõ pixel nào thuộc về cơ thể người và pixel nào là hậu cảnh/bếp xung quanh, giúp tính toán chính xác diện tích thực, chu vi và hình thái học của đối tượng.

- `instance_id` dùng để làm gì và không phải loại ID nào?
  - `instance_id` (ví dụ `kitchen-001`) dùng để định danh duy nhất từng cá thể đối tượng độc lập trong cùng một bức ảnh, giúp phân biệt giữa các đối tượng cùng một lớp (ví dụ phân biệt người thứ nhất `kitchen-001` với người thứ hai `kitchen-009`, phân biệt bát `kitchen-002` với bát `kitchen-003`).
  - `instance_id` **KHÔNG PHẢI** là:
    1. `class_id`: Mã định danh chung của danh mục lớp (ví dụ `class_id = 0` là đại diện cho lớp person nói chung).
    2. `tracking_id`: Mã định danh đối tượng duy trì nhất quán qua chuỗi khung hình video theo thời gian (video tracking).

- Đề xuất một quy tắc biên mask:
  "Đường biên polygon phải bám khít sát ranh giới pixel thực của đối tượng với độ sai lệch không quá ±1 pixel. Không được cắt lẹm vào phần thân của đối tượng (under-segmentation) và không để viền mask tràn ra ngoài hậu cảnh (over-segmentation). Đối với các chi tiết phức tạp hoặc biên cong (như cánh tay, bàn tay, quai thìa), polygon phải có đủ mật độ điểm để phản ánh đúng hình dáng tự nhiên, không làm gãy góc méo mó."

- Với vùng mờ/tiếp xúc/che khuất, điều gì cần guideline hoặc escalation quyết định?
  - **Guideline cần quy định**:
    1. Ranh giới tiếp xúc (contact boundary): Khi hai đối tượng cùng lớp đặt sát/chồng đè lên nhau (như các bát xếp chồng), quy định đường phân cách dựa trên độ tương phản màu sắc hoặc đường bóng đổ.
    2. Vùng biên mờ do chuyển động (motion blur) hoặc out-of-focus: Quy định viền mask đi theo ranh giới trung vị của dải chuyển tiếp (50% alpha/contrast threshold).
  - **Escalation**: Khi một instance bị che khuất tạo thành các mảng pixel rời rạc (ví dụ cánh tay giơ qua kệ bị thanh gỗ chắn ngang), annotator cần escalation để xin chỉ đạo gộp thành một multi-polygon cho 1 instance hay đánh nhãn thành các instance độc lập.

## 4. Vòng đời và kiểm tra chất lượng

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

- Phân loại ảnh:
  - Đơn vị/định dạng ground truth: Nhãn cấp ảnh (Image-level label): Single-label (ID/tên class) hoặc Multi-label set.
  - Lỗi hoặc điểm mơ hồ quan sát được: Ảnh traffic có nhiều loại phương tiện (taxi, bus, xe cảnh sát) cùng xuất hiện, khó xác định 1 nhãn đại diện duy nhất.
  - Annotator làm gì?: Đọc kỹ thứ tự ưu tiên trong guideline; chọn nhãn cho xe ở tiền cảnh chiếm diện tích lớn nhất; nếu ngang hàng thì gửi escalation.
  - Reviewer xem gì?: Kiểm tra nhãn có thuộc đúng taxonomy ImageNet-1K không; đối chiếu quy tắc ưu tiên trong guideline; tránh nhầm giữa các lớp tương đồng.

- Phát hiện vật thể:
  - Đơn vị/định dạng ground truth: Danh sách bounding box 2D (bbox_xyxy pixel) kèm class_id cho từng vật thể.
  - Lỗi hoặc điểm mơ hồ quan sát được: Ảnh kitchen có bát đĩa nhỏ bị che khuất hoặc xếp chồng; đồ kim loại phản chiếu dễ gán nhầm.
  - Annotator làm gì?: Đóng box khít 4 điểm cực biên của phần nhìn thấy; không bỏ sót vật thể rõ nét; tuyệt đối không đóng box lên hình phản chiếu.
  - Reviewer xem gì?: Kiểm tra độ khít (tightness) của bounding box; kiểm tra trùng lặp box (IoU thừa); đối chiếu các ca bị che khuất hoặc cắt mép ảnh.

- Instance segmentation:
  - Đơn vị/định dạng ground truth: Tập hợp đa giác khép kín (polygon_xy pixel) kèm class_id và instance_id.
  - Lỗi hoặc điểm mơ hồ quan sát được: Ranh giới giữa áo người và nền bếp sáng bị lẫn màu; bàn bếp (dining table) bị che khuất phức tạp.
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

- [x] `classification_predictions.json`
- [x] `detection_predictions.json`
- [x] `segmentation_predictions.json`
- [x] `IMAGE_ATTRIBUTION.md`
- [x] `visuals/classification_top5.png`
- [x] `visuals/detection_predictions.png`
- [x] `visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`.
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output.
