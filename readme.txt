_Hai file mô hình đã được huấn luyện quá nặng nên sẽ được trên drive ở địa chỉ sau:
	https://drive.google.com/drive/folders/15UTcUdMfu-nAfVlQqVRYnFT8JLMlNSdq?usp=sharing
	Tải hai file về và copy vào thư mục Flask/trained_model

_Mở Terminal, dịch chuyển đến folder lưu ứng dụng hiện tại và nhập các lệnh sau:
venv\Scripts\activate
pip install -r requirements.txt
cd Flask
$env:FLASK_APP = "app.py"
$env:FLASK_DEBUG = 1 
python -m flask run