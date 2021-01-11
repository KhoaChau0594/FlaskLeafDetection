_Hai file mô hình đã được huấn luyện quá nặng nên sẽ được trên drive ở địa chỉ sau:
	https://drive.google.com/drive/folders/15UTcUdMfu-nAfVlQqVRYnFT8JLMlNSdq?usp=sharing
	Tải hai file về và copy vào thư mục Flask/trained_model

_Tạo folder leaf_disease_app. Copy file requirements.txt từ git vào thư mục vừa tạo. Mở Terminal, dịch chuyển đến thư muck leaf_disease_app và nhập các lệnh sau:
	venv\Scripts\activate
	pip install -r requirements.txt
	

_Tải project về, Copy vào thư mục Flask vào thư mục leaf_disease_app.

_Để chạy app gõ lênh:
	$env:FLASK_APP = "app.py"
	$env:FLASK_DEBUG = 1 
	python -m flask run
