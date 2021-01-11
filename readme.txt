_Hai file mô hình đã được huấn luyện quá nặng nên sẽ được trên drive ở địa chỉ sau:
	https://drive.google.com/drive/folders/15UTcUdMfu-nAfVlQqVRYnFT8JLMlNSdq?usp=sharing
	Tải hai file về và copy vào thư mục Flask/trained_model

_Tạo folder leaf_disease_app. Copy file requirements.txt từ git vào thư mục vừa tạo. Mở Terminal, dịch chuyển đến thư muck leaf_disease_app và nhập các lệnh sau:
	python -m venv venv # để tạo môi trường ảo
	venv\Scripts\activate # để activate môi trường ảo
	pip install -r requirements.txt # cài các thư viện cần thiết
	

_Tải project về, Copy vào thư mục FlaskLeafDetection-master vào thư mục leaf_disease_app.

_Để chạy app gõ lênh, mở terminal, dịch chuyển đến thư mục FlaskLeafDetection-master/Flask và gõ lệnh:
	$env:FLASK_APP = "app.py"
	$env:FLASK_DEBUG = 1 
	python -m flask run
