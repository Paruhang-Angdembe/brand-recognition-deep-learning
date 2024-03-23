# Real Time Brand Detection

This project is to stream object detection on web browser using Django framework.

The project can be deployed on Ubuntu and Windows.

## 1. Steps to use

1. Download file "yolov3_coco.pb" from the link below and locate it in folder "yolov3_weight": ([gotolink](https://drive.google.com/drive/u/1/folders/1apB-yPIxxzC9D6_iAaQrXWuGpbWIK6Lp))

```bashrc
https://drive.google.com/drive/u/1/folders/1apB-yPIxxzC9D6_iAaQrXWuGpbWIK6Lp
```

2. Create your virtual environment and install required packages:

```bashrc
$ pip install -r requirements.txt
```

3. Run program:

```bashrc
$ python manage.py runserver
```

4. Open any web browser and navigate to URL (home page):

```bashrc
http://127.0.0.1:8000/
```
