from django.shortcuts import render
from django.http import HttpResponse, StreamingHttpResponse

import cv2
from .faiss_face_recognition_py38 import FaceRecognitionFAISS

# Create your views here.
def index(request):
    return render(request, "cctv/main.html")

def video_feed(request):
    return StreamingHttpResponse(stream(request),
    content_type='multipart/x-mixed-replace;boundary=frame')

# 영상을 보내는 함수
def stream(request):
    cap = cv2.VideoCapture(0)

    # Faiss 라이브러리
    fr = FaceRecognitionFAISS(
        org_data_path="./dataset/org_data",
        dataset_path="./dataset/data",
        train_dir="./dataset/train",
        test_tmp_path="./dataset/test/test1.jpg",
        face_margin=20,
        top_k=5,
        min_vote=3,
    )
    fr.load_model("./dataset/train/face_20260518.bin")
    result = fr.predict("./dataset/test/test5.jpg")
    print("최종 예측:", result)


    while True:
        ret, frame = cap.read()
        if not ret:
            print("카메라를 인식할 수 없습니다.")
            break

        # frame -> 사진
        # 서비스(ai) 기능 구현

        result = fr.predict_numpy(frame)
        print("최종 예측:", result)


        # 서버로 데이터를 전송
        # 이미지를 binary 웹에서 전송이 가능한 형태
        image_bytes = cv2.imencode('.jpg',frame)[1].tobytes()
        # 서버로 전송
        yield(b'--frame\r\n'
        b'Content-type:image/jpeg\r\n\r\n' + image_bytes
        + b'\r\n')