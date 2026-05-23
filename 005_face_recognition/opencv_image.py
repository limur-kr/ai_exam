# 1. 라이브러리 불러오기
import cv2
import numpy as np


# 2. 데이터 불러오기
img = cv2.imread("photo.jpg")

# 3. 화면에 이미지 출력
cv2.imshow("Image View", img)

# 4. 저장
# cv2.imwrite("people./jpg", img)

# 5. 이미지 분리(B,G,R)
b, g, r = cv2.split(img)
zeros = np.zeros_like(b)

# 각 색상별 3채널 이미지 생성 (OpenCV는 BGR 순서)
blue_img = cv2.merge([b, zeros, zeros])  # B 만 있고 G, R 은 0
green_img = cv2.merge([zeros, g, zeros]) # G 만 있고 B, R 은 0
red_img = cv2.merge([zeros, zeros, r])   # R 만 있고 B, G 은 0

# 결과 출력
# cv2.imshow("Blue Channel (Colored)", blue_img)
# cv2.imshow("Green Channel (Colored)", green_img)
# cv2.imshow("Red Channel (Colored)", red_img)

# 6. 리싸이즈
# resized = cv2.resize(img, (300, 200))
# cv2.imshow("Resized", resized)

# 7. 회전
rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
# cv2.imshow("Rotated", rotated)

# 8. 크롭(특정 부분을 분리해서 잘라내는)
cropped = img[50:250, 100:300]
# cv2.imshow("Cropped", cropped)

# 9. 이진화
gray = cv2.imread("photo.jpg", cv2.IMREAD_GRAYSCALE)
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

cv2.imshow("Binary", binary)



cv2.waitKey(0)
cv2.destroyAllWindows()

