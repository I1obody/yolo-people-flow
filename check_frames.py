import cv2
cap = cv2.VideoCapture('testvideo/144.avi')
cnt = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    cnt += 1
    if cnt % 100000 == 0:
        print(cnt)
cap.release()
print('total', cnt)
