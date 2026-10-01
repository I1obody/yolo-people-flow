import cv2

cap = cv2.VideoCapture('testvideo/144.avi')
fps = cap.get(cv2.CAP_PROP_FPS)
print(f'OpenCV FPS: {fps}')

count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 10000 == 0:
        print(f'Count: {count}')

cap.release()
print(f'Total frames read: {count}')
print(f'Duration at reported FPS: {count/fps}')
