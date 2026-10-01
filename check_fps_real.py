import cv2
import subprocess
import json

video_path = 'testvideo/144.avi'

# Get real FPS from ffprobe
cmd = [
    'ffprobe', '-v', 'error',
    '-select_streams', 'v:0',
    '-show_entries', 'stream=r_frame_rate,avg_frame_rate,time_base',
    '-of', 'json',
    video_path
]
result = subprocess.run(cmd, capture_output=True, text=True)
data = json.loads(result.stdout)
stream = data['streams'][0]
r_frame_rate = stream['r_frame_rate']
avg_frame_rate = stream['avg_frame_rate']
time_base = stream['time_base']

print(f'r_frame_rate: {r_frame_rate}')
print(f'avg_frame_rate: {avg_frame_rate}')
print(f'time_base: {time_base}')

# Parse fractions
def parse_frac(frac):
    num, den = map(int, frac.split('/'))
    return num/den if den != 0 else 0

r_fps = parse_frac(r_frame_rate)
avg_fps = parse_frac(avg_frame_rate)
print(f'r_fps: {r_fps}')
print(f'avg_fps: {avg_fps}')

# OpenCV FPS
cap = cv2.VideoCapture(video_path)
cv2_fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
cap.release()
print(f'OpenCV FPS: {cv2_fps}')
print(f'Frame count: {frame_count}')
print(f'Duration from OpenCV: {frame_count/cv2_fps}')
print(f'Duration from r_fps: {frame_count/r_fps}')
