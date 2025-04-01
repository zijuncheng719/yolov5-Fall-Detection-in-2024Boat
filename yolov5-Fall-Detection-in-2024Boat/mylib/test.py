import cv2

video_capture = cv2.VideoCapture(4)

while True:
    _, frame = video_capture.read()
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
