import cv2
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("دوربین باز نشد!")
else:
    ret, frame = cap.read()
    if ret:
        cv2.imshow("تست دوربین", frame)
        cv2.waitKey(0)
    cap.release()