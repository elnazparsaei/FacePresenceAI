import psycopg2
import cv2
import numpy as np

# اتصال به دیتابیس
conn = psycopg2.connect(
    dbname="face_recognition_db",
    user="faceuser",
    password="123456",
    host="localhost",
    port=5432
)

cur = conn.cursor()

# آی‌دی چهره‌ای که می‌خوای ببینی
target_id = 16  # می‌تونی 10 هم امتحان کنی

# کوئری گرفتن تصویر
cur.execute("SELECT image FROM persons WHERE id=%s;", (target_id,))
record = cur.fetchone()

if record:
    img_data = record[0]
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # نمایش تصویر
    cv2.imshow(f"Face with id={target_id}", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print(f"چهره‌ای با id={target_id} پیدا نشد.")

cur.close()
conn.close()
