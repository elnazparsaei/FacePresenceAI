import cv2 as cv
from face_recognition_module import FaceRecognizer

# -----------------------------
# تنظیمات اولیه
source_type = "camera"  # یا 'ip' یا 'video'
video_source = 0        # 0 برای لپ‌تاپ، یا آدرس IP دوربین موبایل مثل "http://192.168.1.26:8080/video"
thresh = 0.5
images_path = "images"                # مسیر پوشه تصاویر دیتابیس
output_path = "output"               # مسیر ذخیره چهره‌های بریده‌شده

# آدرس مدل‌ها
detector_model_path = "/home/elnaz/soroush/Face_Plate/weight/face_detection_yunet_2023mar.onnx"
recognizer_model_path = "/home/elnaz/soroush/Face_Plate/weight/model.onnx"
# -----------------------------

# مرحله ۱: ساخت سیستم تشخیص
fr_system = FaceRecognizer(
    detector_model_path=detector_model_path,
    recognizer_model_path=recognizer_model_path,
    score_threshold=0.65,
    match_threshold=thresh
)

# مرحله ۲: استخراج ویژگی‌ها از عکس‌ها
print("📸 در حال ساخت دیتابیس ویژگی چهره‌ها...")
fr_system.extract_features_from_folder(images_path, output_path)
print("✅ دیتابیس چهره‌ها ساخته شد.")

# مرحله ۳: شروع شناسایی از منبع تصویری
print("🎥 در حال شروع شناسایی چهره...")
fr_system.set_video_source(video_source)
fr_system.start_recognition(display=True)
print("🔚 پایان برنامه.")

