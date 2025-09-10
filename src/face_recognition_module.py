import cv2 as cv
import numpy as np
import pickle
import logging
import os
from datetime import datetime
from utils.utils import generate_random_nid


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FaceRecognizer:
    """کلاس برای شناسایی چهره با استفاده از مدل‌های YU-Net و FaceRecognizerSF."""

    def __init__(self, detector_model_path, recognizer_model_path, score_threshold=0.65, match_threshold=0.97, db_manager=None):
        """راه‌اندازی FaceRecognizer با مدل‌های تشخیص و شناسایی چهره.

        Args:
            detector_model_path (str): مسیر فایل مدل تشخیص چهره
            recognizer_model_path (str): مسیر فایل مدل شناسایی چهره
            score_threshold (float): آستانه امتیاز تشخیص چهره
            match_threshold (float): آستانه تطبیق ویژگی‌ها
            db_manager (DatabaseManager, optional): نمونه DatabaseManager
        """
        if not os.path.exists(detector_model_path) or not os.path.exists(recognizer_model_path):
            raise FileNotFoundError("یکی از فایل‌های مدل یافت نشد.")
        
        self.detector_model_path = detector_model_path
        self.recognizer_model_path = recognizer_model_path
        self.score_threshold = score_threshold
        self.match_threshold = match_threshold
        self.db_manager = db_manager
        self.features = {}

        try:
            self.face_detector = cv.FaceDetectorYN.create(
                model=detector_model_path,
                config=None,
                input_size=(0, 0)
            )
            self.face_detector.setScoreThreshold(score_threshold)
            self.recognizer = cv.FaceRecognizerSF.create(
                model=recognizer_model_path,
                config=None
            )
            logger.info("✅ مدل‌های تشخیص و شناسایی چهره با موفقیت بارگذاری شدند.")
        except Exception as e:
            logger.error(f"❌ خطا در بارگذاری مدل‌ها: {e}")
            raise

    def set_thresholds(self, score_threshold=None, match_threshold=None):
        """تنظیم آستانه‌های تشخیص و تطبیق.

        Args:
            score_threshold (float, optional): آستانه جدید تشخیص
            match_threshold (float, optional): آستانه جدید تطبیق
        """
        if score_threshold is not None:
            self.score_threshold = score_threshold
            self.face_detector.setScoreThreshold(score_threshold)
            logger.info(f"✅ آستانه تشخیص به {score_threshold} تنظیم شد.")
        if match_threshold is not None:
            self.match_threshold = match_threshold
            logger.info(f"✅ آستانه تطبیق به {match_threshold} تنظیم شد.")

    def extract_features(self, image):
        """استخراج ویژگی چهره از تصویر ورودی.

        Args:
            image (np.ndarray): تصویر ورودی

        Returns:
            np.ndarray: ویژگی‌های چهره یا None
        """
        if image is None or image.size == 0:
            logger.error("❌ تصویر ورودی نامعتبر است.")
            return None

        h, w = image.shape[:2]
        new_w = (w // 32) * 32
        new_h = (h // 32) * 32
        resized_image = cv.resize(image, (new_w, new_h))
        self.face_detector.setInputSize((new_w, new_h))

        try:
            _, faces = self.face_detector.detect(resized_image)
            if faces is None or len(faces) == 0:
                logger.warning("❌ چهره‌ای در تصویر یافت نشد.")
                return None

            face = faces[0]
            aligned = self.recognizer.alignCrop(image, face)
            if aligned is None or aligned.size == 0:
                logger.error("❌ تصویر ترازشده نامعتبر است.")
                return None
            features = self.recognizer.feature(aligned)
            if features is None or features.size == 0:
                logger.error("❌ ویژگی‌های چهره استخراج نشد.")
                return None
            features = np.squeeze(features)
            if features.shape != (512,):
                logger.error(f"❌ شکل ویژگی‌ها نامعتبر است: {features.shape}")
                return None
            logger.info(f"✅ ویژگی‌های چهره استخراج شدند: shape={features.shape}")
            return features
        except Exception as e:
            logger.error(f"❌ خطا در استخراج ویژگی‌ها: {e}")
            return None

    def cosine_similarity(self, feat1, feat2):
        """محاسبه شباهت کسینوسی بین دو بردار ویژگی.

        Args:
            feat1 (np.ndarray): بردار ویژگی اول
            feat2 (np.ndarray): بردار ویژگی دوم

        Returns:
            float: مقدار شباهت کسینوسی
        """
        feat1 = np.squeeze(feat1)
        feat2 = np.squeeze(feat2)
        
        if feat1.shape != (512,) or feat2.shape != (512,):
            logger.error(f"❌ شکل بردارها نامعتبر است: feat1={feat1.shape}, feat2={feat2.shape}")
            raise ValueError(f"شکل بردارها نامعتبر است: feat1={feat1.shape}, feat2={feat2.shape}")
        
        dot_product = np.dot(feat1, feat2)
        norm_product = np.linalg.norm(feat1) * np.linalg.norm(feat2)
        
        if norm_product == 0:
            logger.warning("⚠️ نرم یکی از بردارها صفر است")
            return 0.0
        
        return dot_product / norm_product

    def load_features_from_db(self):
        """بارگذاری ویژگی‌های چهره از پایگاه داده."""
        if self.db_manager is None:
            logger.warning("⚠️ Database manager not set.")
            return

        self.features.clear()
        records = self.db_manager.get_all_person_features()
        for first_name, last_name, national_id, features in records:
            full_name = f"{first_name} {last_name}"
            try:
                if features is not None and len(features) > 0:
                    features_array = np.frombuffer(features, dtype=np.float32)
                    if features_array.shape == (512,) and np.all(np.isfinite(features_array)) and np.linalg.norm(features_array) > 0:
                        self.features[national_id] = features_array
                        logger.info(f"✅ ویژگی‌ها برای کد ملی {national_id} بارگذاری شد: shape={features_array.shape}, norm={np.linalg.norm(features_array)}")
                    else:
                        logger.warning(f"⚠️ ویژگی‌های نامعتبر برای {full_name}: shape={features_array.shape}, norm={np.linalg.norm(features_array)}")
                else:
                    logger.warning(f"⚠️ ویژگی‌های خالی برای {full_name}")
            except Exception as e:
                logger.error(f"❌ خطا در بارگذاری ویژگی‌های {full_name}: {e}")
        logger.info(f"✅ {len(self.features)} ویژگی چهره از دیتابیس بارگذاری شد.")

    def recognize_from_frame(self, frame):
        """شناسایی چهره از یک فریم ویدئو.

        Args:
            frame (np.ndarray): فریم ورودی

        Returns:
            tuple: (فریم حاشیه‌نویسی‌شده, تصویر چهره، تصویر تطبیق‌شده، اطلاعات فرد، ویژگی‌ها)
        """
        if frame is None or frame.size == 0:
            logger.warning("⚠️ فریم ورودی خالی است!")
            return frame, None, None, None, None

        h, w = frame.shape[:2]
        self.face_detector.setInputSize((w, h))
        try:
            _, faces = self.face_detector.detect(frame)
            if faces is None or len(faces) == 0:
                logger.warning("❌ چهره‌ای در فریم یافت نشد.")
                return frame, None, None, None, None

            face = faces[0]
            x, y, fw, fh = list(map(int, face[:4]))
            face_img = frame[y:y + fw, x:x + fh].copy()

            aligned = self.recognizer.alignCrop(frame, face)
            feature = self.recognizer.feature(aligned)
            feature = np.squeeze(feature)
            if feature.shape != (512,):
                logger.warning(f"⚠️ شکل ویژگی استخراج‌شده نامعتبر است: {feature.shape}")
                return frame, face_img, None, None, feature

            best_score = -1
            best_nid = None
            matched_img = None
            matched_info = None

            for nid, db_feat in self.features.items():
                score = self.cosine_similarity(feature, db_feat)
                score_value = score.item() if isinstance(score, np.ndarray) else score

                if score_value > best_score:
                    best_score = score_value
                    best_nid = nid

            current_date = datetime.now().date()
            current_time = datetime.now().time().strftime("%H:%M:%S")

            if best_score >= self.match_threshold and best_nid:
                label = f"({round(best_score * 100, 1)}%)"
                if self.db_manager:
                    person_info = self.db_manager.get_person_by_national_id(best_nid)
                    if person_info:
                        matched_info = {
                            "entry_date": current_date,
                            "entry_time": current_time,
                            "exit_time": None,
                            "permission": person_info.get("permission", "false"),
                            "confidence": round(best_score * 100, 1),
                            "camera_number": 0,
                            "nid": best_nid
                        }
                        matched_img = self.db_manager.get_person_image(best_nid)
                        self.db_manager.update_access_log(best_nid, exit_time=current_time)
            else:
                label = "unknown"
                best_nid = generate_random_nid()
                matched_info = {
                    "entry_date": current_date,
                    "entry_time": current_time,
                    "exit_time": None,
                    "permission": "false",
                    "confidence": round(best_score * 100, 1),
                    "camera_number": 0,
                    "nid": best_nid
                }
                self.db_manager.insert_access_log(best_nid, current_date, current_time, None)

            cv.rectangle(frame, (x, y), (x + fw, y + fh), (0, 0, 255), 2)
            cv.putText(frame, label, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            return frame, face_img, matched_img, matched_info, feature
        except Exception as e:
            logger.error(f"❌ خطا در شناسایی چهره از فریم: {e}")
            return frame, None, None, None, None

    def add_person_to_db(self, first_name, last_name, national_id, image, features):
        """افزودن یک فرد جدید به پایگاه داده.

        Args:
            first_name (str): نام
            last_name (str): نام خانوادگی
            national_id (str): کد ملی
            image (bytes): تصویر چهره
            features (np.ndarray): ویژگی‌های چهره
        """
        if self.db_manager is None:
            logger.warning("⚠️ Database manager not set.")
            return

        try:
            if features is None or features.size == 0:
                logger.error(f"❌ ویژگی‌های چهره برای {first_name} {last_name} خالی یا نامعتبر است.")
                return
            features = np.squeeze(features)
            if features.shape != (512,) or not np.all(np.isfinite(features)) or np.linalg.norm(features) == 0:
                logger.error(f"❌ ویژگی‌های چهره برای {first_name} {last_name} معتبر نیست (شکل: {features.shape}).")
                return

            success = self.db_manager.insert_person(first_name, last_name, national_id, image, features)
            if success:
                self.features[national_id] = features
                logger.info(f"✅ فرد {first_name} {last_name} با کد ملی {national_id} به دیتابیس و ویژگی‌ها افزوده شد.")
            else:
                logger.error(f"❌ خطا در افزودن فرد {first_name} {last_name} به دیتابیس.")
        except Exception as e:
            logger.error(f"❌ خطا در افزودن فرد به دیتابیس: {e}")