import psycopg2
import datetime
import pickle
import numpy as np
import logging
import re


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


FEATURES_DIM = 512
NATIONAL_ID_LENGTH = 10

class DatabaseManager:
    """مدیریت اتصال و عملیات پایگاه داده برای سیستم شناسایی چهره."""
    
    def __init__(self, dbname, user, password, host='localhost', port='5432'):
        """اتصال به پایگاه داده PostgreSQL.

        Args:
            dbname (str): نام پایگاه داده
            user (str): نام کاربر
            password (str): رمز عبور
            host (str, optional): هاست پایگاه داده. Defaults to 'localhost'.
            port (str, optional): پورت پایگاه داده. Defaults to '5432'.
        """
        try:
            self.conn = psycopg2.connect(
                dbname='face_recognition_db',
                user='faceuser',
                password='your_password_here',
                host='localhost',
                port=5432
            )
            self.cursor = self.conn.cursor()
            logger.info("✅ اتصال به پایگاه داده با موفقیت انجام شد.")
            self.create_tables()
        except Exception as e:
            logger.error(f"❌ خطا در اتصال به پایگاه داده: {str(e)}")
            raise

    def create_tables(self):
        """ایجاد جدول‌های موردنیاز در صورت عدم وجود."""
        try:
            create_cameras_table = """
            CREATE TABLE IF NOT EXISTS cameras (
                id SERIAL PRIMARY KEY,
                index INTEGER UNIQUE NOT NULL,
                name VARCHAR(100)
            );
            """
            create_persons_table = """
            CREATE TABLE IF NOT EXISTS persons (
                id SERIAL PRIMARY KEY,
                first_name VARCHAR(100) NOT NULL,
                last_name VARCHAR(100) NOT NULL,
                national_id VARCHAR(10) UNIQUE NOT NULL,
                image BYTEA,
                face_features BYTEA,
                permission BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
            create_access_logs_table = """
            CREATE TABLE IF NOT EXISTS access_logs (
                id SERIAL PRIMARY KEY,
                person_id INTEGER REFERENCES persons(id) ON DELETE SET NULL,
                camera_id INTEGER REFERENCES cameras(id) ON DELETE SET NULL,
                entry_date DATE NOT NULL,
                entry_time TIME WITHOUT TIME ZONE NOT NULL,
                exit_time TIME WITHOUT TIME ZONE,
                confidence NUMERIC(5,2),
                status TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
            self.cursor.execute(create_cameras_table)
            self.cursor.execute(create_persons_table)
            self.cursor.execute(create_access_logs_table)


            self.cursor.execute("INSERT INTO cameras (index, name) VALUES (%s, %s) ON CONFLICT (index) DO NOTHING", (0, "Default Camera"))
            self.conn.commit()
            logger.info("✅ جدول‌های cameras، persons و access_logs بررسی/ایجاد شدند.")
        except Exception as e:
            self.conn.rollback()
            logger.error(f"❌ خطا در ایجاد جدول‌ها: {str(e)}")
            raise

    def validate_national_id(self, national_id):
        """اعتبارسنجی کد ملی.

        Args:
            national_id (str): کد ملی برای بررسی.

        Returns:
            bool: True اگر معتبر باشد، False در غیر این صورت.
        """
        return bool(re.match(r'^\d{10}$', national_id))

    def insert_person(self, first_name, last_name, national_id, image_bytes, features):
        """درج یک فرد جدید در پایگاه داده.

        Args:
            first_name (str): نام
            last_name (str): نام خانوادگی
            national_id (str): کد ملی
            image_bytes (bytes): تصویر چهره به‌صورت بایت
            features (np.ndarray): ویژگی‌های چهره

        Returns:
            int: شناسه فرد درج‌شده یا None در صورت خطا
        """
        first_name = first_name.strip()
        last_name = last_name.strip()

        if not first_name or not last_name:
            logger.error("❌ خطا: نام یا نام خانوادگی خالی است.")
            return None
        if not self.validate_national_id(national_id):
            logger.error(f"❌ خطا: کد ملی {national_id} معتبر نیست.")
            return None
        if features is None:
            logger.error(f"❌ خطا: ویژگی چهره برای {first_name} {last_name} برابر با None است.")
            return None

        features = np.squeeze(features)
        if features.shape != (FEATURES_DIM,) or not np.all(np.isfinite(features)) or np.linalg.norm(features) == 0:
            logger.error(f"❌ خطا: ویژگی چهره برای {first_name} {last_name} معتبر نیست (شکل: {features.shape}).")
            return None

        try:
            features_blob = self.features_to_bytes(features)
            if len(features_blob) <= 4:
                logger.error(f"❌ خطا: ویژگی چهره برای {first_name} {last_name} معتبر نیست (طول داده {len(features_blob)} بایت).")
                return None

            query = """
            INSERT INTO persons (first_name, last_name, national_id, image, face_features, created_at)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id;
            """
            created_at = datetime.datetime.now()

            with self.conn:
                self.cursor.execute(query, (first_name, last_name, national_id, image_bytes, features_blob, created_at))
                person_id = self.cursor.fetchone()[0]
                logger.info(f"✅ شخص {first_name} {last_name} با شناسه {person_id} اضافه شد.")
                return person_id

        except psycopg2.IntegrityError as e:
            self.conn.rollback()
            logger.error(f"❌ خطای یکتا بودن (کدملی تکراری؟): {e}")
            return None
        except Exception as e:
            self.conn.rollback()
            logger.error(f"❌ خطا در درج شخص: {e}")
            return None

    def update_person_features(self, national_id, features):
        """به‌روزرسانی ویژگی‌های چهره یک فرد.

        Args:
            national_id (str): کد ملی فرد
            features (np.ndarray): ویژگی‌های جدید چهره
        """
        if not self.validate_national_id(national_id):
            logger.error(f"❌ خطا: کد ملی {national_id} معتبر نیست.")
            return

        features = np.squeeze(features)
        if features.shape != (FEATURES_DIM,) or not np.all(np.isfinite(features)) or np.linalg.norm(features) == 0:
            logger.error(f"❌ خطا: ویژگی چهره برای کد ملی {national_id} معتبر نیست (شکل: {features.shape}).")
            return

        try:
            features_blob = self.features_to_bytes(features)
            query = """
            UPDATE persons SET face_features = %s WHERE national_id = %s
            """
            with self.conn:
                self.cursor.execute(query, (features_blob, national_id))
                logger.info(f"✅ ویژگی‌های شخص با کد ملی {national_id} به‌روزرسانی شد.")
        except Exception as e:
            self.conn.rollback()
            logger.error(f"❌ خطا در به‌روزرسانی ویژگی‌های چهره: {e}")

    def get_person_image(self, national_id):
        """دریافت تصویر فرد بر اساس کد ملی.

        Args:
            national_id (str): کد ملی

        Returns:
            bytes: تصویر به‌صورت بایت یا None
        """
        try:
            query = """
            SELECT image FROM persons WHERE national_id = %s
            """
            self.cursor.execute(query, (national_id,))
            result = self.cursor.fetchone()
            return result[0] if result else None
        except Exception as e:
            logger.error(f"❌ خطا در دریافت تصویر با کد ملی: {e}")
            return None

    def get_person_by_national_id(self, national_id):
        """دریافت اطلاعات فرد بر اساس کد ملی.

        Args:
            national_id (str): کد ملی

        Returns:
            dict: اطلاعات فرد یا None
        """
        if not self.validate_national_id(national_id):
            logger.error(f"❌ خطا: کد ملی {national_id} معتبر نیست.")
            return None

        query = """
        SELECT id, first_name, last_name, national_id, image, permission 
        FROM persons 
        WHERE national_id = %s
        """
        try:
            self.cursor.execute(query, (national_id,))
            result = self.cursor.fetchone()
            if result:
                return {
                    "id": result[0],
                    "first_name": result[1],
                    "last_name": result[2],
                    "national_id": result[3],
                    "image": result[4],
                    "permission": result[5]
                }
            return None
        except Exception as e:
            logger.error(f"❌ خطا در جستجو با کد ملی: {e}")
            return None

    def get_person_by_fullname(self, first_name, last_name):
        """دریافت کد ملی فرد بر اساس نام و نام خانوادگی.

        Args:
            first_name (str): نام
            last_name (str): نام خانوادگی

        Returns:
            str: کد ملی یا None
        """
        query = """
        SELECT national_id FROM persons WHERE first_name = %s AND last_name = %s
        """
        try:
            self.cursor.execute(query, (first_name.strip(), last_name.strip()))
            result = self.cursor.fetchone()
            return result[0] if result else None
        except Exception as e:
            logger.error(f"❌ خطا در دریافت کد ملی با نام: {e}")
            return None

    def get_all_person_features(self):
        """دریافت تمام ویژگی‌های چهره از پایگاه داده.

        Returns:
            list: لیست تاپل‌های شامل اطلاعات افراد و ویژگی‌های چهره
        """
        query = """
        SELECT first_name, last_name, national_id, face_features FROM persons
        """
        try:
            self.cursor.execute(query)
            records = self.cursor.fetchall()
            result = []
            for row in records:
                first_name, last_name, national_id, features_blob = row
                if features_blob is not None:
                    features = self.bytes_to_features(features_blob)
                    features = np.squeeze(features)
                    if features.shape == (FEATURES_DIM,) and np.all(np.isfinite(features)) and np.linalg.norm(features) > 0:
                        result.append((first_name, last_name, national_id, features))
                    else:
                        logger.warning(f"⚠️ ویژگی‌های نامعتبر برای {first_name} {last_name}: shape={features.shape}")
            logger.info(f"✅ {len(result)} ویژگی چهره بارگذاری شد.")
            return result
        except Exception as e:
            logger.error(f"❌ خطا در بارگذاری ویژگی‌ها: {e}")
            return []

    def update_permission(self, person_id, new_permission):
        """به‌روزرسانی وضعیت مجوز فرد.

        Args:
            person_id (int): شناسه فرد
            new_permission (bool): وضعیت جدید مجوز
        """
        query = "UPDATE persons SET permission = %s WHERE id = %s"
        try:
            with self.conn:
                self.cursor.execute(query, (new_permission, person_id))
                logger.info("✅ وضعیت مجوز بروزرسانی شد.")
        except Exception as e:
            self.conn.rollback()
            logger.error(f"❌ خطا در بروزرسانی مجوز: {e}")

    def delete_person(self, person_id):
        """حذف یک فرد از پایگاه داده.

        Args:
            person_id (int): شناسه فرد
        """
        query = "DELETE FROM persons WHERE id = %s"
        try:
            with self.conn:
                self.cursor.execute(query, (person_id,))
                logger.info("✅ شخص حذف شد.")
        except Exception as e:
            self.conn.rollback()
            logger.error(f"❌ خطا در حذف شخص: {e}")

    def insert_access_log(self, national_id, entry_date, entry_time, exit_time=None, confidence=None, status="unknown", camera_id=0):
        """ثبت لاگ دسترسی در جدول access_logs.

        Args:
            national_id (str): کد ملی فرد
            entry_date (date): تاریخ ورود
            entry_time (str): زمان ورود
            exit_time (str, optional): زمان خروج
            confidence (float, optional): درصد اطمینان
            status (str, optional): وضعیت (پیش‌فرض: unknown)
            camera_id (int, optional): شناسه دوربین (پیش‌فرض: 0)
        """
        if not self.validate_national_id(national_id):
            logger.error(f"❌ خطا: کد ملی {national_id} معتبر نیست.")
            return

        try:

            self.cursor.execute("SELECT id FROM persons WHERE national_id = %s", (national_id,))
            person_id = self.cursor.fetchone()
            if not person_id:
                logger.error(f"❌ فرد با کد ملی {national_id} یافت نشد.")
                return

            person_id = person_id[0]

            self.cursor.execute("SELECT id FROM cameras WHERE index = %s", (camera_id,))
            camera_result = self.cursor.fetchone()
            if not camera_result:
                logger.warning(f"⚠️ دوربین با index {camera_id} یافت نشد. دوربین پیش‌فرض استفاده می‌شود.")
                self.cursor.execute("INSERT INTO cameras (index, name) VALUES (%s, %s) ON CONFLICT (index) DO NOTHING", (camera_id, f"Camera {camera_id}"))
                self.conn.commit()
                self.cursor.execute("SELECT id FROM cameras WHERE index = %s", (camera_id,))
                camera_result = self.cursor.fetchone()

            camera_id = camera_result[0]
            query = """
            INSERT INTO access_logs (person_id, camera_id, entry_date, entry_time, exit_time, confidence, status)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            with self.conn:
                self.cursor.execute(query, (person_id, camera_id, entry_date, entry_time, exit_time, confidence, status))
                logger.info(f"✅ لاگ دسترسی برای کد ملی {national_id} ثبت شد.")
        except Exception as e:
            self.conn.rollback()
            logger.error(f"❌ خطا در ثبت لاگ دسترسی: {e}")

    def update_access_log(self, national_id, exit_time):
        """به‌روزرسانی زمان خروج در لاگ دسترسی.

        Args:
            national_id (str): کد ملی فرد
            exit_time (str): زمان خروج
        """
        if not self.validate_national_id(national_id):
            logger.error(f"❌ خطا: کد ملی {national_id} معتبر نیست.")
            return

        try:
            query = """
            UPDATE access_logs 
            SET exit_time = %s, status = 'completed'
            WHERE person_id = (SELECT id FROM persons WHERE national_id = %s)
            AND exit_time IS NULL
            """
            with self.conn:
                self.cursor.execute(query, (exit_time, national_id))
                if self.cursor.rowcount > 0:
                    logger.info(f"✅ زمان خروج برای کد ملی {national_id} به‌روزرسانی شد.")
                else:
                    logger.warning(f"⚠️ لاگ دسترسی برای کد ملی {national_id} یافت نشد.")
        except Exception as e:
            self.conn.rollback()
            logger.error(f"❌ خطا در به‌روزرسانی لاگ دسترسی: {e}")

    def delete_access_log(self, national_id):
        """حذف لاگ دسترسی بر اساس کد ملی.

        Args:
            national_id (str): کد ملی فرد
        """
        if not self.validate_national_id(national_id):
            logger.error(f"❌ خطا: کد ملی {national_id} معتبر نیست.")
            return

        try:
            query = """
            DELETE FROM access_logs 
            WHERE person_id = (SELECT id FROM persons WHERE national_id = %s)
            """
            with self.conn:
                self.cursor.execute(query, (national_id,))
                if self.cursor.rowcount > 0:
                    logger.info(f"✅ لاگ دسترسی برای کد ملی {national_id} حذف شد.")
                else:
                    logger.warning(f"⚠️ لاگ دسترسی برای کد ملی {national_id} یافت نشد.")
        except Exception as e:
            self.conn.rollback()
            logger.error(f"❌ خطا در حذف لاگ دسترسی: {e}")

    def close(self):
        """بستن اتصال به پایگاه داده."""
        try:
            self.cursor.close()
            self.conn.close()
            logger.info("✅ اتصال به دیتابیس بسته شد.")
        except Exception as e:
            logger.error(f"❌ خطا در بستن اتصال: {e}")

    @staticmethod
    def features_to_bytes(features):
        """تبدیل ویژگی‌های چهره به بایت.

        Args:
            features (np.ndarray): ویژگی‌های چهره

        Returns:
            bytes: داده‌های سریال‌شده
        """
        features = np.squeeze(features)
        if features.shape != (FEATURES_DIM,):
            raise ValueError(f"شکل ویژگی‌ها نامعتبر است: {features.shape}")
        return pickle.dumps(features.astype(np.float32))

    @staticmethod
    def bytes_to_features(byte_data):
        """تبدیل بایت به ویژگی‌های چهره.

        Args:
            byte_data (bytes): داده‌های سریال‌شده

        Returns:
            np.ndarray: ویژگی‌های چهره
        """
        features = pickle.loads(byte_data)
        features = np.squeeze(features)
        if features.shape != (FEATURES_DIM,):
            raise ValueError(f"شکل ویژگی‌های بازیابی‌شده نامعتبر است: {features.shape}")
        return features.astype(np.float32)
