import psycopg2
from datetime import datetime, timedelta
import random

def insert_fake_data():
    try:
        conn = psycopg2.connect(
            host="localhost",
            database="face_recognition_db",
            user="faceuser",
            password="123456"
        )
        cursor = conn.cursor()

        # افزودن افراد به persons
        persons = [
            ("علی", "رضایی", "1234567890"),
            ("مینا", "کاظمی", "2345678901"),
            ("سارا", "جعفری", "3456789012")
        ]

        for first, last, nid in persons:
            cursor.execute("""
            INSERT INTO persons (first_name, last_name, national_id)
            VALUES (%s, %s, %s)
            RETURNING id
            """, (first, last, nid))
            person_id = cursor.fetchone()[0]

            # برای هر نفر چند لاگ ورود/خروج بساز
            for i in range(3):
                entry_time = datetime.now() - timedelta(days=random.randint(1,5), hours=random.randint(1,5))
                exit_time = entry_time + timedelta(hours=random.randint(1,3))
                duration = exit_time - entry_time
                status = random.choice(["مجاز", "غیرمجاز"])

                cursor.execute("""
                INSERT INTO access_logs (person_id, entry_time, exit_time, duration, status)
                VALUES (%s, %s, %s, %s, %s)
                """, (person_id, entry_time, exit_time, duration, status))

        conn.commit()
        cursor.close()
        conn.close()
        print("✅ داده‌ی تستی با موفقیت وارد شد.")

    except Exception as e:
        print(f"❌ خطا در درج داده‌ی تستی: {e}")


if __name__ == "__main__":
    insert_fake_data()
