import sys
import cv2
import tkinter as tk
from tkinter import messagebox
import numpy as np
import threading
import webbrowser
import pandas as pd
import psycopg2
import logging
from datetime import datetime
from PyQt5.QtCore import QTime, Qt, QDateTime, QThread, pyqtSignal, QDate
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMessageBox, QTableWidgetItem, QFileDialog, QInputDialog, QMainWindow, QApplication
from PyQt5.QtGui import QImage, QPixmap
from PyQt5 import QtCore
from utils.utils import generate_random_nid
from gui_main import Ui_MainWindow
from face_recognition_module import FaceRecognizer
from db_manager import DatabaseManager
from threads.yolo_thread import YoloCameraThread



logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CameraThread(QThread):
    frame_ready = pyqtSignal(np.ndarray)
    face_info_ready = pyqtSignal(object, object, object)  # person_info, face_img, features

    def __init__(self, face_system, video_source=0):
        super().__init__()
        self.face_system = face_system
        self.video_source = video_source
        self.running = True

    def run(self):
        try:
            cap = cv2.VideoCapture(self.video_source)
            if not cap.isOpened():
                logger.error(f"❌ نمی‌توان به منبع ویدئو {self.video_source} متصل شد.")
                return

            while self.running:
                ret, frame = cap.read()
                if not ret or frame is None:
                    logger.warning("⚠️ فریم ویدئو دریافت نشد.")
                    continue

                annotated_frame, face_img, matched_img, person_info, features = self.face_system.recognize_from_frame(frame)
                self.frame_ready.emit(annotated_frame)
                self.face_info_ready.emit(person_info, face_img, features)

            cap.release()
            cv2.destroyAllWindows()
            logger.info("✅ نخ دوربین متوقف شد.")
        except Exception as e:
            logger.error(f"❌ خطا در نخ دوربین: {e}")

    def stop(self):
        self.running = False
        self.wait()

class NewUserCameraThread(QThread):
    frame_ready = pyqtSignal(np.ndarray)
    capture_ready = pyqtSignal(np.ndarray)

    def __init__(self, video_source=0):
        super().__init__()
        self.video_source = video_source
        self.running = False
        self.capture_frame = None

    def run(self):
        try:
            cap = cv2.VideoCapture(self.video_source)
            if not cap.isOpened():
                logger.error(f"❌ نمی‌توان به منبع ویدئو {self.video_source} متصل شد.")
                return

            self.running = True
            while self.running:
                ret, frame = cap.read()
                if not ret or frame is None:
                    logger.warning("⚠️ فریم ویدئو دریافت نشد.")
                    continue
                self.frame_ready.emit(frame)
                if self.capture_frame is not None:
                    self.capture_ready.emit(frame.copy())
                    self.capture_frame = None

            cap.release()
            cv2.destroyAllWindows()
            logger.info("✅ نخ دوربین جدید متوقف شد.")
        except Exception as e:
            logger.error(f"❌ خطا در نخ دوربین جدید: {e}")

    def capture(self):
        self.capture_frame = True

    def stop(self):
        self.running = False
        self.wait()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.searchDateFrom.setDate(QDate(1900, 1, 1))  
        self.ui.searchDateTo.setDate(QDate(2100, 1, 1))    

        self.db = DatabaseManager(
            dbname="face_recognition_db",
            user="faceuser",
            password="123456"
        )

        detector_model_path = "/home/elnaz/soroush/Face_Plate/weight/face_detection_yunet_2023mar.onnx"
        recognizer_model_path = "/home/elnaz/soroush/Face_Plate/weight/model.onnx"

        try:
            self.face_system = FaceRecognizer(
                detector_model_path=detector_model_path,
                recognizer_model_path=recognizer_model_path,
                db_manager=self.db
            )
            self.face_system.load_features_from_db()
            logger.info(f"✅ تعداد ویژگی‌های بارگذاری‌شده: {len(self.face_system.features)}")
        except Exception as e:
            logger.error(f"❌ خطا در راه‌اندازی FaceRecognizer: {e}")
            self.show_message("خطا", "نمی‌توان سیستم شناسایی چهره را راه‌اندازی کرد.", QMessageBox.Critical)
            sys.exit(1)

        self.camera_thread = None
        self.new_user_camera_thread = None
        self.yolo_thread = None
        self.video_source = 0
        self.last_features = None
        self.last_face_img = None
        self.last_matched_img = None
        self.captured_image = None


        self.ui.name_input.setEnabled(True)
        self.ui.lname_input.setEnabled(True)
        self.ui.nid_input.setEnabled(True)
        logger.info(f"وضعیت فیلدها: name_input={self.ui.name_input.isEnabled()}, lname_input={self.ui.lname_input.isEnabled()}, nid_input={self.ui.nid_input.isEnabled()}")


        self.ui.name_input.setFocus()


        self.ui.btn_toggle_camera.clicked.connect(self.toggle_camera)
        self.ui.register_person_btn.clicked.connect(self.switch_to_newuser_tab)
        self.ui.delete_person_btn.clicked.connect(self.delete_person_from_db)
        self.ui.phone_btn.clicked.connect(self.call_phone)
        self.ui.whatsapp_btn.clicked.connect(self.call_whatsapp)
        self.ui.gmail_btn.clicked.connect(self.call_gmail)
        self.ui.savePermissionButton.clicked.connect(self.save_permission) 
        self.ui.searchButton.clicked.connect(self.search_permissions)
        self.ui.resetFormButton.clicked.connect(self.clear_permission_form)  
        self.ui.btnSearchReport.clicked.connect(self.load_report_data)
        self.ui.save_excel_btn.clicked.connect(self.save_report_to_excel)
        self.ui.btn_start_yolo.clicked.connect(self.start_yolo_camera)
        self.ui.btn_stop_yolo.clicked.connect(self.stop_yolo_camera)
        self.ui.start_camera_btn.clicked.connect(self.start_new_user_camera)
        self.ui.capture_btn.clicked.connect(self.capture_image)
        self.ui.save_new_user_btn.clicked.connect(self.save_new_user)
        self.ui.update_info_btn.clicked.connect(self.update_info)

    def reset_form(self):
        self.ui.name_input.clear()
        self.ui.lname_input.clear()
        self.ui.nid_input.clear()
        self.ui.name_input.setEnabled(True)
        self.ui.lname_input.setEnabled(True)
        self.ui.nid_input.setEnabled(True)
        self.ui.update_info_btn.setEnabled(False)
        logger.info("✅ فرم اطلاعات فرد بازنشانی شد.")


    def clear_permission_form(self):
        self.ui.searchBar.clear() 
        self.ui.searchDateFrom.setDate(QtCore.QDate.currentDate()) 
        self.ui.searchDateTo.setDate(QtCore.QDate.currentDate().addYears(1))  
        self.ui.searchStatus.setCurrentIndex(0)  
        self.ui.permissionTable.setRowCount(0) 
        print("فرم مجوز با موفقیت بازنشانی شد.")  


    def switch_to_newuser_tab(self):
        self.ui.tabWidget.setCurrentWidget(self.ui.newuser_tab)
        logger.info("✅ به تب ثبت کاربر جدید هدایت شد.")

    def toggle_camera(self):
        if not hasattr(self, 'camera_thread') or self.camera_thread is None or not self.camera_thread.isRunning():
            try:
                self.camera_thread = CameraThread(self.face_system, self.video_source)
                self.camera_thread.frame_ready.connect(self.update_frame)
                self.camera_thread.face_info_ready.connect(self.update_face_info)
                self.camera_thread.start()
                self.ui.btn_toggle_camera.setText("خاموش کردن دوربین")
                logger.info("✅ دوربین روشن شد.")
            except Exception as e:
                logger.error(f"❌ خطا در روشن کردن دوربین: {e}")
                self.show_message("خطا", "نمی‌توان دوربین را روشن کرد.", QMessageBox.Critical)
        else:
            self.camera_thread.stop()
            self.camera_thread = None
            self.ui.btn_toggle_camera.setText("روشن کردن دوربین")
            logger.info("✅ دوربین خاموش شد.")

    def start_new_user_camera(self):
        if not hasattr(self, 'new_user_camera_thread') or self.new_user_camera_thread is None or not self.new_user_camera_thread.isRunning():
            try:
                self.new_user_camera_thread = NewUserCameraThread(self.video_source)
                self.new_user_camera_thread.frame_ready.connect(self.update_new_user_frame)
                self.new_user_camera_thread.capture_ready.connect(self.update_captured_image)
                self.new_user_camera_thread.start()
                self.ui.start_camera_btn.setText("خاموش کردن دوربین")
                logger.info("✅ دوربین تب ثبت کاربر روشن شد.")
            except Exception as e:
                logger.error(f"❌ خطا در روشن کردن دوربین تب ثبت کاربر: {e}")
                self.show_message("خطا", "نمی‌توان دوربین را روشن کرد.", QMessageBox.Critical)
        else:
            self.new_user_camera_thread.stop()
            self.new_user_camera_thread = None
            self.ui.start_camera_btn.setText("روشن کردن دوربین")
            logger.info("✅ دوربین تب ثبت کاربر خاموش شد.")

    def capture_image(self):
        if self.new_user_camera_thread and self.new_user_camera_thread.isRunning():
            self.new_user_camera_thread.capture()
            logger.info("✅ درخواست عکس‌برداری ارسال شد.")

    def update_new_user_frame(self, frame):
        if frame is not None and frame.size > 0:
            self.set_label_image(self.ui.live_camera_label, frame)
        else:
            logger.warning("⚠️ فریم خالی دریافت شد.")

    def update_captured_image(self, frame):
        if frame is not None and frame.size > 0:
            self.captured_image = frame.copy()
            self.set_label_image(self.ui.captured_image_label, frame)
            logger.info("✅ تصویر گرفته‌شده به‌روزرسانی شد.")
        else:
            logger.error("❌ فریم گرفته‌شده خالی است.")

    def save_new_user(self):
        if self.captured_image is None or self.captured_image.size == 0:
            self.show_message("خطا", "هیچ تصویری برای ثبت وجود ندارد.", QMessageBox.Warning)
            return

        first_name = self.ui.first_name.text().strip()
        last_name = self.ui.last_name.text().strip()
        national_id = self.ui.national_id.text().strip()

        if not all([first_name, last_name, national_id]):
            self.show_message("خطا", "لطفاً همه فیلدها را وارد کنید.", QMessageBox.Warning)
            return

        if not self.db.validate_national_id(national_id):
            self.show_message("خطا", "کد ملی نامعتبر است. باید 10 رقم باشد.", QMessageBox.Warning)
            return

        try:
            success, buffer = cv2.imencode('.jpg', self.captured_image)
            if not success:
                raise Exception("ذخیره تصویر ناموفق بود.")
            image_bytes = buffer.tobytes()
            features = self.face_system.extract_features(self.captured_image)
            if features is not None:
                self.face_system.add_person_to_db(first_name, last_name, national_id, image_bytes, features)
                self.show_message("ثبت موفق", "فرد با موفقیت ثبت شد.", QMessageBox.Information)
                self.clear_new_user_form()
            else:
                logger.error("❌ ویژگی‌های چهره استخراج نشد.")
                self.show_message("خطا", "ویژگی‌های چهره استخراج نشد.", QMessageBox.Warning)
        except Exception as e:
            logger.error(f"❌ خطا در ثبت کاربر جدید: {e}")
            self.show_message("خطا", str(e), QMessageBox.Critical)

    def clear_new_user_form(self):
        self.ui.first_name.clear()
        self.ui.last_name.clear()
        self.ui.national_id.clear()
        self.ui.live_camera_label.clear()
        self.ui.captured_image_label.clear()
        self.captured_image = None
        logger.info("✅ فرم ثبت کاربر جدید بازنشانی شد.")

    def start_yolo_camera(self):
        if self.yolo_thread is None or not self.yolo_thread.isRunning():
            try:
                self.yolo_thread = YoloCameraThread(camera_index=0)
                self.yolo_thread.frame_updated.connect(self.update_yolo_frame)
                self.yolo_thread.start()
                logger.info("✅ دوربین YOLO روشن شد.")
            except Exception as e:
                logger.error(f"❌ خطا در روشن کردن دوربین YOLO: {e}")
                self.show_message("خطا", "نمی‌توان دوربین YOLO را روشن کرد.", QMessageBox.Critical)
        else:
            logger.warning("⚠️ دوربین YOLO قبلاً فعال است.")

    def stop_yolo_camera(self):
        if self.yolo_thread is not None and self.yolo_thread.isRunning():
            self.yolo_thread.stop()
            self.yolo_thread = None
            logger.info("✅ دوربین YOLO متوقف شد.")
        else:
            logger.warning("⚠️ دوربین YOLO فعال نیست.")

    def update_yolo_frame(self, frame, person_count):
        try:
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image).scaled(
                self.ui.label_yolo_camera.width(),
                self.ui.label_yolo_camera.height(),
                Qt.KeepAspectRatio
            )
            self.ui.label_yolo_camera.setPixmap(pixmap)
            self.ui.person_count_input.setText(str(person_count))
        except Exception as e:
            logger.error(f"❌ خطا در به‌روزرسانی فریم YOLO: {e}")

    def update_frame(self, frame):
        logging.info(f"📷 فریم دریافت شد: shape={frame.shape if frame is not None else 'None'}")
        self.set_label_image(self.ui.label_camera_face, frame)

    def set_label_image(self, label, frame):
        try:
            if frame is None or frame.size == 0:
                logging.warning("⚠️ فریم خالی است!")
                return
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image).scaled(label.width(), label.height(), Qt.KeepAspectRatio)
            label.setPixmap(pixmap)
        except Exception as e:
            logging.error(f"❌ خطا در تنظیم تصویر لیبل: {e}")

    def set_recognized_face(self, face_img):
        if face_img is None or (hasattr(face_img, 'size') and face_img.size == 0):
            self.ui.recognized_face_label.clear()
            return
        try:
            if isinstance(face_img, memoryview):
                face_img = np.frombuffer(face_img.tobytes(), dtype=np.uint8).reshape(self.last_face_img.shape) if self.last_face_img is not None else None
            if face_img is not None and face_img.size > 0:
                img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                h, w, ch = img_rgb.shape
                aspect_ratio = w / h
                target_width = self.ui.recognized_face_label.width()
                target_height = int(target_width / aspect_ratio)
                if target_height > self.ui.recognized_face_label.height():
                    target_height = self.ui.recognized_face_label.height()
                    target_width = int(target_height * aspect_ratio)
                qt_image = QImage(img_rgb.data, w, h, w * ch, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qt_image).scaled(
                    target_width, target_height, Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
                self.ui.recognized_face_label.setPixmap(pixmap)
        except Exception as e:
            logger.error(f"❌ خطا در تنظیم تصویر چهره شناسایی‌شده: {e}")

    def set_matched_face(self, matched_img):
        if matched_img is None:
            self.ui.matched_face_label.clear()
            logger.warning("⚠️ تصویر منطبق خالی یا نامعتبر است.")
            return
        try:
            if isinstance(matched_img, memoryview):
                img_bytes = matched_img.tobytes()
                matched_img = cv2.imdecode(np.frombuffer(img_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
            if matched_img is not None and matched_img.size > 0:
                img_rgb = cv2.cvtColor(matched_img, cv2.COLOR_BGR2RGB)
                h, w, ch = img_rgb.shape
                aspect_ratio = w / h
                target_width = self.ui.matched_face_label.width()
                target_height = int(target_width / aspect_ratio)
                if target_height > self.ui.matched_face_label.height():
                    target_height = self.ui.matched_face_label.height()
                    target_width = int(target_height * aspect_ratio)
                qt_image = QImage(img_rgb.data, w, h, w * ch, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qt_image).scaled(
                    target_width, target_height, Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
                self.ui.matched_face_label.setPixmap(pixmap)
                logger.info("✅ تصویر منطبق با موفقیت نمایش داده شد.")
        except Exception as e:
            logger.error(f"❌ خطا در تنظیم تصویر چهره منطبق: {e}")

    def show_person_info(self, person_info):
        try:
            self.ui.date_input.setDate(person_info.get("entry_date", QDateTime.currentDateTime().date()))
            self.ui.entry_time_input.setTime(QTime.fromString(person_info.get("entry_time", "00:00:00"), "HH:mm:ss"))
            self.ui.exit_time_input.setTime(QTime.fromString(person_info.get("exit_time", "00:00:00"), "HH:mm:ss"))
            self.ui.cam_input.setCurrentText(str(person_info.get("camera_number", "0")))
            confidence = person_info.get("confidence", 0)
            self.ui.confidence_input.setText(str(round(confidence, 2)))
            permission = person_info.get("permission", "false")
            self.ui.permission_state_input.setCurrentText("مجاز" if permission == "true" else "غیرمجاز")
            self.ui.update_info_btn.setEnabled(True)
            self.ui.nid_input.setText(person_info.get("nid", ""))
        except Exception as e:
            logger.error(f"❌ خطا در نمایش اطلاعات فرد: {e}")

    def update_info(self):
        try:
            national_id = self.ui.nid_input.text().strip()
            if not national_id:
                self.show_message("خطا", "لطفاً کد ملی را وارد کنید.", QMessageBox.Warning)
                return

            entry_date = self.ui.date_input.date().toPyDate()
            entry_time = self.ui.entry_time_input.time().toString("HH:mm:ss")
            exit_time = self.ui.exit_time_input.time().toString("HH:mm:ss")
            camera_number = int(self.ui.cam_input.currentText())
            confidence = float(self.ui.confidence_input.text()) if self.ui.confidence_input.text() else 0.0
            permission = "true" if self.ui.permission_state_input.currentText() == "مجاز" else "false"

            self.db.insert_access_log(national_id, entry_date, entry_time, exit_time if exit_time != "00:00:00" else None, confidence, "recognized" if permission == "true" else "unknown", camera_number)
            logger.info(f"✅ اطلاعات دسترسی برای کد ملی {national_id} در access_logs ذخیره شد.")
            self.show_message("موفق", "اطلاعات با موفقیت ذخیره شد.", QMessageBox.Information)
        except Exception as e:
            logger.error(f"❌ خطا در ذخیره اطلاعات دسترسی: {e}")
            self.show_message("خطا", str(e), QMessageBox.Critical)

    def delete_person_from_db(self):
        national_id = self.ui.nid_input.text().strip()
        if not national_id:
            self.show_message("خطا", "لطفاً کد ملی را وارد کنید.", QMessageBox.Warning)
            return

        try:
            person = self.db.get_person_by_national_id(national_id)
            if not person:
                self.show_message("خطا", "فرد با این کد ملی یافت نشد.", QMessageBox.Warning)
                return

            reply = QMessageBox.question(
                self, "تأیید", f"آیا مطمئن هستید که می‌خواهید فرد {person['first_name']} {person['last_name']} را حذف کنید؟",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self.db.delete_person(person['id'])
                self.face_system.load_features_from_db()
                logger.info(f"✅ فرد با کد ملی {national_id} حذف شد.")
                self.show_message("موفق", "فرد با موفقیت حذف شد.", QMessageBox.Information)
                self.clear_form()
        except Exception as e:
            logger.error(f"❌ خطا در حذف فرد: {e}")
            self.show_message("خطا", str(e), QMessageBox.Critical)

    def save_permission(self):
        full_name = self.ui.fullNameInput.text().strip()
        national_id = self.ui.nationalIdInput.text().strip()
        camera_id = int(self.ui.cameraSelection.currentText())
        start_date = self.ui.dateFromPicker.date().toString("yyyy-MM-dd")
        start_time = self.ui.startTimePicker.time().toString("HH:mm:ss")
        end_date = self.ui.dateToPicker.date().toString("yyyy-MM-dd")
        end_time = self.ui.endTimePicker.time().toString("HH:mm:ss")
        type_perm = self.ui.typeInput.currentText()  
        status_ui = self.ui.permissionStatus.currentText()

        status_mapping = {
            "فعال": "active",
            "غیرفعال": "inactive"
        }
        status = status_mapping.get(status_ui, "active") 

        if not national_id or not full_name:
            self.show_message("خطا", "نام و کد ملی را وارد کنید.", QMessageBox.Warning)
            return

        if not self.db.validate_national_id(national_id):
            self.show_message("خطا", "کد ملی نامعتبر است. باید 10 رقم باشد.", QMessageBox.Warning)
            return

        try:
            with psycopg2.connect(
                host="localhost",
                database="face_recognition_db",
                user="faceuser",
                password="123456"
            ) as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT id FROM persons WHERE national_id = %s", (national_id,))
                    person = cursor.fetchone()
                    if not person:
                        self.show_message("خطا", "فرد مورد نظر در پایگاه داده وجود ندارد.", QMessageBox.Warning)
                        return

                    person_id = person[0]
                    cursor.execute("SELECT id FROM cameras WHERE index = %s", (camera_id,))
                    camera_result = cursor.fetchone()
                    if not camera_result:
                        self.show_message("خطا", "دوربین انتخاب‌شده در دیتابیس وجود ندارد.", QMessageBox.Critical)
                        return

                    camera_id = camera_result[0]
                    start_datetime = f"{start_date} {start_time}"
                    end_datetime = f"{end_date} {end_time}"
                    insert_query = """
                        INSERT INTO permissions (person_id, camera_id, start_datetime, end_datetime, type, status)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        RETURNING id
                    """
                    cursor.execute(insert_query, (person_id, camera_id, start_datetime, end_datetime, type_perm, status))
                    permission_id = cursor.fetchone()[0]
                    conn.commit()
                    logger.info(f"✅ مجوز برای فرد با کد ملی {national_id} ثبت شد.")
                    self.show_message("موفق", f"مجوز با ID {permission_id} با موفقیت ثبت شد.", QMessageBox.Information)
                    self.clear_permission_form()
        except Exception as e:
            logger.error(f"❌ خطا در ذخیره مجوز: {e}")
            self.show_message("خطا", str(e), QMessageBox.Critical)




    def search_permissions(self):

        search_text = self.ui.searchBar.text().strip()
        start_date = self.ui.searchDateFrom.date().toString("yyyy-MM-dd") if self.ui.searchDateFrom.date() != QDate(1900, 1, 1) else None
        end_date = self.ui.searchDateTo.date().toString("yyyy-MM-dd") if self.ui.searchDateTo.date() != QDate(2100, 1, 1) else None
        status = self.ui.searchStatus.currentText()


        query = "SELECT p.id, pe.first_name, pe.last_name, pe.national_id, c.index, p.start_datetime, p.end_datetime, p.status " \
                "FROM permissions p " \
                "JOIN persons pe ON p.person_id = pe.id " \
                "JOIN cameras c ON p.camera_id = c.id " \
                "WHERE 1=1"
        params = []

        if search_text and search_text != "نام / کد ملی مورد نظر را وارد کنید":
            parts = search_text.split()
            if len(parts) > 1: 
                query += " AND (pe.first_name ILIKE %s OR pe.last_name ILIKE %s OR pe.national_id ILIKE %s)"
                params.extend([f"%{parts[0]}%", f"%{parts[1]}%", f"%{search_text}%"])
            else:  
                query += " AND (pe.first_name ILIKE %s OR pe.national_id ILIKE %s)"
                params.extend([f"%{search_text}%", f"%{search_text}%"])
            print(f"Searching for: {search_text}")  


        if start_date and end_date and self.ui.searchDateFrom.date() != QDate(1900, 1, 1) and self.ui.searchDateTo.date() != QDate(2100, 1, 1):
            query += " AND p.start_datetime BETWEEN %s AND %s"
            params.extend([f"{start_date} 00:00:00", f"{end_date} 23:59:59"])
        
        if status and status in ["فعال", "غیرفعال"]:
            status_db = "active" if status == "فعال" else "inactive"
            query += " AND p.status = %s"
            params.append(status_db)

        query += " ORDER BY p.start_datetime DESC"
        print(f"Executing query: {query} with params: {params}")  

        try:
            with psycopg2.connect(
                host="localhost",
                database="face_recognition_db",
                user="faceuser",
                password="123456"
            ) as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query, tuple(params))
                    results = cursor.fetchall()


                    self.ui.permissionTable.setRowCount(0)


                    if not results:
                        QMessageBox.warning(self, "هشدار", "هیچ نتیجه‌ای یافت نشد.", QMessageBox.Ok)
                        return


                    for row_num, row in enumerate(results):
                        self.ui.permissionTable.insertRow(row_num)
                        for col_num, data in enumerate(row):
                            item = QTableWidgetItem(str(data))
                            item.setTextAlignment(Qt.AlignCenter)
                            self.ui.permissionTable.setItem(row_num, col_num, item)


                    self.ui.permissionTable.setColumnCount(len(results[0]))
                    headers = ["آیدی", "نام", "نام خانوادگی", "کد ملی", "شماره دوربین", "تاریخ ایجاد", "تا زمان", "وضعیت"]
                    for col, header in enumerate(headers[:len(results[0])]):
                        item = self.ui.permissionTable.horizontalHeaderItem(col)
                        if item:
                            item.setText(header)
                        else:
                            self.ui.permissionTable.setHorizontalHeaderItem(col, QTableWidgetItem(header))


                    self.ui.permissionTable.resizeColumnsToContents()
                    self.ui.permissionTable.setMaximumWidth(800)
                    header = self.ui.permissionTable.horizontalHeader()
                    header.setSectionResizeMode(QtWidgets.QHeaderView.Stretch)

        except Exception as e:
            QMessageBox.critical(self, "خطا", f"خطا در جستجو: {str(e)}", QMessageBox.Ok)


    def clear_form(self):
        self.ui.date_input.setDate(QDateTime.currentDateTime().date())
        self.ui.entry_time_input.setTime(QTime.fromString("00:00:00", "HH:mm:ss"))
        self.ui.exit_time_input.setTime(QTime.fromString("00:00:00", "HH:mm:ss"))
        self.ui.confidence_input.clear()
        self.ui.permission_state_input.setCurrentIndex(0)
        self.ui.recognized_face_label.clear()
        self.ui.matched_face_label.clear()
        self.ui.unknown_label.clear()
        logger.info("✅ فرم اطلاعات فرد بازنشانی شد.")


    
    def load_report_data(self):

        name = self.ui.lineEdit_name_report.text().strip()
        national_id = self.ui.lineEdit_nationalid_report.text().strip()
        from_date = self.ui.dateEdit_from_date.date().toString("yyyy-MM-dd") if self.ui.dateEdit_from_date.date() != QDate(1900, 1, 1) else None
        to_date = self.ui.dateEdit_to_date.date().toString("yyyy-MM-dd") if self.ui.dateEdit_to_date.date() != QDate(2100, 1, 1) else None
        from_hour = self.ui.timeEdit_from_hour.time().toString("HH:mm:ss") if self.ui.timeEdit_from_hour.time() != QTime(0, 0) else None
        to_hour = self.ui.timeEdit_to_hour.time().toString("HH:mm:ss") if self.ui.timeEdit_to_hour.time() != QTime(23, 59) else None


        if national_id and not self.db.validate_national_id(national_id):
            self.show_message("خطا", "کد ملی نامعتبر است. باید 10 رقم باشد.", QMessageBox.Warning)
            return


        if from_date and to_date:
            from_date_dt = datetime.strptime(from_date, "%Y-%m-%d")
            to_date_dt = datetime.strptime(to_date, "%Y-%m-%d")
            if from_date_dt > to_date_dt:
                from_date, to_date = to_date, from_date  


        query = """
            SELECT 
                pe.first_name,
                pe.last_name,
                pe.national_id,
                al.entry_time AS entry_time,
                al.exit_time,
                CASE 
                    WHEN al.exit_time IS NOT NULL THEN 
                        TO_CHAR(al.exit_time - al.entry_time, 'HH24:MI:SS') 
                    ELSE 
                        'در حال حضور'
                END AS duration,
                COALESCE(c.index::text, '0') AS camera_id,
                al.status,
                al.entry_date AS entry_date
            FROM access_logs al
            LEFT JOIN persons pe ON al.person_id = pe.id
            LEFT JOIN cameras c ON al.camera_id = c.id
            WHERE 1=1
        """
        params = []

        if name:
            query += " AND (pe.first_name || ' ' || pe.last_name) ILIKE %s"
            params.append(f"%{name}%")
        if national_id:
            query += " AND pe.national_id ILIKE %s"
            params.append(f"%{national_id}%")
        if from_date and to_date:
            query += " AND al.entry_date BETWEEN %s AND %s"
            params.extend([from_date, to_date])
        if from_hour and to_hour:
            query += " AND al.entry_time BETWEEN %s AND %s"
            params.extend([from_hour, to_hour])

        query += " ORDER BY al.entry_date DESC, al.entry_time DESC"
        print(f"Executing query: {query} with params: {params}")

        try:
            with psycopg2.connect(
                host="localhost",
                database="face_recognition_db",
                user="faceuser",
                password="123456"
            ) as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query, tuple(params))
                    results = cursor.fetchall()
                    self.current_report_results = results


                    self.ui.report_table.setRowCount(0)


                    if not results:
                        QMessageBox.warning(self, "هشدار", "هیچ نتیجه‌ای یافت نشد.", QMessageBox.Ok)
                        return


                    for row_num, row in enumerate(results):
                        self.ui.report_table.insertRow(row_num)
                        for col_num, data in enumerate(row):
                            item = QTableWidgetItem(str(data) if data else "-")
                            item.setTextAlignment(Qt.AlignCenter)
                            item.setFlags(item.flags() | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                            item.setForeground(Qt.black)
                            self.ui.report_table.setItem(row_num, col_num, item)


                    self.ui.report_table.setColumnCount(len(results[0]))
                    headers = ["نام", "نام خانوادگی", "کد ملی", "ساعت ورود", "ساعت خروج", "مدت زمان حضور", "مکان حضور", "وضعیت", "تاریخ ورود"]
                    for col, header in enumerate(headers):
                        item = self.ui.report_table.horizontalHeaderItem(col)
                        if item:
                            item.setText(header)
                        else:
                            self.ui.report_table.setHorizontalHeaderItem(col, QTableWidgetItem(header))


                    header = self.ui.report_table.horizontalHeader()
                    header.setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
                    self.ui.report_table.setMaximumWidth(1400)


                    font = self.ui.report_table.font()
                    font.setPointSize(10)
                    self.ui.report_table.setFont(font)

        except Exception as e:
            logger.error(f"❌ خطا در بارگذاری گزارش: {e}")
            self.show_message("خطا", f"در هنگام بارگذاری گزارش خطایی رخ داد:\n{str(e)}", QMessageBox.Critical)



    def save_report_to_excel(self):
        try:
            data = getattr(self, 'current_report_results', None)
            if not data:
                self.show_message("اخطار", "داده‌ای برای ذخیره وجود ندارد.", QMessageBox.Warning)
                return

            df = pd.DataFrame(data, columns=["نام", "نام خانوادگی", "کد ملی", "ساعت ورود", "ساعت خروج", "مدت زمان حضور", "مکان حضور", "وضعیت", "تاریخ ورود"])
            options = QFileDialog.Options()
            file_path, _ = QFileDialog.getSaveFileName(self, "ذخیره گزارش به Excel", "", "Excel Files (*.xlsx);;All Files (*)", options=options)
            if file_path:
                if not file_path.endswith('.xlsx'):
                    file_path += '.xlsx'
                df.to_excel(file_path, index=False)
                self.show_message("موفقیت", f"گزارش با موفقیت ذخیره شد:\n{file_path}", QMessageBox.Information)
                logger.info(f"✅ گزارش در {file_path} ذخیره شد.")
        except Exception as e:
            logger.error(f"❌ خطا در ذخیره گزارش به Excel: {e}")
            self.show_message("خطا", f"ذخیره گزارش به Excel انجام نشد:\n{str(e)}", QMessageBox.Critical)

    def call_phone(self):
        self.show_message("تماس", "تماس با پشتیبانی: 0912XXXXXXX", QMessageBox.Information)

    def call_whatsapp(self):
        webbrowser.open("https://wa.me/98912XXXXXXX")
        logger.info("✅ WhatsApp باز شد.")

    def call_gmail(self):
        webbrowser.open("https://mail.google.com")
        logger.info("✅ Gmail باز شد.")

    def show_message(self, title, message, icon):
        msg_box = QMessageBox()
        msg_box.setWindowTitle(title)
        msg_box.setText(message)
        msg_box.setIcon(icon)
        msg_box.exec_()

    def update_face_info(self, person_info, face_img, features):
        try:
            if face_img is not None:
                if isinstance(face_img, memoryview):
                    face_img = np.frombuffer(face_img.tobytes(), dtype=np.uint8).reshape(self.last_face_img.shape) if self.last_face_img is not None else None
                self.set_recognized_face(face_img)
                self.last_face_img = face_img
            else:
                self.ui.recognized_face_label.clear()

            self.last_features = features
            self.ui.date_input.setDate(QDateTime.currentDateTime().date())
            self.ui.entry_time_input.setTime(QTime.currentTime())
            self.ui.exit_time_input.setTime(QTime.currentTime())
            self.ui.cam_input.setCurrentText(str(self.video_source))

            self.ui.name_input.setEnabled(True)
            self.ui.lname_input.setEnabled(True)
            self.ui.nid_input.setEnabled(True)

            if person_info is None:
                self.ui.name_input.setText("ناشناس")
                self.ui.lname_input.setText("ناشناس")
                self.ui.nid_input.setText(generate_random_nid())
                self.ui.confidence_input.clear()
                self.ui.permission_state_input.setCurrentText("ناشناس")
                self.ui.permission_state_input.setStyleSheet("color: red;")
                self.ui.unknown_label.setText("ناشناس")
                self.ui.matched_face_label.clear()
                self.last_matched_img = None
                logger.info("⚠️ چهره‌ای تطبیق داده نشد.")
            else:
                person = self.db.get_person_by_national_id(person_info["nid"])
                if person:
                    self.ui.name_input.setText(person["first_name"])
                    self.ui.lname_input.setText(person["last_name"])
                    self.ui.nid_input.setText(person["national_id"])
                    self.show_person_info(person_info)
                    self.ui.permission_state_input.setStyleSheet(
                        "color: green;" if person_info.get("permission") == "true" else "color: red;"
                    )
                    self.ui.unknown_label.clear()
                    self.last_matched_img = self.db.get_person_image(person_info["nid"])
                    if self.last_matched_img is not None:
                        logger.info(f"✅ تصویر منطبق برای کد ملی {person_info['nid']} دریافت شد: type={type(self.last_matched_img)}")
                        self.set_matched_face(self.last_matched_img)
                    else:
                        self.ui.matched_face_label.clear()
                        logger.warning(f"⚠️ تصویر منطبق برای کد ملی {person_info['nid']} یافت نشد.")
                    logger.info(f"✅ چهره تطبیق داده شد: {person['first_name']} {person['last_name']}")

        except Exception as e:
            logging.error(f"❌ خطا در به‌روزرسانی اطلاعات چهره: {e}")

    def closeEvent(self, event):
        if hasattr(self, 'camera_thread') and self.camera_thread and self.camera_thread.isRunning():
            self.camera_thread.stop()
        if hasattr(self, 'new_user_camera_thread') and self.new_user_camera_thread and self.new_user_camera_thread.isRunning():
            self.new_user_camera_thread.stop()
        if hasattr(self, 'yolo_thread') and self.yolo_thread and self.yolo_thread.isRunning():
            self.yolo_thread.stop()
        self.db.close()
        event.accept()
        logger.info("✅ برنامه بسته شد.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())