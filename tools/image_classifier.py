import sys
import os
import shutil
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QPushButton,
    QVBoxLayout,
    QLabel,
    QFileDialog,
    QHBoxLayout,
    QShortcut,
    QListWidget,
    QListWidgetItem,
    QScrollArea,
    QInputDialog,
    QMessageBox,
    QLineEdit,
    QGroupBox,
)
from PyQt5.QtGui import QPixmap, QWheelEvent, QImage, QKeySequence
from PyQt5.QtCore import Qt
import cv2
import numpy as np
import re


def natural_sort_key(s):
    """숫자가 포함된 문자열의 자연스러운 정렬을 위한 키 함수"""
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split("([0-9]+)", s)
    ]


class ImageViewer(QWidget):
    def __init__(self):
        super(ImageViewer, self).__init__()
        self.image_list = []
        self.current_index = 0
        self.scale_factor = 1.0
        self.original_pixmap = None

        # 이미지 패닝(이동) 관련 변수들
        self.panning = False
        self.pan_start_pos = None

        # 실행 취소 관련 변수들 (이미지 이동 취소용)
        self.undo_stack = []  # [(원본 경로, 이동된 경로, 클래스)]
        self.max_undo_history = 50

        # 폴더 경로 설정
        self.image_folder = None
        self.output_folder = None  # 출력 폴더 (0~10 클래스별 폴더가 생성될 위치)

        self.initUI()

    def initUI(self):
        self.setWindowTitle("Crop Image Classification Tool")
        self.resize(1400, 800)
        self.setMinimumSize(800, 600)

        # 메인 레이아웃
        main_layout = QHBoxLayout()

        # 왼쪽 패널
        left_panel = QVBoxLayout()

        # 폴더 선택 버튼
        folder_layout = QHBoxLayout()
        self.select_image_folder_button = QPushButton("Select Image Folder", self)
        self.select_image_folder_button.clicked.connect(self.select_image_folder)
        folder_layout.addWidget(self.select_image_folder_button)

        self.select_output_folder_button = QPushButton("Select Output Folder", self)
        self.select_output_folder_button.clicked.connect(self.select_output_folder)
        folder_layout.addWidget(self.select_output_folder_button)

        # 이미지 표시 영역
        self.scroll_area = QScrollArea()
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMouseTracking(True)
        self.scroll_area.setWidget(self.image_label)
        self.scroll_area.setWidgetResizable(True)
        self.setMouseTracking(True)

        # 파일명 표시
        self.filename_label = QLabel()
        self.filename_label.setAlignment(Qt.AlignCenter)

        # 현재 클래스 표시
        self.current_class_label = QLabel("Class: None")
        self.current_class_label.setAlignment(Qt.AlignCenter)
        self.current_class_label.setStyleSheet(
            "font-size: 20px; font-weight: bold; padding: 10px;"
        )

        # 버튼 그룹
        button_layout = QHBoxLayout()
        self.previous_button = QPushButton("Previous (←)")
        self.next_button = QPushButton("Next (→)")
        self.previous_button.clicked.connect(self.show_previous_image)
        self.next_button.clicked.connect(self.show_next_image)
        button_layout.addWidget(self.previous_button)
        button_layout.addWidget(self.next_button)

        # 왼쪽 패널에 위젯 추가
        left_panel.addLayout(folder_layout)
        left_panel.addWidget(self.scroll_area)
        left_panel.addWidget(self.filename_label)
        left_panel.addWidget(self.current_class_label)
        left_panel.addLayout(button_layout)

        # 오른쪽 패널
        right_panel = QVBoxLayout()

        # 클래스 선택 패널
        class_group = QGroupBox("Class Selection (0-10)")
        class_layout = QVBoxLayout()

        # 클래스 버튼들 (0~10)
        self.class_buttons = []
        for i in range(11):
            btn = QPushButton(f"Class {i}", self)
            btn.setMinimumHeight(40)
            btn.clicked.connect(lambda checked, cls=i: self.move_image_to_class(cls))
            class_layout.addWidget(btn)
            self.class_buttons.append(btn)

        class_group.setLayout(class_layout)
        right_panel.addWidget(class_group)

        # 현재 위치 정보
        position_layout = QHBoxLayout()
        position_layout.addWidget(QLabel("Position:"))
        self.position_info_label = QLineEdit("0/0")
        self.position_info_label.setReadOnly(False)
        self.position_info_label.setMaximumWidth(200)
        self.position_info_label.returnPressed.connect(self.jump_to_image)
        position_layout.addWidget(self.position_info_label)
        position_layout.addStretch()
        right_panel.addLayout(position_layout)

        # 이미지 목록
        image_list_label = QLabel("Image List:")
        right_panel.addWidget(image_list_label)
        self.image_list_widget = QListWidget()
        self.image_list_widget.itemClicked.connect(self.on_image_list_item_clicked)
        right_panel.addWidget(self.image_list_widget)

        # 메인 레이아웃에 패널 추가
        main_layout.addLayout(left_panel, 3)
        main_layout.addLayout(right_panel, 1)

        self.setLayout(main_layout)

        # 단축키 설정
        self.setup_shortcuts()

    def setup_shortcuts(self):
        """단축키 설정"""
        # 이미지 네비게이션
        left_shortcut = QShortcut(Qt.Key_Left, self)
        left_shortcut.activated.connect(self.show_previous_image)
        right_shortcut = QShortcut(Qt.Key_Right, self)
        right_shortcut.activated.connect(self.show_next_image)

        comma_shortcut = QShortcut(Qt.Key_Comma, self)
        comma_shortcut.activated.connect(self.show_previous_image)
        backtick_shortcut = QShortcut(Qt.Key_QuoteLeft, self)
        backtick_shortcut.activated.connect(self.show_next_image)

        # 삭제 (Del 키)
        delete_shortcut = QShortcut(Qt.Key_Delete, self)
        delete_shortcut.activated.connect(self.delete_current_image)

        # 확대/축소
        plus_shortcut = QShortcut(Qt.Key_Plus, self)
        plus_shortcut.activated.connect(lambda: self.zoom_image(1.25))
        minus_shortcut = QShortcut(Qt.Key_Minus, self)
        minus_shortcut.activated.connect(lambda: self.zoom_image(0.8))

        # ESC로 fit to window
        esc_shortcut = QShortcut(QKeySequence("Escape"), self)
        esc_shortcut.activated.connect(self.fit_to_window)

        # 클래스 선택 단축키 (0~9는 숫자 키, 10은 . 키)
        for i in range(10):
            key = getattr(Qt, f"Key_{i}")
            shortcut = QShortcut(key, self)
            shortcut.activated.connect(
                lambda checked=False, cls=i: self.move_image_to_class(cls)
            )

        # 클래스 10은 . 키
        period_shortcut = QShortcut(Qt.Key_Period, self)
        period_shortcut.activated.connect(lambda: self.move_image_to_class(10))

        # Ctrl+Z로 이동 취소
        undo_shortcut = QShortcut(QKeySequence("Ctrl+Z"), self)
        undo_shortcut.activated.connect(self.undo_move)

    def select_image_folder(self):
        """이미지 폴더 선택"""
        folder_path = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if folder_path:
            self.image_folder = folder_path
            self.load_images()

    def select_output_folder(self):
        """출력 폴더 선택"""
        folder_path = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder_path:
            self.output_folder = folder_path
            # 클래스별 폴더 생성 (0~10)
            for i in range(11):
                class_folder = os.path.join(folder_path, str(i))
                os.makedirs(class_folder, exist_ok=True)
            QMessageBox.information(
                self,
                "Output Folder Set",
                f"Output folder set to: {folder_path}\nClass folders (0-10) created.",
            )

    def load_images(self):
        """이미지 목록 로드"""
        if not self.image_folder:
            return

        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
        self.image_list = [
            os.path.join(self.image_folder, file)
            for file in os.listdir(self.image_folder)
            if os.path.splitext(file.lower())[1] in image_extensions
        ]

        # 자연스러운 정렬
        self.image_list.sort(key=lambda x: natural_sort_key(os.path.basename(x)))
        self.current_index = 0

        if self.image_list:
            self.show_image()
            self.update_image_list_widget()
        else:
            QMessageBox.warning(
                self, "No Images", "No images found in the selected folder."
            )

    def show_image(self):
        """현재 이미지 표시"""
        if not self.image_list:
            return

        if self.current_index < 0 or self.current_index >= len(self.image_list):
            return

        image_path = self.image_list[self.current_index]

        try:
            # 이미지 로드 (한글 경로 지원)
            img_array = np.fromfile(image_path, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img is None:
                print(f"Warning: Failed to load image: {image_path}")
                return

            # OpenCV 이미지를 QImage로 변환
            height, width, channel = img.shape
            bytes_per_line = 3 * width
            q_img = QImage(
                img.data, width, height, bytes_per_line, QImage.Format_RGB888
            ).rgbSwapped()

            # QImage를 QPixmap으로 변환
            self.original_pixmap = QPixmap.fromImage(q_img)
            self.update_image_display()

            # 파일명 표시
            filename = os.path.basename(image_path)
            self.filename_label.setText(filename)

            # 위치 정보 업데이트
            self.update_position_info()

        except Exception as e:
            print(f"Error in show_image: {str(e)}")

    def update_image_display(self):
        """이미지 표시 업데이트"""
        if self.original_pixmap:
            pixmap = self.original_pixmap.copy()

            # 스케일 적용
            scaled_pixmap = pixmap.scaled(
                pixmap.size() * self.scale_factor,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
            self.image_label.setPixmap(scaled_pixmap)

    def zoom_image(self, factor):
        """이미지 확대/축소"""
        self.scale_factor *= factor
        self.update_image_display()

    def fit_to_window(self):
        """이미지를 창 크기에 맞게 조정"""
        if not self.original_pixmap:
            return

        scroll_width = self.scroll_area.viewport().width()
        scroll_height = self.scroll_area.viewport().height()

        img_width = self.original_pixmap.width()
        img_height = self.original_pixmap.height()

        scale_x = scroll_width / img_width if img_width > 0 else 1.0
        scale_y = scroll_height / img_height if img_height > 0 else 1.0

        self.scale_factor = min(scale_x, scale_y)

        if self.scale_factor < 0.1:
            self.scale_factor = 0.1

        self.update_image_display()

    def move_image_to_class(self, class_id):
        """이미지를 클래스별 폴더로 이동"""
        if not self.image_list or not self.output_folder:
            QMessageBox.warning(
                self,
                "No Output Folder",
                "Please select output folder first.",
            )
            return

        if self.current_index < 0 or self.current_index >= len(self.image_list):
            return

        image_path = self.image_list[self.current_index]

        if not os.path.exists(image_path):
            QMessageBox.warning(self, "Error", "Image file does not exist.")
            return

        try:
            # 클래스 폴더 경로
            class_folder = os.path.join(self.output_folder, str(class_id))
            os.makedirs(class_folder, exist_ok=True)

            # 대상 경로
            filename = os.path.basename(image_path)
            dst_path = os.path.join(class_folder, filename)

            # 이미 존재하는 경우 처리
            if os.path.exists(dst_path):
                reply = QMessageBox.question(
                    self,
                    "File Exists",
                    f"File already exists in class {class_id} folder.\nOverwrite?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No,
                )
                if reply == QMessageBox.No:
                    return
                os.remove(dst_path)

            # 실행 취소를 위한 상태 저장
            self.undo_stack.append((image_path, dst_path, class_id))
            if len(self.undo_stack) > self.max_undo_history:
                self.undo_stack.pop(0)

            # 파일 이동
            shutil.move(image_path, dst_path)

            # 이미지 목록에서 제거
            self.image_list.pop(self.current_index)

            # 인덱스 조정
            if self.current_index >= len(self.image_list):
                self.current_index = len(self.image_list) - 1
            if self.current_index < 0 and len(self.image_list) > 0:
                self.current_index = 0

            # 현재 클래스 표시 업데이트
            self.current_class_label.setText(f"Class: {class_id} (Moved)")
            self.current_class_label.setStyleSheet(
                "font-size: 20px; font-weight: bold; padding: 10px; background-color: lightgreen;"
            )

            # 이미지 목록 업데이트
            self.update_image_list_widget()

            # 다음 이미지 표시
            if len(self.image_list) > 0:
                self.show_image()
            else:
                self.image_label.clear()
                self.filename_label.setText("No more images")
                QMessageBox.information(
                    self, "Complete", "All images have been classified."
                )

            print(f"Moved {filename} to class {class_id} folder")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to move image: {str(e)}")

    def undo_move(self):
        """마지막 이동 취소"""
        if not self.undo_stack:
            QMessageBox.information(self, "No Undo", "No move to undo.")
            return

        # 마지막 이동 정보 가져오기
        original_path, moved_path, class_id = self.undo_stack.pop()

        if not os.path.exists(moved_path):
            QMessageBox.warning(self, "Error", "Moved file does not exist anymore.")
            return

        try:
            # 파일을 원래 위치로 이동
            shutil.move(moved_path, original_path)

            # 이미지 목록에 다시 추가하고 정렬
            self.image_list.append(original_path)
            self.image_list.sort(key=lambda x: natural_sort_key(os.path.basename(x)))

            # 현재 인덱스를 이동된 이미지로 설정
            self.current_index = self.image_list.index(original_path)

            # 현재 클래스 표시 초기화
            self.current_class_label.setText("Class: None")
            self.current_class_label.setStyleSheet(
                "font-size: 20px; font-weight: bold; padding: 10px;"
            )

            # 이미지 목록 업데이트
            self.update_image_list_widget()

            # 이미지 표시
            self.show_image()

            print(
                f"Undo: Moved {os.path.basename(original_path)} back to original location"
            )

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to undo move: {str(e)}")

    def delete_current_image(self):
        """현재 이미지를 삭제 (Del 키) - 확인 없이 바로 삭제"""
        if not self.image_list:
            return

        if self.current_index < 0 or self.current_index >= len(self.image_list):
            return

        image_path = self.image_list[self.current_index]
        filename = os.path.basename(image_path)

        try:
            if os.path.exists(image_path):
                os.remove(image_path)

            # 목록에서 제거
            self.image_list.pop(self.current_index)

            # 인덱스 조정
            if self.current_index >= len(self.image_list):
                self.current_index = len(self.image_list) - 1
            if self.current_index < 0 and len(self.image_list) > 0:
                self.current_index = 0

            # 표시 초기화
            self.current_class_label.setText("Class: Deleted")
            self.current_class_label.setStyleSheet(
                "font-size: 20px; font-weight: bold; padding: 10px; background-color: lightcoral;"
            )

            # UI 업데이트
            self.update_image_list_widget()
            if self.image_list:
                self.show_image()
            else:
                self.image_label.clear()
                self.filename_label.setText("No more images")
            self.update_position_info()

            print(f"Deleted image: {filename}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to delete image: {str(e)}")

    def show_previous_image(self):
        """이전 이미지 표시"""
        if self.image_list and self.current_index > 0:
            self.current_index -= 1
            self.current_class_label.setText("Class: None")
            self.current_class_label.setStyleSheet(
                "font-size: 20px; font-weight: bold; padding: 10px;"
            )
            self.show_image()
            self.update_image_list_widget()

    def show_next_image(self):
        """다음 이미지 표시"""
        if self.image_list and self.current_index < len(self.image_list) - 1:
            self.current_index += 1
            self.current_class_label.setText("Class: None")
            self.current_class_label.setStyleSheet(
                "font-size: 20px; font-weight: bold; padding: 10px;"
            )
            self.show_image()
            self.update_image_list_widget()

    def update_image_list_widget(self):
        """이미지 목록 위젯 업데이트"""
        self.image_list_widget.clear()
        for path in self.image_list:
            item = QListWidgetItem(os.path.basename(path))
            self.image_list_widget.addItem(item)

        if self.image_list:
            self.image_list_widget.setCurrentRow(self.current_index)

    def on_image_list_item_clicked(self, item):
        """이미지 목록 아이템 클릭"""
        index = self.image_list_widget.row(item)
        if 0 <= index < len(self.image_list):
            self.current_index = index
            self.current_class_label.setText("Class: None")
            self.current_class_label.setStyleSheet(
                "font-size: 20px; font-weight: bold; padding: 10px;"
            )
            self.show_image()

    def update_position_info(self):
        """위치 정보 업데이트"""
        if self.image_list:
            current_pos = self.current_index + 1
            total_images = len(self.image_list)
            self.position_info_label.setText(f"{current_pos}/{total_images}")
        else:
            self.position_info_label.setText("0/0")

    def jump_to_image(self):
        """입력된 이미지 번호로 이동"""
        if not self.image_list:
            return

        text = self.position_info_label.text()
        try:
            if "/" in text:
                target_pos = int(text.split("/")[0].strip())
            else:
                target_pos = int(text.strip())

            target_index = target_pos - 1

            if 0 <= target_index < len(self.image_list):
                self.current_index = target_index
                self.current_class_label.setText("Class: None")
                self.current_class_label.setStyleSheet(
                    "font-size: 20px; font-weight: bold; padding: 10px;"
                )
                self.show_image()
                self.update_image_list_widget()
                self.update_position_info()
            else:
                QMessageBox.warning(
                    self,
                    "Invalid Index",
                    f"Image number must be between 1 and {len(self.image_list)}.",
                )
                self.update_position_info()
        except ValueError:
            QMessageBox.warning(
                self, "Invalid Input", "Please enter a valid number. (e.g., 1 or 1/100)"
            )
            self.update_position_info()

    def wheelEvent(self, event: QWheelEvent):
        """마우스 휠 이벤트"""
        if event.modifiers() == Qt.ControlModifier:
            if event.angleDelta().y() > 0:
                self.zoom_image(1.25)
            else:
                self.zoom_image(0.8)
        else:
            if event.angleDelta().y() > 0:
                self.scroll_area.verticalScrollBar().setValue(
                    self.scroll_area.verticalScrollBar().value() - 20
                )
            else:
                self.scroll_area.verticalScrollBar().setValue(
                    self.scroll_area.verticalScrollBar().value() + 20
                )

    def mousePressEvent(self, event):
        """마우스 클릭 이벤트 (패닝용)"""
        if event.button() == Qt.MiddleButton:
            self.panning = True
            self.pan_start_pos = event.pos()
            self.image_label.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, event):
        """마우스 이동 이벤트 (패닝용)"""
        if self.panning and self.pan_start_pos:
            current_pos = event.pos()
            dx = current_pos.x() - self.pan_start_pos.x()
            dy = current_pos.y() - self.pan_start_pos.y()

            h_scroll = self.scroll_area.horizontalScrollBar()
            v_scroll = self.scroll_area.verticalScrollBar()
            h_scroll.setValue(h_scroll.value() - dx)
            v_scroll.setValue(v_scroll.value() - dy)

            self.pan_start_pos = current_pos

    def mouseReleaseEvent(self, event):
        """마우스 릴리스 이벤트"""
        if event.button() == Qt.MiddleButton:
            self.panning = False
            self.pan_start_pos = None
            self.image_label.setCursor(Qt.ArrowCursor)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = ImageViewer()
    viewer.show()
    sys.exit(app.exec_())
