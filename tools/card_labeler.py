import sys
import os
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
    QMenu,
    QAction,  # unused
    QMessageBox,
    QProgressBar,
    QDialog,
    QLineEdit,
    QComboBox,
    QDialogButtonBox,
    QGroupBox,
    QRadioButton,
    QButtonGroup,
    QSpinBox,
    QTextEdit,
    QGridLayout,
)
from PyQt5.QtGui import (
    QPixmap,
    QWheelEvent,
    QImage,
    QPainter,
    QTransform,  # unused
    QPen,
    QColor,
    QCursor,
    QMouseEvent,
    QKeySequence,
)
from PyQt5.QtCore import (
    Qt,
    QThread,
    pyqtSignal,
    QPoint,
    QEvent,
)  # QThread, pyqtSignal, QPoint unused
import cv2
import numpy as np  # unused
from PIL import Image, ImageQt  # unused

try:
    import exifread

    EXIFREAD_AVAILABLE = True
except ImportError:
    EXIFREAD_AVAILABLE = False
import time  # unusedC
import re
import shutil
import yaml
import numpy as np
import onnxruntime as ort


def natural_sort_key(s):
    """숫자가 포함된 문자열의 자연스러운 정렬을 위한 키 함수"""
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split("([0-9]+)", s)
    ]


def calculate_iou(box1, box2):
    """두 바운딩 박스의 IOU(Intersection over Union) 계산

    Args:
        box1, box2: [x_center, y_center, width, height] (정규화된 좌표)

    Returns:
        float: IOU 값 (0.0 ~ 1.0)
    """
    # YOLO 형식을 절대 좌표로 변환
    x1_1, y1_1, w1, h1 = box1
    x1_2, y1_2, w2, h2 = box2

    # 바운딩 박스의 좌상단, 우하단 좌표 계산
    x1_min1 = x1_1 - w1 / 2
    y1_min1 = y1_1 - h1 / 2
    x1_max1 = x1_1 + w1 / 2
    y1_max1 = y1_1 + h1 / 2

    x1_min2 = x1_2 - w2 / 2
    y1_min2 = y1_2 - h2 / 2
    x1_max2 = x1_2 + w2 / 2
    y1_max2 = y1_2 + h2 / 2

    # 교집합 영역 계산
    inter_x_min = max(x1_min1, x1_min2)
    inter_y_min = max(y1_min1, y1_min2)
    inter_x_max = min(x1_max1, x1_max2)
    inter_y_max = min(y1_max1, y1_max2)

    # 교집합이 없는 경우
    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0

    # 교집합과 합집합 면적 계산
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area

    # IOU 계산
    if union_area == 0:
        return 0.0

    return inter_area / union_area


def letterbox(
    im,
    new_shape=(640, 640),
    color=(114, 114, 114),
    auto=True,
    scaleFill=False,
    scaleup=True,
):
    """YOLO 전처리: letterbox padding"""
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)
    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(
        im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return im, ratio, (dw, dh)


def xyxy2xywh(x):
    """xyxy 좌표를 xywh 좌표로 변환"""
    y = np.zeros_like(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # center x
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # center y
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


class LabelDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add/Edit Label")

        layout = QVBoxLayout()

        # 클래스 선택
        self.class_combo = QComboBox()
        self.class_combo.addItems(["0", "1", "2", "3", "4"])  # 클래스 목록
        layout.addWidget(QLabel("Class:"))
        layout.addWidget(self.class_combo)

        # 좌표 입력
        self.x_edit = QLineEdit()
        self.y_edit = QLineEdit()
        self.w_edit = QLineEdit()
        self.h_edit = QLineEdit()

        layout.addWidget(QLabel("X (center):"))
        layout.addWidget(self.x_edit)
        layout.addWidget(QLabel("Y (center):"))
        layout.addWidget(self.y_edit)
        layout.addWidget(QLabel("Width:"))
        layout.addWidget(self.w_edit)
        layout.addWidget(QLabel("Height:"))
        layout.addWidget(self.h_edit)

        # 버튼
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.setLayout(layout)

    def get_label(self):
        try:
            return [
                int(self.class_combo.currentText()),
                float(self.x_edit.text()),
                float(self.y_edit.text()),
                float(self.w_edit.text()),
                float(self.h_edit.text()),
            ]
        except ValueError:
            return None


class ImageViewer(QWidget):
    # 클래스 시작 인덱스 오프셋
    CLASS_OFFSET = 3

    def __init__(self):
        super(ImageViewer, self).__init__()
        self.image_list = []
        self.current_index = 0
        self.selected_label_index = None  # 선택된 라벨의 인덱스
        self.label_selection_mode = False  # 라벨 선택 모드 상태 (레거시)
        self.erase_mode = False  # 삭제 모드 활성화 여부
        self.selected_class = 3  # 선택된 클래스 (기본값: 3, Heart A)
        self.copy_mode_all = True  # True: Copy All, False: Copy Class
        self.selected_suit = None  # 선택된 문양 (z/x/c/v로 선택)

        # 기본 폴더 경로 설정
        self.default_image_folder = "./images"  # 기본 이미지 폴더
        self.label_folder = "./labels"  # 기본 라벨 폴더

        self.labels = {}  # {image_path: [label1, label2, ...]}
        self.drawing = False
        self.start_point = None
        self.end_point = None
        self.scale_factor = 1.0
        self.current_mouse_pos = None  # 현재 마우스 위치 추적
        self.fixed_scale_factor = None  # unused
        self.original_pixmap = None

        # 이미지 패닝(이동) 관련 변수들
        self.panning = False  # 이미지 패닝 모드
        self.pan_start_pos = None  # 패닝 시작 위치

        # 박스 편집 관련 변수들
        self.edit_mode = False  # 편집 모드 활성화 여부
        self.editing_label_index = None  # 편집 중인 라벨 인덱스
        self.drag_handle = None  # 드래그 중인 핸들 ('move', 'nw', 'ne', 'sw', 'se', 'n', 's', 'e', 'w')
        self.drag_start_pos = None  # 드래그 시작 위치 (이미지 좌표)
        self.drag_start_box = None  # 드래그 시작 시점의 박스 좌표 (x1, y1, x2, y2)
        self.handle_size = 4  # 핸들 크기 (픽셀) - 줄임

        # 표시 설정 관련 변수들
        self.show_class_name = True  # 클래스 이름 표시 여부
        self.bbox_thickness = 0.7  # 기본 bbox 두께
        self.new_bbox_thickness = 0.7  # 새로 추가된 라벨의 bbox 두께

        # ONNX Runtime 관련 변수들
        self.onnx_session = None
        self.model_path = None
        self.device = "cpu"  # 기본값: CPU

        # 실행 취소 관련 변수들
        self.undo_stack = []  # 실행 취소 스택
        self.max_undo_history = 50  # 최대 실행 취소 히스토리 개수

        # 단축키 다이얼로그 인스턴스
        self.shortcuts_dialog = None

        # 삭제 커서
        self.erase_cursor = None

        self.initUI()

    def initUI(self):
        self.setWindowTitle("Enhanced Image Viewer")
        self.resize(1600, 900)  # 창 크기 조절 가능하도록 변경
        self.setMinimumSize(800, 600)  # 최소 크기 설정

        # 메인 레이아웃
        main_layout = QHBoxLayout()

        # 왼쪽 패널
        left_panel = QVBoxLayout()

        # 폴더 선택 버튼 그룹
        folder_layout = QHBoxLayout()

        self.select_image_folder_button = QPushButton("Select Image Folder", self)
        self.select_image_folder_button.clicked.connect(self.select_folder)

        folder_layout.addWidget(self.select_image_folder_button)

        # 이미지 표시 영역
        self.scroll_area = QScrollArea()
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMouseTracking(True)  # 마우스 추적 활성화 (십자선 표시용)
        # 이미지 레이블의 마우스 이벤트를 위젯으로 전달하기 위한 이벤트 필터 설치
        self.image_label.installEventFilter(self)
        self.scroll_area.setWidget(self.image_label)
        self.scroll_area.setWidgetResizable(True)

        # 위젯 레벨에서도 마우스 추적 활성화
        self.setMouseTracking(True)

        # 편집 모드 표시용 레이블 (이미지 바깥에 표시)
        self.edit_mode_label = QLabel("Edit ON", self)
        self.edit_mode_label.setStyleSheet(
            "background-color: rgba(0, 255, 0, 200); "
            "color: white; "
            "font-weight: bold; "
            "font-size: 30px; "
            "padding: 5px 10px; "
            "border-radius: 5px;"
        )
        self.edit_mode_label.setAlignment(Qt.AlignCenter)
        self.edit_mode_label.hide()  # 초기에는 숨김

        # 삭제 모드 표시용 레이블
        self.erase_mode_label = QLabel("Erase ON", self)
        self.erase_mode_label.setStyleSheet(
            "background-color: rgba(255, 0, 0, 200); "
            "color: white; "
            "font-weight: bold; "
            "font-size: 30px; "
            "padding: 5px 10px; "
            "border-radius: 5px;"
        )
        self.erase_mode_label.setAlignment(Qt.AlignCenter)
        self.erase_mode_label.hide()  # 초기에는 숨김

        # 빨간색 화살표 커서 생성
        self.create_erase_cursor()

        # 파일명 표시
        self.filename_label = QLabel()
        self.filename_label.setAlignment(Qt.AlignCenter)

        # 단축키 리스트 버튼
        self.shortcuts_button = QPushButton("Keyboard Shortcuts", self)
        self.shortcuts_button.clicked.connect(self.show_shortcuts_dialog)

        # 진행 상태 표시
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)

        # 버튼 그룹
        button_layout = QHBoxLayout()

        # 기본 버튼들
        self.previous_button = QPushButton("Previous (←)")
        self.next_button = QPushButton("Next (→)")

        # YOLO 라벨 관련 버튼 추가
        self.select_label_folder_button = QPushButton("Select Label Folder", self)
        self.select_label_folder_button.clicked.connect(self.select_label_folder)

        self.add_label_button = QPushButton("Add Label (A)", self)
        self.add_label_button.clicked.connect(self.start_drawing)

        self.erase_mode_button = QPushButton("Erase Mode (E)", self)
        self.erase_mode_button.setCheckable(True)
        self.erase_mode_button.clicked.connect(self.toggle_erase_mode)

        self.clear_all_labels_button = QPushButton("Clear All Labels (O)", self)
        self.clear_all_labels_button.clicked.connect(self.clear_all_labels)

        # 버튼 연결
        self.previous_button.clicked.connect(self.show_previous_image)
        self.next_button.clicked.connect(self.show_next_image)

        # 버튼 레이아웃에 추가
        button_layout.addWidget(self.previous_button)
        button_layout.addWidget(self.next_button)
        button_layout.addWidget(self.select_label_folder_button)
        button_layout.addWidget(self.add_label_button)
        button_layout.addWidget(self.erase_mode_button)
        button_layout.addWidget(self.clear_all_labels_button)

        # ONNX 모델 관련 버튼들 추가
        self.load_model_button = QPushButton("Load Model (L)", self)
        self.load_model_button.clicked.connect(self.load_existing_model)
        button_layout.addWidget(self.load_model_button)

        self.auto_label_button = QPushButton("Auto Label Current (G)", self)
        self.auto_label_button.clicked.connect(self.auto_label_current_image)
        button_layout.addWidget(self.auto_label_button)

        # 박스 편집 모드 버튼 추가
        self.edit_mode_button = QPushButton("Edit Mode (R)", self)
        self.edit_mode_button.setCheckable(True)
        self.edit_mode_button.clicked.connect(self.toggle_edit_mode)
        button_layout.addWidget(self.edit_mode_button)

        # 이전 이미지 라벨 복사 버튼 추가
        self.copy_prev_label_button = QPushButton("Copy Prev Label (F)", self)
        self.copy_prev_label_button.clicked.connect(self.copy_previous_label)
        button_layout.addWidget(self.copy_prev_label_button)

        # 왼쪽 패널에 위젯 추가
        left_panel.addLayout(folder_layout)
        left_panel.addWidget(self.scroll_area)
        left_panel.addWidget(self.filename_label)
        left_panel.addWidget(self.shortcuts_button)
        left_panel.addWidget(self.progress_bar)
        left_panel.addLayout(button_layout)

        # 오른쪽 패널 (이미지 목록)
        right_panel = QVBoxLayout()

        # 클래스 선택 패널 추가 (포커 카드용)
        class_group = QGroupBox("Card Selection")
        class_layout = QVBoxLayout()

        # 라디오 버튼 그룹 생성
        self.class_button_group = QButtonGroup()
        self.class_buttons = []

        # 그리드 레이아웃 추가
        grid_layout = QGridLayout()

        # 포커 카드 문양과 숫자 정의
        suits = ["♥", "♦", "♣", "♠"]
        ranks = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]

        # 헤더 행 추가 (문양)
        for col, suit in enumerate(suits):
            header_label = QLabel(suit)
            header_label.setAlignment(Qt.AlignCenter)
            header_label.setStyleSheet("font-weight: bold; font-size: 30px;")
            grid_layout.addWidget(header_label, 0, col + 1)

        # 헤더 열 추가 (숫자)
        for row, rank in enumerate(ranks):
            header_label = QLabel(rank)
            header_label.setAlignment(Qt.AlignCenter)
            header_label.setStyleSheet("font-weight: bold; font-size: 30px;")
            grid_layout.addWidget(header_label, row + 1, 0)

        # 카드 버튼 생성 (52개)
        # 클래스 ID: chip=0, null=1, back=2, Heart A-K=3-15, Diamond A-K=16-28, Club A-K=29-41, Spade A-K=42-54
        for col, suit in enumerate(suits):
            for row, rank in enumerate(ranks):
                class_id = (
                    col * 13 + row + self.CLASS_OFFSET
                )  # Heart A-K=3-15, Diamond A-K=16-28, Club A-K=29-41, Spade A-K=42-54
                button = QRadioButton(f"{rank}{suit}")
                button.setMinimumSize(50, 30)
                # 버튼 글자 크기 설정
                button.setStyleSheet("font-size: 30px;")
                self.class_button_group.addButton(button, class_id)
                grid_layout.addWidget(button, row + 1, col + 1)
                self.class_buttons.append(button)

        class_layout.addLayout(grid_layout)

        # chip, null, back 버튼 추가 (맨 아래에)
        chip_null_back_layout = QHBoxLayout()
        chip_button = QRadioButton("chip")
        chip_button.setMinimumSize(60, 30)
        chip_button.setStyleSheet("font-size: 15px;")
        null_button = QRadioButton("null")
        null_button.setMinimumSize(60, 30)
        null_button.setStyleSheet("font-size: 15px;")
        back_button = QRadioButton("back")
        back_button.setMinimumSize(60, 30)
        back_button.setStyleSheet("font-size: 15px;")

        # chip, null, back을 클래스 0, 1, 2로 추가
        self.class_button_group.addButton(chip_button, 0)
        self.class_button_group.addButton(null_button, 1)
        self.class_button_group.addButton(back_button, 2)
        self.class_buttons.append(chip_button)
        self.class_buttons.append(null_button)
        self.class_buttons.append(back_button)

        chip_null_back_layout.addWidget(chip_button)
        chip_null_back_layout.addWidget(null_button)
        chip_null_back_layout.addWidget(back_button)
        chip_null_back_layout.addStretch()
        class_layout.addLayout(chip_null_back_layout)

        # 클래스 선택 이벤트 연결
        self.class_button_group.buttonClicked.connect(self.on_class_selected)

        # 기본 선택: Heart A (클래스 3)
        if len(self.class_buttons) > 0:
            self.class_buttons[0].setChecked(True)
            self.selected_class = 3

        class_group.setLayout(class_layout)
        right_panel.addWidget(class_group, 1)  # stretch factor를 1로 설정하여 공간 확보

        # 현재 선택된 클래스 표시
        self.selected_class_label = QLabel("Selected: A♥ (Class 3)")
        right_panel.addWidget(self.selected_class_label)

        # 구분선 추가
        separator = QLabel("─" * 30)
        separator.setAlignment(Qt.AlignCenter)
        right_panel.addWidget(separator)

        # ONNX 모델 설정 패널 추가
        yolo_group = QGroupBox("ONNX Model Settings")
        yolo_layout = QVBoxLayout()

        # 모델 상태 표시
        self.model_status_label = QLabel("Model: Not loaded")
        yolo_layout.addWidget(self.model_status_label)

        # 현재 위치 정보 표시 (클릭 가능한 입력 필드)
        position_layout = QHBoxLayout()
        position_layout.addWidget(QLabel("Position:"))
        self.position_info_label = QLineEdit("0/0")
        self.position_info_label.setReadOnly(False)
        self.position_info_label.setMaximumWidth(300)
        self.position_info_label.setToolTip(
            "이미지 번호를 입력하고 Enter를 누르면 해당 이미지로 이동합니다"
        )
        self.position_info_label.returnPressed.connect(self.jump_to_image)
        position_layout.addWidget(self.position_info_label)
        position_layout.addStretch()
        yolo_layout.addLayout(position_layout)

        yolo_group.setLayout(yolo_layout)
        right_panel.addWidget(yolo_group)

        # 구분선 추가
        separator2 = QLabel("─" * 30)
        separator2.setAlignment(Qt.AlignCenter)
        right_panel.addWidget(separator2)

        # 표시 설정 패널 추가
        display_group = QGroupBox("Display Settings")
        display_layout = QVBoxLayout()

        # 클래스 이름 표시 on/off
        class_name_layout = QHBoxLayout()
        class_name_layout.addWidget(QLabel("Show Class Name:"))
        self.show_class_name_button = QPushButton("ON", self)
        self.show_class_name_button.setCheckable(True)
        self.show_class_name_button.setChecked(True)
        self.show_class_name_button.setMinimumWidth(60)
        self.show_class_name_button.setMaximumWidth(60)
        self.show_class_name_button.clicked.connect(self.toggle_class_name_display)
        class_name_layout.addWidget(self.show_class_name_button)
        display_layout.addLayout(class_name_layout)

        display_group.setLayout(display_layout)
        right_panel.addWidget(display_group)

        # 구분선 추가
        separator_copy = QLabel("─" * 30)
        separator_copy.setAlignment(Qt.AlignCenter)
        right_panel.addWidget(separator_copy)

        # 복사 모드 설정 패널 추가
        copy_group = QGroupBox("Copy Mode")
        copy_layout = QVBoxLayout()

        # 복사 모드 선택 (Copy All / Copy Class)
        copy_mode_layout = QHBoxLayout()
        copy_mode_layout.addWidget(QLabel("Copy Mode:"))
        self.copy_mode_button = QPushButton("Copy All", self)
        self.copy_mode_button.setCheckable(True)
        self.copy_mode_button.setChecked(True)  # 기본값: Copy All
        self.copy_mode_button.setMinimumWidth(120)
        self.copy_mode_button.setMaximumWidth(120)
        self.copy_mode_button.setMinimumHeight(35)
        self.copy_mode_button.clicked.connect(self.toggle_copy_mode)
        copy_mode_layout.addWidget(self.copy_mode_button)
        copy_layout.addLayout(copy_mode_layout)

        copy_group.setLayout(copy_layout)
        right_panel.addWidget(copy_group)

        # 구분선 추가
        separator3 = QLabel("─" * 30)
        separator3.setAlignment(Qt.AlignCenter)
        right_panel.addWidget(separator3)

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

        # 컨텍스트 메뉴 설정
        self.setup_context_menu()

        # 애플리케이션 시작 시 기본 이미지 폴더 로드
        if os.path.exists(self.default_image_folder):
            self.load_images(self.default_image_folder, 0, 1000)

    def create_erase_cursor(self):
        """빨간색 X 모양 삭제 커서 생성"""
        # 16x16 크기의 빨간색 X 이미지 생성
        pixmap = QPixmap(16, 16)
        pixmap.fill(Qt.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)

        # 빨간색 X 그리기
        painter.setPen(QPen(QColor(255, 0, 0), 2))

        # 대각선 두 개로 X 그리기
        painter.drawLine(2, 2, 14, 14)  # 왼쪽 위에서 오른쪽 아래
        painter.drawLine(14, 2, 2, 14)  # 오른쪽 위에서 왼쪽 아래
        painter.end()

        self.erase_cursor = QCursor(pixmap, 8, 8)  # 핫스팟을 중앙으로 설정

    def resizeEvent(self, event):
        """윈도우 크기 변경 시 편집/삭제 모드 레이블 위치 업데이트"""
        super().resizeEvent(event)
        if self.edit_mode:
            self.update_edit_mode_label_position()
        if self.erase_mode:
            self.update_erase_mode_label_position()

    def setup_shortcuts(self):
        shortcuts = {
            Qt.Key_A: self.start_drawing,
            Qt.Key_E: self.toggle_erase_mode,
            Qt.Key_O: self.clear_all_labels,
            Qt.Key_Plus: lambda: self.zoom_image(1.25),
            Qt.Key_Minus: lambda: self.zoom_image(0.8),
            Qt.Key_G: self.auto_label_current_image,
            Qt.Key_R: self.toggle_edit_mode,
            Qt.Key_F: self.copy_previous_label,
            Qt.Key_M: self.toggle_class_name_display,
            Qt.Key_L: self.load_existing_model,
        }

        # Ctrl+Z 단축키 추가
        undo_shortcut = QShortcut(QKeySequence("Ctrl+Z"), self)
        undo_shortcut.activated.connect(self.undo_last_action)

        # ESC 키로 fit to window
        esc_shortcut = QShortcut(QKeySequence("Escape"), self)
        esc_shortcut.activated.connect(self.fit_to_window)

        for key, slot in shortcuts.items():
            shortcut = QShortcut(key, self)
            shortcut.activated.connect(slot)

        # 좌우 화살표 키는 편집 모드가 아닐 때만 이미지 이동
        left_shortcut = QShortcut(Qt.Key_Left, self)
        left_shortcut.activated.connect(self.handle_left_key)
        right_shortcut = QShortcut(Qt.Key_Right, self)
        right_shortcut.activated.connect(self.handle_right_key)

        # 콤마(,)와 백틱(`) 키로 이미지 이동 (편집 모드가 아닐 때만)
        comma_shortcut = QShortcut(Qt.Key_Comma, self)
        comma_shortcut.activated.connect(self.handle_comma_key)
        backtick_shortcut = QShortcut(Qt.Key_QuoteLeft, self)
        backtick_shortcut.activated.connect(self.handle_backtick_key)

        # 위아래 화살표 키는 편집 모드에서만 박스 크기 조정
        up_shortcut = QShortcut(Qt.Key_Up, self)
        up_shortcut.activated.connect(self.handle_up_key)
        down_shortcut = QShortcut(Qt.Key_Down, self)
        down_shortcut.activated.connect(self.handle_down_key)

        # 문양 선택 단축키 (z, x, c, v)
        suit_shortcuts = {
            Qt.Key_Z: 0,  # Heart
            Qt.Key_X: 1,  # Diamond
            Qt.Key_C: 2,  # Club
            Qt.Key_V: 3,  # Spade
        }
        for key, suit_index in suit_shortcuts.items():
            shortcut = QShortcut(key, self)
            shortcut.activated.connect(
                lambda checked=False, suit=suit_index: self.select_suit(suit)
            )

        # 숫자/문자 선택 단축키 (1-0, j, q, k)
        # 1->A, 2->2, ..., 9->9, 0->10, j->J, q->Q, k->K
        rank_mapping = {
            Qt.Key_1: 0,  # A
            Qt.Key_2: 1,  # 2
            Qt.Key_3: 2,  # 3
            Qt.Key_4: 3,  # 4
            Qt.Key_5: 4,  # 5
            Qt.Key_6: 5,  # 6
            Qt.Key_7: 6,  # 7
            Qt.Key_8: 7,  # 8
            Qt.Key_9: 8,  # 9
            Qt.Key_0: 9,  # 10
            Qt.Key_J: 10,  # J
            Qt.Key_Q: 11,  # Q
            Qt.Key_K: 12,  # K
        }
        for key, rank_index in rank_mapping.items():
            shortcut = QShortcut(key, self)
            shortcut.activated.connect(
                lambda checked=False, rank=rank_index: self.select_rank(rank)
            )

        # chip, null, back 클래스 선택 단축키 (H: chip, N: null, B: back)
        # C 키는 Club 문양 선택에 사용되므로 chip은 H 키 사용
        chip_shortcut = QShortcut(Qt.Key_H, self)
        chip_shortcut.activated.connect(lambda: self.select_class_by_id(0))
        null_shortcut = QShortcut(Qt.Key_N, self)
        null_shortcut.activated.connect(lambda: self.select_class_by_id(1))
        back_shortcut = QShortcut(Qt.Key_B, self)
        back_shortcut.activated.connect(lambda: self.select_class_by_id(2))

    def setup_context_menu(self):
        self.image_label.setContextMenuPolicy(Qt.CustomContextMenu)
        self.image_label.customContextMenuRequested.connect(self.show_context_menu)

    def show_context_menu(self, pos):
        menu = QMenu(self)

        zoom_in_action = menu.addAction("Zoom In")
        zoom_in_action.triggered.connect(lambda: self.zoom_image(1.25))
        zoom_out_action = menu.addAction("Zoom Out")
        zoom_out_action.triggered.connect(lambda: self.zoom_image(0.8))
        fit_to_window_action = menu.addAction("Fit to Window (ESC)")
        fit_to_window_action.triggered.connect(self.fit_to_window)

        menu.exec_(self.image_label.mapToGlobal(pos))

    def load_images(self, folder_path, start, end):
        try:
            exclude_files = set()
            try:
                with open("D:/edge_dataset/camco_miss.txt", "r") as f:
                    exclude_files = set(line.strip() for line in f)
            except Exception as e:
                print(f"Error reading camco_miss.txt: {e}")

            if not os.path.exists(folder_path):
                QMessageBox.warning(
                    self, "Error", f"Folder does not exist: {folder_path}"
                )
                return

            self.image_list = [
                os.path.join(folder_path, file)
                for file in os.listdir(folder_path)
                if file.lower().endswith((".png", ".jpg", ".jpeg"))
                and file not in exclude_files
            ]
            # 자연스러운 숫자 정렬 적용
            self.image_list.sort(key=lambda x: natural_sort_key(os.path.basename(x)))
            self.image_list = self.image_list[start:end]
            self.current_index = 0

            if self.image_list:
                self.show_image()
            else:
                QMessageBox.warning(
                    self, "No Images", "No images found in the selected folder."
                )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load images: {str(e)}")
            print(f"Error in load_images: {str(e)}")

    def show_image(self, skip_label_reload=False):
        if not self.image_list:
            return

        # current_index 범위 체크
        if self.current_index < 0 or self.current_index >= len(self.image_list):
            return

        image_path = self.image_list[self.current_index]

        # 현재 이미지의 라벨 파일이 새로 생성되었는지 확인하고 로드
        # (드래그 중에는 스킵)
        if self.label_folder and not skip_label_reload:
            self.load_current_image_label(image_path)

        try:
            # 이미지 로드 (한글 경로 지원)
            # OpenCV의 imread는 한글 경로를 제대로 처리하지 못하므로
            # np.fromfile과 cv2.imdecode를 사용
            img_array = np.fromfile(image_path, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img is None:
                print(f"Warning: Failed to load image: {image_path}")
                return

            # 바운딩 박스 그리기
            if image_path in self.labels:
                for i, label in enumerate(self.labels[image_path]):
                    if len(label) >= 6:
                        if len(label) == 20:  # 새로운 형식: cls x y w h (k1 k2 vis)x5
                            class_id, x, y, w, h = (
                                label[0],
                                label[1],
                                label[2],
                                label[3],
                                label[4],
                            )
                        else:  # 기존 형식: cls x y w h conf
                            class_id, x, y, w, h, conf = label
                    else:  # 예전 라벨 형식: cls x y w h
                        class_id, x, y, w, h = label
                    height, width = img.shape[:2]

                    # YOLO 좌표를 픽셀 좌표로 변환
                    x1 = int((x - w / 2) * width)
                    y1 = int((y - h / 2) * height)
                    x2 = int((x + w / 2) * width)
                    y2 = int((y + h / 2) * height)

                    # 좌표가 이미지 범위를 벗어나지 않도록 조정
                    x1 = max(0, min(x1, width - 1))
                    y1 = max(0, min(y1, height - 1))
                    x2 = max(0, min(x2, width - 1))
                    y2 = max(0, min(y2, height - 1))

                    # 포커 카드 문양별 색상 결정
                    suits = ["♥", "♦", "♣", "♠"]
                    ranks = [
                        "A",
                        "2",
                        "3",
                        "4",
                        "5",
                        "6",
                        "7",
                        "8",
                        "9",
                        "10",
                        "J",
                        "Q",
                        "K",
                    ]

                    # 클래스 ID에 따라 이름과 색상 결정
                    if class_id == 0:
                        card_name = "chip"
                        color = (0, 255, 255)  # 청록색
                    elif class_id == 1:
                        card_name = "null"
                        color = (128, 128, 128)  # 회색
                    elif class_id == 2:
                        card_name = "back"
                        color = (200, 200, 200)  # 밝은 회색
                    else:
                        # 카드: class_id = (suit_index * 13 + rank_index) + CLASS_OFFSET
                        # suit_index와 rank_index 계산 시 오프셋 제거
                        adjusted_id = class_id - self.CLASS_OFFSET
                        suit_index = (
                            adjusted_id // 13
                        )  # 0: Heart, 1: Diamond, 2: Club, 3: Spade
                        rank_index = adjusted_id % 13  # 0: A, 1: 2, ..., 12: K

                        # 카드 이름 생성 (유니코드 문양 사용, 문양+숫자 순서)
                        if 0 <= suit_index < len(suits) and 0 <= rank_index < len(
                            ranks
                        ):
                            card_name = f"{suits[suit_index]}{ranks[rank_index]}"
                        else:
                            card_name = f"Class_{class_id}"

                        # 문양별 색상 지정
                        suit_colors = {
                            0: (0, 0, 255),  # Heart: 빨간색
                            1: (0, 100, 255),  # Diamond: 빨간색 계열 (밝은 빨간색)
                            2: (0, 255, 0),  # Club: 녹색
                            3: (139, 0, 139),  # Spade: 보라색 계열 (진한 보라색)
                        }

                        if 0 <= suit_index < len(suits):
                            color = suit_colors.get(suit_index, (255, 255, 255))
                        else:
                            color = (255, 255, 255)  # 기본값: 흰색

                    # 새로 추가된 라벨인지 확인 (라벨 길이가 5이고 confidence가 없는 경우)
                    is_new_label = len(label) == 5
                    thickness = max(
                        1,
                        int(
                            self.new_bbox_thickness
                            if is_new_label
                            else self.bbox_thickness
                        ),
                    )

                    # 바운딩 박스 그리기
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

                    # 클래스 이름 표시 (설정에 따라) - 카드 이름으로 표시 (PIL 사용)
                    if self.show_class_name:
                        try:
                            from PIL import Image, ImageDraw, ImageFont

                            # OpenCV 이미지를 PIL 이미지로 변환
                            img_pil = Image.fromarray(
                                cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            )
                            draw = ImageDraw.Draw(img_pil)

                            # 폰트 설정 (시스템 기본 폰트 사용, 크기 증가)
                            try:
                                # Windows의 경우
                                font = ImageFont.truetype("arial.ttf", 40)
                            except:
                                try:
                                    # Linux의 경우
                                    font = ImageFont.truetype(
                                        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                                        40,
                                    )
                                except:
                                    # 기본 폰트 사용
                                    font = ImageFont.load_default()

                            # 텍스트 위치 (bbox 위쪽으로 더 많이 이동)
                            text_y = max(40, y1 - 40)

                            # PIL은 RGB 색상 사용
                            text_color = (color[2], color[1], color[0])  # BGR -> RGB

                            # 텍스트 그리기
                            draw.text(
                                (x1, text_y), card_name, fill=text_color, font=font
                            )

                            # PIL 이미지를 다시 OpenCV 이미지로 변환
                            img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
                        except Exception as e:
                            # PIL 사용 실패 시 기본 cv2.putText 사용 (영문자로 표시)
                            print(f"PIL text rendering failed: {e}, using cv2.putText")
                            suit_texts = ["H", "D", "C", "S"]
                            if class_id >= 3:  # 포커 카드 클래스 (3 이상)
                                adjusted_id = class_id - self.CLASS_OFFSET
                                suit_index = adjusted_id // 13
                                rank_index = adjusted_id % 13
                                if 0 <= suit_index < len(
                                    suit_texts
                                ) and 0 <= rank_index < len(ranks):
                                    card_name = (
                                        f"{ranks[rank_index]}{suit_texts[suit_index]}"
                                    )
                            cv2.putText(
                                img,
                                card_name,
                                (x1, max(30, y1 - 30)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.2,
                                color,
                                max(2, thickness),
                            )

                    # 편집 모드이고 선택된 박스인 경우 핸들 그리기
                    if self.edit_mode and self.editing_label_index == i:
                        # 선택된 박스는 더 두껍게 그리고 파란색으로
                        edit_thickness = max(
                            1,
                            int(max(self.bbox_thickness, self.new_bbox_thickness) + 1),
                        )
                        cv2.rectangle(
                            img, (x1, y1), (x2, y2), (255, 0, 0), edit_thickness
                        )
                        # 핸들 그리기 (8개 모서리/변)
                        handle_color = (255, 255, 0)  # 노란색
                        handle_thickness = 2
                        hs = self.handle_size

                        # 모서리 핸들
                        cv2.rectangle(
                            img,
                            (x1 - hs, y1 - hs),
                            (x1 + hs, y1 + hs),
                            handle_color,
                            -1,
                        )  # 좌상단
                        cv2.rectangle(
                            img,
                            (x2 - hs, y1 - hs),
                            (x2 + hs, y1 + hs),
                            handle_color,
                            -1,
                        )  # 우상단
                        cv2.rectangle(
                            img,
                            (x1 - hs, y2 - hs),
                            (x1 + hs, y2 + hs),
                            handle_color,
                            -1,
                        )  # 좌하단
                        cv2.rectangle(
                            img,
                            (x2 - hs, y2 - hs),
                            (x2 + hs, y2 + hs),
                            handle_color,
                            -1,
                        )  # 우하단

                        # 변 중앙 핸들
                        mid_x = (x1 + x2) // 2
                        mid_y = (y1 + y2) // 2
                        cv2.rectangle(
                            img,
                            (mid_x - hs, y1 - hs),
                            (mid_x + hs, y1 + hs),
                            handle_color,
                            -1,
                        )  # 상단 중앙
                        cv2.rectangle(
                            img,
                            (mid_x - hs, y2 - hs),
                            (mid_x + hs, y2 + hs),
                            handle_color,
                            -1,
                        )  # 하단 중앙
                        cv2.rectangle(
                            img,
                            (x1 - hs, mid_y - hs),
                            (x1 + hs, mid_y + hs),
                            handle_color,
                            -1,
                        )  # 좌측 중앙
                        cv2.rectangle(
                            img,
                            (x2 - hs, mid_y - hs),
                            (x2 + hs, mid_y + hs),
                            handle_color,
                            -1,
                        )  # 우측 중앙

            # OpenCV 이미지를 QImage로 변환
            height, width, channel = img.shape
            bytes_per_line = 3 * width
            q_img = QImage(
                img.data, width, height, bytes_per_line, QImage.Format_RGB888
            ).rgbSwapped()

            # QImage를 QPixmap으로 변환
            self.original_pixmap = QPixmap.fromImage(q_img)
            self.update_image_display()

            # 파일명 표시 (라벨 개수 포함)
            filename = os.path.basename(image_path)
            label_count = len(self.labels.get(image_path, []))
            filename_with_labels = f"{filename} ({label_count} labels)"
            self.filename_label.setText(filename_with_labels)

            # 이미지 목록 업데이트
            self.update_image_list_widget()

            # 위치 정보 업데이트
            self.update_position_info()

        except Exception as e:
            # 이미지 로드 실패 시 조용히 처리 (콘솔에만 출력)
            print(f"Error in show_image: {str(e)}")
            # 이미지가 없을 때는 빈 화면을 표시하지 않음
            return

    def show_shortcuts_dialog(self):
        """단축키 리스트를 팝업으로 표시 (크기 조절 가능, 모달리스)"""
        # 이미 다이얼로그가 열려있으면 포커스만 주기
        if self.shortcuts_dialog is not None:
            self.shortcuts_dialog.raise_()
            self.shortcuts_dialog.activateWindow()
            return

        # 새 다이얼로그 생성
        dialog = QDialog(self)
        dialog.setWindowTitle("Keyboard Shortcuts")
        dialog.setModal(False)  # 모달리스로 설정
        dialog.resize(500, 600)  # 초기 크기 설정
        dialog.setMinimumSize(400, 400)  # 최소 크기 설정

        layout = QVBoxLayout()

        # 텍스트 에디터로 단축키 목록 표시
        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setPlainText(
            """키보드 단축키 목록

기본 네비게이션:
  ← (일반 모드)       : 이전 이미지
  → (일반 모드)       : 다음 이미지
  , (콤마)           : 이전 이미지
  ` (백틱)           : 다음 이미지
  + (Plus)           : 확대
  - (Minus)          : 축소
  Ctrl + 마우스 휠   : 확대/축소

라벨링 관련:
  A                  : 라벨 추가 모드 시작
  E                  : 삭제 모드 진입
  O                  : 모든 라벨 삭제
  R                  : 편집 모드 진입
  F                  : 이전 이미지 라벨 복사

클래스 선택 (포커 카드):
  1단계: 문양 선택 (z/x/c/v)
    z                  : 하트 (♥)
    x                  : 다이아몬드 (♦)
    c                  : 클럽 (♣)
    v                  : 스페이드 (♠)
  2단계: 숫자/문자 선택
    1                  : A
    2~9                : 2~9
    0                  : 10
    j                  : J
    q                  : Q
    k                  : K
  예시: 'z' 누른 후 '1' 누르면 하트 A (클래스 0) 선택
  예시: 'x' 누른 후 '2' 누르면 다이아몬드 2 (클래스 14) 선택

편집 모드 (R 키로 진입):
  - 박스 클릭하여 선택
  - 드래그하여 이동
  - 모서리/변 드래그하여 크기 조정
  ← (편집 모드)       : 선택된 박스 width 축소
  → (편집 모드)       : 선택된 박스 width 확대
  ↑ (편집 모드)       : 선택된 박스 height 축소
  ↓ (편집 모드)       : 선택된 박스 height 확대

ONNX 모델 관련:
  L                  : ONNX 모델 로드
  G                  : 현재 이미지 자동 라벨링

기타:
  M                  : 클래스 이름 표시 토글
  Ctrl+Z             : 실행 취소

마우스:
  휠 클릭 + 드래그     : 확대된 이미지 이동
  휠                 : 스크롤"""
        )

        layout.addWidget(text_edit)

        # 닫기 버튼
        close_button = QPushButton("Close")
        close_button.clicked.connect(dialog.close)
        layout.addWidget(close_button)

        dialog.setLayout(layout)

        # 다이얼로그가 닫힐 때 인스턴스 변수 초기화
        def on_dialog_finished():
            self.shortcuts_dialog = None

        dialog.finished.connect(on_dialog_finished)

        # 인스턴스 변수에 저장
        self.shortcuts_dialog = dialog

        # 다이얼로그 표시
        dialog.show()

    def zoom_image(self, factor):
        self.scale_factor *= factor
        self.update_image_display()

    def fit_to_window(self):
        """이미지를 창 크기에 맞게 조정"""
        if not self.original_pixmap:
            return

        # 스크롤 영역의 크기 가져오기
        scroll_width = self.scroll_area.viewport().width()
        scroll_height = self.scroll_area.viewport().height()

        # 이미지 원본 크기
        img_width = self.original_pixmap.width()
        img_height = self.original_pixmap.height()

        # 비율을 유지하면서 창에 맞는 scale_factor 계산
        scale_x = scroll_width / img_width if img_width > 0 else 1.0
        scale_y = scroll_height / img_height if img_height > 0 else 1.0

        # 더 작은 비율을 사용하여 이미지가 완전히 보이도록
        self.scale_factor = min(scale_x, scale_y)

        # 최소 크기 제한 (너무 작아지지 않도록)
        if self.scale_factor < 0.1:
            self.scale_factor = 0.1

        self.update_image_display()

    def update_image_display(self):
        if self.original_pixmap:
            # 원본 이미지 복사
            pixmap = self.original_pixmap.copy()

            painter = QPainter(pixmap)

            # 항상 마우스 위치에 십자선 그리기 (편집 모드가 아닐 때만)
            if not self.edit_mode and self.current_mouse_pos:
                try:
                    actual_x, actual_y = self.get_image_coordinates(
                        self.current_mouse_pos
                    )
                    # 십자선 그리기 (이미지 범위 체크는 하되, 범위를 약간 벗어나도 표시)
                    painter.setPen(QPen(QColor(255, 0, 0), 1, Qt.DashLine))
                    # 수평선 (이미지 전체 너비)
                    painter.drawLine(0, actual_y, pixmap.width(), actual_y)
                    # 수직선 (이미지 전체 높이)
                    painter.drawLine(actual_x, 0, actual_x, pixmap.height())
                except Exception as e:
                    # 디버깅을 위해 에러 출력
                    pass  # 좌표 계산 실패 시 무시

            # 드래그 중인 경우 바운딩 박스 미리보기 표시
            if self.drawing and self.start_point:
                # end_point가 없으면 현재 마우스 위치 사용
                if self.end_point:
                    end_pos = self.end_point
                elif self.current_mouse_pos:
                    end_pos = self.current_mouse_pos
                else:
                    end_pos = None

                if end_pos:
                    try:
                        start_x, start_y = self.get_image_coordinates(self.start_point)
                        end_x, end_y = self.get_image_coordinates(end_pos)

                        x1 = min(start_x, end_x)
                        y1 = min(start_y, end_y)
                        x2 = max(start_x, end_x)
                        y2 = max(start_y, end_y)

                        # bbox 미리보기 (청록색)
                        painter.setPen(QPen(QColor(0, 255, 255), 2))
                        painter.drawRect(x1, y1, x2 - x1, y2 - y1)
                    except:
                        pass  # 좌표 계산 실패 시 무시

            painter.end()

            # 스케일 적용
            scaled_pixmap = pixmap.scaled(
                pixmap.size() * self.scale_factor,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
            self.image_label.setPixmap(scaled_pixmap)

        # 편집 모드일 때 스크롤 영역 테두리와 텍스트 표시 업데이트
        self.update_edit_mode_display()

    def get_image_coordinates(self, pos):
        """마우스 좌표를 이미지 좌표로 변환

        Args:
            pos: QPoint - 위젯 좌표
        """
        # 스크롤 영역의 위치를 고려
        scroll_pos = self.scroll_area.viewport().mapFrom(self, pos)

        # 이미지 레이블의 위치를 고려
        label_pos = self.image_label.mapFrom(self.scroll_area.viewport(), scroll_pos)

        # 스케일 팩터를 고려하여 실제 이미지 좌표 계산
        img_width = self.original_pixmap.width()
        img_height = self.original_pixmap.height()

        # 이미지가 중앙 정렬되어 있으므로 오프셋 계산
        label_width = self.image_label.width()
        label_height = self.image_label.height()

        x_offset = (label_width - img_width * self.scale_factor) / 2
        y_offset = (label_height - img_height * self.scale_factor) / 2

        # 실제 이미지 좌표 계산
        x = (label_pos.x() - x_offset) / self.scale_factor
        y = (label_pos.y() - y_offset) / self.scale_factor

        # 좌표가 이미지 범위를 벗어나지 않도록 조정
        x = max(0, min(x, img_width - 1))
        y = max(0, min(y, img_height - 1))

        return int(x), int(y)

    def mousePressEvent(self, event):
        # 이미지가 로드되지 않은 경우 처리하지 않음
        if not self.image_list or not self.original_pixmap:
            return

        # current_index 범위 체크
        if self.current_index < 0 or self.current_index >= len(self.image_list):
            return

        if self.edit_mode and event.button() == Qt.LeftButton:
            # 편집 모드에서 클릭 처리
            click_pos = self.get_image_coordinates(event.pos())
            click_x, click_y = click_pos

            image_path = self.image_list[self.current_index]
            if image_path in self.labels and self.labels[image_path]:
                # 현재 편집 중인 박스가 있으면 핸들 체크
                if self.editing_label_index is not None:
                    if 0 <= self.editing_label_index < len(self.labels[image_path]):
                        label = self.labels[image_path][self.editing_label_index]
                        img_width = self.original_pixmap.width()
                        img_height = self.original_pixmap.height()
                        x1, y1, x2, y2 = self.get_box_coordinates(
                            label, img_width, img_height
                        )
                        handle = self.get_handle_at_position(
                            click_x, click_y, x1, y1, x2, y2
                        )

                        if handle:
                            # 실행 취소를 위한 상태 저장 (편집 시작 시점)
                            self.save_state_for_undo()
                            self.drag_handle = handle
                            self.drag_start_pos = click_pos
                            self.drag_start_box = (
                                x1,
                                y1,
                                x2,
                                y2,
                            )  # 원본 박스 좌표 저장
                            # 커서 변경
                            if handle == "move":
                                self.image_label.setCursor(Qt.SizeAllCursor)
                            else:
                                self.image_label.setCursor(Qt.SizeFDiagCursor)
                            return

                # 새로운 박스 선택
                label_idx = self.find_label_at_position(click_x, click_y)
                if label_idx is not None:
                    if 0 <= label_idx < len(self.labels[image_path]):
                        self.editing_label_index = label_idx
                        label = self.labels[image_path][label_idx]
                        img_width = self.original_pixmap.width()
                        img_height = self.original_pixmap.height()
                        x1, y1, x2, y2 = self.get_box_coordinates(
                            label, img_width, img_height
                        )
                        handle = self.get_handle_at_position(
                            click_x, click_y, x1, y1, x2, y2
                        )

                        if handle:
                            # 실행 취소를 위한 상태 저장 (편집 시작 시점)
                            self.save_state_for_undo()
                            self.drag_handle = handle
                            self.drag_start_pos = click_pos
                            self.drag_start_box = (
                                x1,
                                y1,
                                x2,
                                y2,
                            )  # 원본 박스 좌표 저장
                            # 커서 변경
                            if handle == "move":
                                self.image_label.setCursor(Qt.SizeAllCursor)
                            else:
                                self.image_label.setCursor(Qt.SizeFDiagCursor)
                        else:
                            self.editing_label_index = label_idx
                        self.show_image()
                    else:
                        self.editing_label_index = None
                        self.show_image()
                else:
                    self.editing_label_index = None
                    self.show_image()
            else:
                self.editing_label_index = None
                self.show_image()
        elif self.erase_mode and event.button() == Qt.LeftButton:
            # 삭제 모드에서 클릭 처리
            if 0 <= self.current_index < len(self.image_list):
                image_path = self.image_list[self.current_index]
                if image_path in self.labels:
                    click_pos = self.get_image_coordinates(event.pos())
                    self.erase_label_at_position(click_pos)
        elif (
            not self.edit_mode
            and not self.erase_mode
            and event.button() == Qt.LeftButton
        ):
            # 편집 모드가 아닐 때 좌클릭으로 바로 bbox 그리기 시작
            self.drawing = True
            # 위젯 좌표로 저장 (get_image_coordinates에서 변환)
            self.start_point = event.pos()
            self.end_point = None  # 드래그 시작 시점에는 end_point 없음
            self.image_label.setCursor(Qt.CrossCursor)
            self.update_image_display()
        elif (
            not self.drawing
            and not self.edit_mode
            and not self.erase_mode
            and event.button() == Qt.MiddleButton
        ):
            # 이미지 패닝 시작 (마우스 휠 클릭)
            self.panning = True
            self.pan_start_pos = event.pos()
            self.image_label.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, event):
        # 이미지가 로드되지 않은 경우 처리하지 않음
        if not self.image_list or not self.original_pixmap:
            return

        # current_index 범위 체크
        if self.current_index < 0 or self.current_index >= len(self.image_list):
            return

        # 마우스 위치 추적 (십자선 표시용)
        # 위젯 좌표로 저장 (get_image_coordinates에서 변환)
        self.current_mouse_pos = event.pos()

        if self.edit_mode:
            if self.drag_handle and self.drag_start_pos:
                # 편집 모드에서 드래그 중
                current_pos = self.get_image_coordinates(event.pos())
                self.update_box_during_drag(current_pos)
                self.update_image_display()
            else:
                # 편집 모드에서 마우스 이동 시 커서 업데이트
                click_pos = self.get_image_coordinates(event.pos())
                click_x, click_y = click_pos

                if 0 <= self.current_index < len(self.image_list):
                    image_path = self.image_list[self.current_index]
                    if image_path in self.labels and self.labels[image_path]:
                        if self.editing_label_index is not None:
                            if (
                                0
                                <= self.editing_label_index
                                < len(self.labels[image_path])
                            ):
                                label = self.labels[image_path][
                                    self.editing_label_index
                                ]
                                img_width = self.original_pixmap.width()
                                img_height = self.original_pixmap.height()
                                x1, y1, x2, y2 = self.get_box_coordinates(
                                    label, img_width, img_height
                                )
                                handle = self.get_handle_at_position(
                                    click_x, click_y, x1, y1, x2, y2
                                )

                                if handle == "move":
                                    self.image_label.setCursor(Qt.SizeAllCursor)
                                elif handle in ["nw", "se"]:
                                    self.image_label.setCursor(Qt.SizeFDiagCursor)
                                elif handle in ["ne", "sw"]:
                                    self.image_label.setCursor(Qt.SizeBDiagCursor)
                                elif handle in ["n", "s"]:
                                    self.image_label.setCursor(Qt.SizeVerCursor)
                                elif handle in ["e", "w"]:
                                    self.image_label.setCursor(Qt.SizeHorCursor)
                                else:
                                    self.image_label.setCursor(Qt.ArrowCursor)
                            else:
                                self.image_label.setCursor(Qt.ArrowCursor)
                        else:
                            # 박스 위에 있으면 선택 가능한 커서
                            label_idx = self.find_label_at_position(click_x, click_y)
                            if label_idx is not None:
                                self.image_label.setCursor(Qt.PointingHandCursor)
                            else:
                                self.image_label.setCursor(Qt.ArrowCursor)
                    else:
                        self.image_label.setCursor(Qt.ArrowCursor)
                else:
                    self.image_label.setCursor(Qt.ArrowCursor)
        elif (
            not self.drawing
            and not self.edit_mode
            and not self.erase_mode
            and not self.panning
        ):
            # 일반 상태일 때 십자선 커서
            self.image_label.setCursor(Qt.CrossCursor)
        # 마우스 이동 시 십자선 및 bbox 미리보기 업데이트
        if self.panning and self.pan_start_pos:
            # 이미지 패닝 중
            current_pos = event.pos()
            dx = current_pos.x() - self.pan_start_pos.x()
            dy = current_pos.y() - self.pan_start_pos.y()

            # 스크롤 위치 업데이트
            h_scroll = self.scroll_area.horizontalScrollBar()
            v_scroll = self.scroll_area.verticalScrollBar()
            h_scroll.setValue(h_scroll.value() - dx)
            v_scroll.setValue(v_scroll.value() - dy)

            self.pan_start_pos = current_pos
        elif self.drawing and self.start_point:
            # 드래그 중일 때 end_point 업데이트 (위젯 좌표로 저장)
            self.end_point = event.pos()
            self.update_image_display()

        # 항상 십자선 업데이트 (편집 모드가 아니고, 드래그/패닝 중이 아닐 때)
        if not self.edit_mode and not self.drawing and not self.panning:
            self.update_image_display()

    def eventFilter(self, obj, event):
        """이미지 레이블의 마우스 이벤트 필터"""
        if obj == self.image_label:
            if event.type() == QEvent.MouseMove:
                # 이미지가 로드되지 않은 경우 처리하지 않음
                if not self.image_list or not self.original_pixmap:
                    return super().eventFilter(obj, event)

                # 이미지 레이블 위에서 마우스 이동 시
                mouse_event = event
                # 이미지 레이블의 로컬 좌표를 위젯 좌표로 변환
                global_pos = self.image_label.mapToGlobal(mouse_event.pos())
                widget_pos = self.mapFromGlobal(global_pos)
                self.current_mouse_pos = widget_pos
                # 십자선 업데이트 (편집 모드가 아니고, 드래그/패닝 중이 아닐 때)
                if not self.edit_mode and not self.drawing and not self.panning:
                    self.update_image_display()
        return super().eventFilter(obj, event)

    def mouseReleaseEvent(self, event):
        # 이미지가 로드되지 않은 경우 처리하지 않음
        if not self.image_list or not self.original_pixmap:
            return

        # current_index 범위 체크
        if self.current_index < 0 or self.current_index >= len(self.image_list):
            return

        if self.panning and event.button() == Qt.MiddleButton:
            # 이미지 패닝 종료 (마우스 휠)
            self.panning = False
            self.pan_start_pos = None
            self.image_label.setCursor(Qt.CrossCursor)
        elif self.edit_mode and self.drag_handle and event.button() == Qt.LeftButton:
            # 편집 모드에서 드래그 종료
            current_pos = self.get_image_coordinates(event.pos())
            self.finish_box_edit(current_pos)
            self.drag_handle = None
            self.drag_start_pos = None
            self.drag_start_box = None
            self.image_label.setCursor(Qt.ArrowCursor)
        elif self.drawing and event.button() == Qt.LeftButton and self.start_point:
            # 마우스 좌표를 이미지 좌표로 변환
            # start_point와 event.pos() 모두 위젯 좌표이므로 동일하게 변환
            start_x, start_y = self.get_image_coordinates(self.start_point)
            end_x, end_y = self.get_image_coordinates(event.pos())

            # 좌표 정규화
            x1 = min(start_x, end_x)
            y1 = min(start_y, end_y)
            x2 = max(start_x, end_x)
            y2 = max(start_y, end_y)

            w = x2 - x1
            h = y2 - y1

            # 너무 작은 영역은 무시
            if w < 5 or h < 5:
                self.drawing = False
                self.start_point = None
                self.end_point = None
                self.image_label.setCursor(Qt.ArrowCursor)
                return

            # YOLO 형식으로 변환 (0~1 사이의 값)
            if not self.original_pixmap:
                return
            img_width = self.original_pixmap.width()
            img_height = self.original_pixmap.height()

            x_center = (x1 + w / 2) / img_width
            y_center = (y1 + h / 2) / img_height
            w_norm = w / img_width
            h_norm = h / img_height

            # 클래스 0으로 자동 설정하고 바로 저장
            # 새로운 형식: cls x y w h (k1 k2 vis)x5
            label = [
                self.selected_class,
                x_center,
                y_center,
                w_norm,
                h_norm,
            ]  # 선택된 클래스 사용

            ## kw 수정 라벨 길이, 키포인트 추가 / 제거
            # k1 k2 vis x5 추가 (모두 0으로 초기화)
            # for _ in range(5):
            #     label.extend([0.0, 0.0, 0.0])  # k1, k2, vis

            if 0 <= self.current_index < len(self.image_list):
                image_path = self.image_list[self.current_index]
                # 실행 취소를 위한 상태 저장
                self.save_state_for_undo()
                if image_path not in self.labels:
                    self.labels[image_path] = []
                self.labels[image_path].append(label)  # 단순히 라벨 추가
                self.save_label(image_path)
                self.show_image()

                self.drawing = False
                self.start_point = None
                self.end_point = None
                self.image_label.setCursor(Qt.CrossCursor)
            else:
                self.drawing = False
                self.start_point = None
                self.end_point = None
                self.image_label.setCursor(Qt.CrossCursor)

    def scale_down_bbox(self):
        if not self.image_list:
            return

        image_path = self.image_list[self.current_index]
        if image_path not in self.labels or not self.labels[image_path]:
            return

        # 실행 취소를 위한 상태 저장
        self.save_state_for_undo()

        # 모든 바운딩 박스의 크기를 0.05만큼 줄임
        for i in range(len(self.labels[image_path])):
            if (
                len(self.labels[image_path][i]) == 20
            ):  # 새로운 형식: cls x y w h (k1 k2 vis)x5
                class_id, x, y, w, h = self.labels[image_path][i][0:5]
                # 너비와 높이를 0.05만큼 줄임 (최소 0.01 유지)
                new_w = max(0.01, w - w * 0.01)
                new_h = max(0.01, h - h * 0.05)
                # 새로운 형식 유지하면서 업데이트
                self.labels[image_path][i][3] = new_w  # w 업데이트
                self.labels[image_path][i][4] = new_h  # h 업데이트
            elif len(self.labels[image_path][i]) >= 6:
                class_id, x, y, w, h, conf = self.labels[image_path][i]
                # 너비와 높이를 0.05만큼 줄임 (최소 0.01 유지)
                new_w = max(0.01, w - w * 0.01)
                new_h = max(0.01, h - h * 0.05)
                self.labels[image_path][i] = [class_id, x, y, new_w, new_h, conf]
            else:
                class_id, x, y, w, h = self.labels[image_path][i]
                # 너비와 높이를 0.05만큼 줄임 (최소 0.01 유지)
                new_w = max(0.01, w - w * 0.01)
                new_h = max(0.01, h - h * 0.05)
                self.labels[image_path][i] = [class_id, x, y, new_w, new_h]

        # 라벨 파일 저장
        self.save_label(image_path)
        # 이미지 다시 표시
        self.show_image()

    def delete_current_label(self):
        if not self.image_list:
            return

        image_path = self.image_list[self.current_index]
        if image_path not in self.labels or not self.labels[image_path]:
            return

        # 실행 취소를 위한 상태 저장
        self.save_state_for_undo()

        if self.selected_label_index is not None:
            # 선택된 라벨만 삭제
            self.labels[image_path].pop(self.selected_label_index)
            self.selected_label_index = None
        else:
            # 라벨 선택 모드 활성화
            self.label_selection_mode = True
            self.image_label.setCursor(Qt.PointingHandCursor)

        self.save_label(image_path)
        self.show_image()

    def clear_all_labels(self):
        """현재 이미지의 모든 라벨을 삭제"""
        if not self.image_list:
            return

        image_path = self.image_list[self.current_index]
        if image_path not in self.labels or not self.labels[image_path]:
            return

        # 실행 취소를 위한 상태 저장
        self.save_state_for_undo()

        # 모든 라벨 삭제 (확인 다이얼로그 없이 바로 실행)
        label_count = len(self.labels[image_path])
        self.labels[image_path] = []
        self.selected_label_index = None
        self.label_selection_mode = False
        self.editing_label_index = None
        self.edit_mode = False
        self.edit_mode_button.setChecked(False)
        self.erase_mode = False
        self.erase_mode_button.setChecked(False)
        self.image_label.setCursor(Qt.CrossCursor)

        # 라벨 파일 저장 및 이미지 다시 표시
        self.save_label(image_path)
        self.show_image()

        print(f"Cleared all {label_count} labels from {os.path.basename(image_path)}")

    def toggle_edit_mode(self):
        # 박스 크기 조정 플래그 리셋
        if hasattr(self, "_box_size_adjusting"):
            delattr(self, "_box_size_adjusting")
        """편집 모드 토글"""
        # 편집 모드 종료 시 현재 편집 중인 박스가 있으면 저장
        if self.edit_mode and self.editing_label_index is not None:
            if self.image_list:
                image_path = self.image_list[self.current_index]
                if image_path in self.labels:
                    self.save_label(image_path)
                    print("Saved label changes before exiting edit mode")

        # 박스 크기 조정 플래그 리셋
        if hasattr(self, "_box_size_adjusting"):
            delattr(self, "_box_size_adjusting")

        self.edit_mode = not self.edit_mode
        self.edit_mode_button.setChecked(self.edit_mode)

        if self.edit_mode:
            self.drawing = False
            self.label_selection_mode = False
            self.erase_mode = False  # 편집 모드와 충돌 방지
            self.erase_mode_button.setChecked(False)
            self.image_label.setCursor(Qt.ArrowCursor)
            print("Edit mode: ON - Click on a box to select and edit it")
        else:
            self.editing_label_index = None
            self.drag_handle = None
            self.drag_start_pos = None
            self.drag_start_box = None
            if not self.erase_mode:
                self.image_label.setCursor(Qt.CrossCursor)
            print("Edit mode: OFF")

        # 편집 모드 표시 업데이트
        self.update_edit_mode_display()
        self.show_image()

    def update_edit_mode_display(self):
        """편집 모드일 때 테두리와 텍스트 표시 업데이트"""
        if self.edit_mode:
            # 녹색 테두리 표시
            self.scroll_area.setStyleSheet("QScrollArea { border: 3px solid green; }")
            # "Edit ON" 텍스트 표시
            self.edit_mode_label.show()
            # 레이블 위치 조정 (스크롤 영역 위쪽 중앙)
            self.update_edit_mode_label_position()
            # 삭제 모드 레이블 숨김
            self.erase_mode_label.hide()
        else:
            # 편집 모드가 아닐 때는 삭제 모드 확인
            if not self.erase_mode:
                # 테두리 제거
                self.scroll_area.setStyleSheet("")
                # 텍스트 숨김
                self.edit_mode_label.hide()
            else:
                # 삭제 모드일 때는 삭제 모드 표시
                self.update_erase_mode_display()

    def update_edit_mode_label_position(self):
        """편집 모드 레이블 위치 업데이트"""
        if not self.edit_mode:
            return

        # 스크롤 영역의 위치와 크기 가져오기
        scroll_rect = self.scroll_area.geometry()
        # 스크롤 영역 위쪽 중앙에 배치
        label_width = 100
        label_height = 30
        x = scroll_rect.x() + (scroll_rect.width() - label_width) // 2
        y = scroll_rect.y() + 10  # 위쪽에서 10px 떨어진 위치

        self.edit_mode_label.setGeometry(x, y, label_width, label_height)

    def toggle_erase_mode(self):
        """삭제 모드 토글"""
        self.erase_mode = not self.erase_mode
        self.erase_mode_button.setChecked(self.erase_mode)

        if self.erase_mode:
            # 삭제 모드 ON
            self.edit_mode = False  # 편집 모드와 충돌 방지
            self.edit_mode_button.setChecked(False)
            self.drawing = False
            self.label_selection_mode = False
            if self.erase_cursor:
                self.image_label.setCursor(self.erase_cursor)
            print("Erase mode: ON - Click on a box to delete it")
        else:
            # 삭제 모드 OFF
            self.image_label.setCursor(Qt.CrossCursor)
            print("Erase mode: OFF")

        # 삭제 모드 표시 업데이트
        self.update_erase_mode_display()
        self.show_image()

    def erase_label_at_position(self, click_pos):
        """클릭 위치의 라벨 삭제"""
        if not self.image_list or not self.original_pixmap:
            return

        if self.current_index < 0 or self.current_index >= len(self.image_list):
            return

        image_path = self.image_list[self.current_index]
        if image_path not in self.labels or not self.labels[image_path]:
            return

        click_x, click_y = click_pos
        img_width = self.original_pixmap.width()
        img_height = self.original_pixmap.height()

        # 뒤에서부터 확인 (나중에 그린 박스가 우선)
        for i in range(len(self.labels[image_path]) - 1, -1, -1):
            label = self.labels[image_path][i]
            x1, y1, x2, y2 = self.get_box_coordinates(label, img_width, img_height)

            if x1 <= click_x <= x2 and y1 <= click_y <= y2:
                # 실행 취소를 위한 상태 저장
                self.save_state_for_undo()
                # 라벨 삭제
                self.labels[image_path].pop(i)
                self.save_label(image_path)
                self.show_image()
                print(f"Deleted label {i}")
                return

    def update_erase_mode_display(self):
        """삭제 모드일 때 테두리와 텍스트 표시 업데이트"""
        if self.erase_mode:
            # 빨간색 테두리 표시
            self.scroll_area.setStyleSheet("QScrollArea { border: 3px solid red; }")
            # "Erase ON" 텍스트 표시
            self.erase_mode_label.show()
            # 레이블 위치 조정 (스크롤 영역 위쪽 중앙)
            self.update_erase_mode_label_position()
            # 편집 모드 레이블 숨김
            self.edit_mode_label.hide()
        else:
            # 삭제 모드가 아닐 때는 편집 모드 확인
            if not self.edit_mode:
                # 테두리 제거
                self.scroll_area.setStyleSheet("")
                # 텍스트 숨김
                self.erase_mode_label.hide()
            else:
                # 편집 모드일 때는 편집 모드 표시
                self.update_edit_mode_display()

    def update_erase_mode_label_position(self):
        """삭제 모드 레이블 위치 업데이트"""
        if not self.erase_mode:
            return

        # 스크롤 영역의 위치와 크기 가져오기
        scroll_rect = self.scroll_area.geometry()
        # 스크롤 영역 위쪽 중앙에 배치
        label_width = 100
        label_height = 30
        x = scroll_rect.x() + (scroll_rect.width() - label_width) // 2
        y = scroll_rect.y() + 10  # 위쪽에서 10px 떨어진 위치

        self.erase_mode_label.setGeometry(x, y, label_width, label_height)

    def get_box_coordinates(self, label, img_width, img_height):
        """라벨에서 박스 좌표 추출"""
        if not label or len(label) < 5:
            return (0, 0, 0, 0)

        try:
            if len(label) == 20:  # 새로운 형식
                class_id, x, y, w, h = label[0:5]
            elif len(label) >= 6:
                class_id, x, y, w, h, conf = label
            else:
                class_id, x, y, w, h = label

            x1 = int((x - w / 2) * img_width)
            y1 = int((y - h / 2) * img_height)
            x2 = int((x + w / 2) * img_width)
            y2 = int((y + h / 2) * img_height)

            return x1, y1, x2, y2
        except (ValueError, IndexError, TypeError):
            return (0, 0, 0, 0)

    def get_handle_at_position(self, click_x, click_y, x1, y1, x2, y2):
        """클릭 위치에 있는 핸들 반환"""
        hs = self.handle_size
        mid_x = (x1 + x2) // 2
        mid_y = (y1 + y2) // 2

        # 모서리 핸들 체크
        if abs(click_x - x1) <= hs and abs(click_y - y1) <= hs:
            return "nw"  # 좌상단
        if abs(click_x - x2) <= hs and abs(click_y - y1) <= hs:
            return "ne"  # 우상단
        if abs(click_x - x1) <= hs and abs(click_y - y2) <= hs:
            return "sw"  # 좌하단
        if abs(click_x - x2) <= hs and abs(click_y - y2) <= hs:
            return "se"  # 우하단

        # 변 중앙 핸들 체크
        if abs(click_x - mid_x) <= hs and abs(click_y - y1) <= hs:
            return "n"  # 상단 중앙
        if abs(click_x - mid_x) <= hs and abs(click_y - y2) <= hs:
            return "s"  # 하단 중앙
        if abs(click_x - x1) <= hs and abs(click_y - mid_y) <= hs:
            return "w"  # 좌측 중앙
        if abs(click_x - x2) <= hs and abs(click_y - mid_y) <= hs:
            return "e"  # 우측 중앙

        # 박스 내부인지 체크 (이동용)
        if x1 <= click_x <= x2 and y1 <= click_y <= y2:
            return "move"

        return None

    def find_label_at_position(self, click_x, click_y):
        """클릭 위치에 있는 라벨 찾기"""
        if not self.image_list or not self.original_pixmap:
            return None

        if self.current_index < 0 or self.current_index >= len(self.image_list):
            return None

        image_path = self.image_list[self.current_index]
        if image_path not in self.labels or not self.labels[image_path]:
            return None

        img_width = self.original_pixmap.width()
        img_height = self.original_pixmap.height()

        # 뒤에서부터 확인 (나중에 그린 박스가 우선)
        for i in range(len(self.labels[image_path]) - 1, -1, -1):
            if i < 0 or i >= len(self.labels[image_path]):
                continue
            label = self.labels[image_path][i]
            if not label or len(label) < 5:
                continue
            x1, y1, x2, y2 = self.get_box_coordinates(label, img_width, img_height)

            if x1 <= click_x <= x2 and y1 <= click_y <= y2:
                return i

        return None

    def update_box_during_drag(self, current_pos):
        """드래그 중 박스 업데이트 (실시간 미리보기)"""
        if (
            not self.image_list
            or not self.original_pixmap
            or self.editing_label_index is None
            or not self.drag_handle
            or not self.drag_start_pos
            or not self.drag_start_box
        ):
            return

        if self.current_index < 0 or self.current_index >= len(self.image_list):
            return

        image_path = self.image_list[self.current_index]
        if image_path not in self.labels:
            return

        if self.editing_label_index < 0 or self.editing_label_index >= len(
            self.labels[image_path]
        ):
            return

        label = self.labels[image_path][self.editing_label_index]
        img_width = self.original_pixmap.width()
        img_height = self.original_pixmap.height()

        # 원본 박스 좌표 사용
        x1, y1, x2, y2 = self.drag_start_box

        # 드래그 거리 계산
        dx = current_pos[0] - self.drag_start_pos[0]
        dy = current_pos[1] - self.drag_start_pos[1]

        # 핸들에 따라 박스 수정
        if self.drag_handle == "move":
            # 박스 전체 이동
            x1 = max(0, min(x1 + dx, img_width - 1))
            y1 = max(0, min(y1 + dy, img_height - 1))
            x2 = max(0, min(x2 + dx, img_width - 1))
            y2 = max(0, min(y2 + dy, img_height - 1))
        elif self.drag_handle == "nw":
            x1 = max(0, min(x1 + dx, x2 - 5))
            y1 = max(0, min(y1 + dy, y2 - 5))
        elif self.drag_handle == "ne":
            x2 = max(x1 + 5, min(x2 + dx, img_width - 1))
            y1 = max(0, min(y1 + dy, y2 - 5))
        elif self.drag_handle == "sw":
            x1 = max(0, min(x1 + dx, x2 - 5))
            y2 = max(y1 + 5, min(y2 + dy, img_height - 1))
        elif self.drag_handle == "se":
            x2 = max(x1 + 5, min(x2 + dx, img_width - 1))
            y2 = max(y1 + 5, min(y2 + dy, img_height - 1))
        elif self.drag_handle == "n":
            y1 = max(0, min(y1 + dy, y2 - 5))
        elif self.drag_handle == "s":
            y2 = max(y1 + 5, min(y2 + dy, img_height - 1))
        elif self.drag_handle == "w":
            x1 = max(0, min(x1 + dx, x2 - 5))
        elif self.drag_handle == "e":
            x2 = max(x1 + 5, min(x2 + dx, img_width - 1))

        # YOLO 형식으로 변환
        w = x2 - x1
        h = y2 - y1
        x_center = (x1 + w / 2) / img_width
        y_center = (y1 + h / 2) / img_height
        w_norm = w / img_width
        h_norm = h / img_height

        # 라벨 임시 업데이트 (드래그 중에는 저장하지 않음)
        if len(label) == 20:  # 새로운 형식
            class_id = label[0]
            self.labels[image_path][self.editing_label_index] = [
                class_id,
                x_center,
                y_center,
                w_norm,
                h_norm,
            ] + label[
                5:
            ]  # 키포인트 정보 유지
        elif len(label) >= 6:
            class_id = label[0]
            conf = label[5]
            self.labels[image_path][self.editing_label_index] = [
                class_id,
                x_center,
                y_center,
                w_norm,
                h_norm,
                conf,
            ]
        else:
            class_id = label[0]
            self.labels[image_path][self.editing_label_index] = [
                class_id,
                x_center,
                y_center,
                w_norm,
                h_norm,
            ]

        # 실시간으로 화면만 업데이트 (파일 저장은 하지 않음, 라벨 재로드도 스킵)
        self.show_image(skip_label_reload=True)

    def finish_box_edit(self, current_pos):
        """박스 편집 완료 및 저장"""
        if (
            not self.image_list
            or self.editing_label_index is None
            or not self.drag_handle
            or not self.drag_start_pos
        ):
            return

        # 실행 취소를 위한 상태 저장 (편집 시작 시점의 상태)
        # 드래그 시작 시점에 이미 저장되어 있으므로 여기서는 저장하지 않음
        # 대신 드래그 시작 시점에 저장하도록 mousePressEvent에서 처리

        # 마지막 위치로 한 번 더 업데이트 (드래그 종료 시 정확한 위치 보장)
        self.update_box_during_drag(current_pos)

        # 저장
        image_path = self.image_list[self.current_index]
        self.save_label(image_path)

    def handle_left_key(self):
        """좌측 화살표 키 처리"""
        if self.edit_mode and self.editing_label_index is not None:
            # 편집 모드: width 축소
            self.adjust_box_size(-1, 0)
        else:
            # 일반 모드: 이전 이미지로 이동
            self.show_previous_image()

    def handle_right_key(self):
        """우측 화살표 키 처리"""
        if self.edit_mode and self.editing_label_index is not None:
            # 편집 모드: width 확대
            self.adjust_box_size(1, 0)
        else:
            # 일반 모드: 다음 이미지로 이동
            self.show_next_image()

    def handle_comma_key(self):
        """콤마(,) 키 처리 - 이전 이미지로 이동"""
        if not self.edit_mode:
            # 편집 모드가 아닐 때만 이미지 이동
            self.show_previous_image()

    def handle_backtick_key(self):
        """백틱(`) 키 처리 - 다음 이미지로 이동"""
        if not self.edit_mode:
            # 편집 모드가 아닐 때만 이미지 이동
            self.show_next_image()

    def handle_up_key(self):
        """위쪽 화살표 키 처리"""
        if self.edit_mode and self.editing_label_index is not None:
            # 편집 모드: height 축소
            self.adjust_box_size(0, -1)

    def handle_down_key(self):
        """아래쪽 화살표 키 처리"""
        if self.edit_mode and self.editing_label_index is not None:
            # 편집 모드: height 확대
            self.adjust_box_size(0, 1)

    def adjust_box_size(self, width_delta, height_delta):
        """편집 모드에서 선택된 박스 크기 조정 (1픽셀 단위)"""
        if (
            not self.image_list
            or not self.original_pixmap
            or self.editing_label_index is None
        ):
            return

        if self.current_index < 0 or self.current_index >= len(self.image_list):
            return

        image_path = self.image_list[self.current_index]
        if image_path not in self.labels:
            return

        if self.editing_label_index < 0 or self.editing_label_index >= len(
            self.labels[image_path]
        ):
            return

        label = self.labels[image_path][self.editing_label_index]
        img_width = self.original_pixmap.width()
        img_height = self.original_pixmap.height()

        # 현재 박스 좌표 가져오기
        x1, y1, x2, y2 = self.get_box_coordinates(label, img_width, img_height)

        # 크기 조정 (1픽셀 단위로 조정)
        step_size = 1.0

        # width 조정
        if width_delta > 0:
            # 확대: 양쪽으로 확장
            x1 = max(0, x1 - step_size)
            x2 = min(img_width - 1, x2 + step_size)
        elif width_delta < 0:
            # 축소: 양쪽으로 축소
            x1 = min(x1 + step_size, (x1 + x2) / 2)
            x2 = max(x2 - step_size, (x1 + x2) / 2)

        # height 조정
        if height_delta > 0:
            # 확대: 위아래로 확장
            y1 = max(0, y1 - step_size)
            y2 = min(img_height - 1, y2 + step_size)
        elif height_delta < 0:
            # 축소: 위아래로 축소
            y1 = min(y1 + step_size, (y1 + y2) / 2)
            y2 = max(y2 - step_size, (y1 + y2) / 2)

        # 최소 크기 보장
        if x2 - x1 < 5:
            center_x = (x1 + x2) / 2
            x1 = center_x - 2.5
            x2 = center_x + 2.5
        if y2 - y1 < 5:
            center_y = (y1 + y2) / 2
            y1 = center_y - 2.5
            y2 = center_y + 2.5

        # YOLO 형식으로 변환
        w = x2 - x1
        h = y2 - y1
        x_center = (x1 + w / 2) / img_width
        y_center = (y1 + h / 2) / img_height
        w_norm = w / img_width
        h_norm = h / img_height

        # 실행 취소를 위한 상태 저장 (첫 조정 시에만)
        if not hasattr(self, "_box_size_adjusting"):
            self.save_state_for_undo()
            self._box_size_adjusting = True

        # 라벨 업데이트
        if len(label) == 20:  # 새로운 형식
            class_id = label[0]
            self.labels[image_path][self.editing_label_index] = [
                class_id,
                x_center,
                y_center,
                w_norm,
                h_norm,
            ] + label[
                5:
            ]  # 키포인트 정보 유지
        elif len(label) >= 6:
            class_id = label[0]
            conf = label[5]
            self.labels[image_path][self.editing_label_index] = [
                class_id,
                x_center,
                y_center,
                w_norm,
                h_norm,
                conf,
            ]
        else:
            class_id = label[0]
            self.labels[image_path][self.editing_label_index] = [
                class_id,
                x_center,
                y_center,
                w_norm,
                h_norm,
            ]

        # 화면 업데이트
        self.show_image(skip_label_reload=True)

        # 파일 저장
        self.save_label(image_path)

    def toggle_copy_mode(self):
        """복사 모드 토글 (Copy All / Copy Class)"""
        self.copy_mode_all = not self.copy_mode_all
        if self.copy_mode_all:
            self.copy_mode_button.setText("Copy All")
        else:
            self.copy_mode_button.setText("Copy Class")

    def copy_previous_label(self):
        """이전 이미지의 라벨을 현재 이미지로 복사"""
        if not self.image_list or self.current_index == 0:
            QMessageBox.information(
                self,
                "No Previous Image",
                "이전 이미지가 없습니다.",
            )
            return

        # 이전 이미지 경로
        prev_image_path = self.image_list[self.current_index - 1]
        current_image_path = self.image_list[self.current_index]

        # 이전 이미지의 라벨이 메모리에 없으면 파일에서 로드
        if prev_image_path not in self.labels or not self.labels[prev_image_path]:
            if self.label_folder:
                # 라벨 파일에서 로드 시도
                image_name = os.path.splitext(os.path.basename(prev_image_path))[0]
                label_path = os.path.join(self.label_folder, f"{image_name}.txt")

                if os.path.exists(label_path):
                    # 라벨 로드
                    self.load_current_image_label(prev_image_path)
                else:
                    QMessageBox.information(
                        self,
                        "No Labels",
                        "이전 이미지에 라벨이 없습니다.",
                    )
                    return
            else:
                QMessageBox.information(
                    self,
                    "No Labels",
                    "이전 이미지에 라벨이 없습니다.",
                )
                return

        # 이전 이미지에 라벨이 있는지 확인
        if prev_image_path not in self.labels or not self.labels[prev_image_path]:
            QMessageBox.information(
                self,
                "No Labels",
                "이전 이미지에 라벨이 없습니다.",
            )
            return

        # 실행 취소를 위한 상태 저장
        self.save_state_for_undo()

        # 복사 모드에 따라 라벨 복사
        import copy

        if self.copy_mode_all:
            # Copy All: 전체 라벨 복사
            prev_labels = self.labels[prev_image_path]
            copied_labels = []
            for label in prev_labels:
                # 라벨을 깊은 복사 (리스트 복사)
                if isinstance(label, list):
                    copied_labels.append(label.copy())
                else:
                    copied_labels.append(label)
            self.labels[current_image_path] = copied_labels
        else:
            # Copy Class: 선택된 클래스만 복사
            prev_labels = self.labels[prev_image_path]
            filtered_labels = []
            for label in prev_labels:
                # 선택된 클래스와 일치하는 라벨만 복사
                if (
                    isinstance(label, list)
                    and len(label) > 0
                    and label[0] == self.selected_class
                ):
                    if isinstance(label, list):
                        filtered_labels.append(label.copy())
                    else:
                        filtered_labels.append(label)

            # 현재 이미지에 기존 라벨이 있으면 병합 (같은 클래스는 덮어쓰기)
            if current_image_path not in self.labels:
                self.labels[current_image_path] = []

            # 기존 라벨에서 선택된 클래스 제거
            existing_labels = self.labels[current_image_path]
            self.labels[current_image_path] = [
                label
                for label in existing_labels
                if not isinstance(label, list)
                or len(label) == 0
                or label[0] != self.selected_class
            ]

            # 필터링된 라벨 추가
            self.labels[current_image_path].extend(filtered_labels)

        # 라벨 파일 저장
        self.save_label(current_image_path)

        # 이미지 다시 표시
        self.show_image()

        # 복사된 라벨 개수 출력
        if self.copy_mode_all:
            label_count = len(self.labels[current_image_path])
            print(
                f"Copied {label_count} label(s) from previous image to current image (Copy All mode)"
            )
        else:
            label_count = len(
                [
                    l
                    for l in self.labels[current_image_path]
                    if isinstance(l, list)
                    and len(l) > 0
                    and l[0] == self.selected_class
                ]
            )
            print(
                f"Copied {label_count} label(s) of class {self.selected_class} from previous image to current image (Copy Class mode)"
            )

    def find_clicked_label(self, click_pos):
        image_path = self.image_list[self.current_index]
        if image_path not in self.labels:
            return

        img_width = self.original_pixmap.width()
        img_height = self.original_pixmap.height()
        click_x, click_y = click_pos

        # 각 라벨 박스를 확인하여 클릭된 위치가 박스 안에 있는지 검사
        for i, label in enumerate(self.labels[image_path]):
            if len(label) == 20:  # 새로운 형식: cls x y w h (k1 k2 vis)x5
                class_id, x, y, w, h = label[0:5]
            elif len(label) >= 6:
                class_id, x, y, w, h, conf = label
            else:
                class_id, x, y, w, h = label

            # YOLO 좌표를 픽셀 좌표로 변환
            x1 = int((x - w / 2) * img_width)
            y1 = int((y - h / 2) * img_height)
            x2 = int((x + w / 2) * img_width)
            y2 = int((y + h / 2) * img_height)

            # 클릭 위치가 박스 안에 있는지 확인
            if x1 <= click_x <= x2 and y1 <= click_y <= y2:
                self.selected_label_index = i
                self.label_selection_mode = False
                self.image_label.setCursor(Qt.ArrowCursor)
                self.delete_current_label()
                break

    def save_label(self, image_path):
        if not self.label_folder:
            return

        image_name = os.path.splitext(os.path.basename(image_path))[0]
        label_path = os.path.join(self.label_folder, f"{image_name}.txt")

        try:
            with open(label_path, "w") as f:
                for label in self.labels[image_path]:
                    # 기존 라벨 형식 유지 (5개 또는 6개 값)
                    if len(label) == 5:  # cls x y w h
                        f.write(" ".join(map(str, label[:5])) + "\n")
                    elif len(label) == 6:  # cls x y w h conf
                        f.write(" ".join(map(str, label[:6])) + "\n")
                    elif len(label) == 20:  # 새로운 형식: cls x y w h (k1 k2 vis)x5
                        f.write(" ".join(map(str, label)) + "\n")
                    else:
                        # 기타 형식은 처음 5개 값만 저장
                        f.write(" ".join(map(str, label[:5])) + "\n")
            print(f"Label file saved: {label_path}")
        except Exception as e:
            print(f"Error saving label file {label_path}: {e}")

    def select_folder(self):
        start, ok1 = QInputDialog.getInt(
            self, "Input Start Index", "Enter start index:", 0, 0, 100000, 1
        )
        if not ok1:
            return

        end, ok2 = QInputDialog.getInt(
            self, "Input End Index", "Enter end index:", 1000, 0, 1000000, 1
        )
        if not ok2:
            return

        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder_path:
            self.load_images(folder_path, start, end)

    def show_previous_image(self):
        if self.image_list and self.current_index > 0:
            self.current_index -= 1
            # 이미지 변경 시 실행 취소 스택 초기화
            self.undo_stack = []
            self.show_image()

    def show_next_image(self):
        if self.image_list and self.current_index < len(self.image_list) - 1:
            self.current_index += 1
            # 이미지 변경 시 실행 취소 스택 초기화
            self.undo_stack = []
            self.show_image()

    def update_image_list_widget(self):
        self.image_list_widget.clear()
        for path in self.image_list:
            item = QListWidgetItem(os.path.basename(path))
            self.image_list_widget.addItem(item)

        # 현재 선택된 이미지 강조 표시
        if self.image_list:
            self.image_list_widget.setCurrentRow(self.current_index)

    def on_image_list_item_clicked(self, item):
        index = self.image_list_widget.row(item)
        if 0 <= index < len(self.image_list):
            self.current_index = index
            # 이미지 변경 시 실행 취소 스택 초기화
            self.undo_stack = []
            self.show_image()

    def select_label_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Label Folder")
        if folder_path:
            self.label_folder = folder_path
            self.load_labels()

    def load_current_image_label(self, image_path):
        """현재 이미지의 라벨 파일만 다시 로드 (새로 생성된 파일 인식용)"""
        if not self.label_folder:
            return

        image_name = os.path.splitext(os.path.basename(image_path))[0]
        label_path = os.path.join(self.label_folder, f"{image_name}.txt")

        if os.path.exists(label_path):
            try:
                with open(label_path, "r") as f:
                    labels = []
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 3:  # 최소한 class와 좌표 2개는 필요
                            try:
                                class_id = int(parts[0])
                                coords = [float(x) for x in parts[1:]]

                                if len(coords) >= 4:
                                    if len(coords) == 4:  # cls x y w h
                                        labels.append([class_id] + coords[:4])
                                    elif len(coords) == 5:  # cls x y w h conf
                                        labels.append([class_id] + coords[:5])
                                    elif (
                                        len(coords) == 19
                                    ):  # 새로운 형식: cls x y w h (k1 k2 vis)x5
                                        labels.append([class_id] + coords[:19])
                                    else:
                                        # 기본적으로 처음 4개 좌표만 사용
                                        labels.append([class_id] + coords[:4])
                                else:
                                    print(
                                        f"Skipping invalid label format in {label_path}: {line} - not enough coordinates, got {len(coords)}"
                                    )
                            except ValueError:
                                print(f"Invalid label format in {label_path}: {line}")
                    self.labels[image_path] = labels
            except Exception as e:
                print(f"Error reading label file {label_path}: {e}")
                self.labels[image_path] = []
        else:
            # 라벨 파일이 없으면 빈 리스트로 설정
            self.labels[image_path] = []

    def load_labels(self):
        if not self.label_folder:
            return

        self.labels = {}
        for image_path in self.image_list:
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            label_path = os.path.join(self.label_folder, f"{image_name}.txt")

            if os.path.exists(label_path):
                try:
                    with open(label_path, "r") as f:
                        labels = []
                        original_content = f.read()  # 원본 내용 저장
                        f.seek(0)  # 파일 포인터를 처음으로 되돌림
                        needs_save = False  # 변환 필요 여부 체크
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 3:  # 최소한 class와 좌표 2개는 필요
                                try:
                                    class_id = int(parts[0])
                                    coords = [float(x) for x in parts[1:]]

                                    # # 폴리곤 좌표인 경우 (좌표가 7개 이상)
                                    # if len(coords) >= 7:
                                    #     needs_save = True  # 변환이 필요한 경우 표시
                                    #     # x, y 좌표 분리
                                    #     x_coords = coords[::2]
                                    #     y_coords = coords[1::2]

                                    #     # 최소/최대 좌표 찾기
                                    #     x_min = min(x_coords)
                                    #     x_max = max(x_coords)
                                    #     y_min = min(y_coords)
                                    #     y_max = max(y_coords)

                                    #     # 바운딩 박스 중심과 크기 계산
                                    #     x_center = (x_min + x_max) / 2
                                    #     y_center = (y_min + y_max) / 2
                                    #     width = x_max - x_min
                                    #     height = y_max - y_min

                                    #     # YOLO 형식으로 변환
                                    #     labels.append(
                                    #         [
                                    #             class_id,
                                    #             x_center,
                                    #             y_center,
                                    #             width,
                                    #             height,
                                    #         ]
                                    #     )
                                    if (
                                        len(coords) >= 4
                                    ):  # confidence score가 포함된 경우 (x,y,w,h + 옵션: conf)
                                        # confidence score를 제외한 좌표만 사용
                                        # labels.append([class_id] + coords[:4])
                                        if len(coords) == 4:  # cls x y w h
                                            labels.append([class_id] + coords[:4])
                                        elif len(coords) == 5:  # cls x y w h conf
                                            labels.append([class_id] + coords[:5])
                                        elif (
                                            len(coords) == 19
                                        ):  # 새로운 형식: cls x y w h (k1 k2 vis)x5
                                            labels.append([class_id] + coords[:19])
                                        else:
                                            # 기본적으로 처음 4개 좌표만 사용
                                            labels.append([class_id] + coords[:4])
                                    else:
                                        print(
                                            f"Skipping invalid label format in {label_path}: {line} - not enough coordinates, got {len(coords)}"
                                        )
                                except ValueError:
                                    print(
                                        f"Invalid label format in {label_path}: {line}"
                                    )
                        self.labels[image_path] = labels

                        # 실제로 변환이 발생한 경우에만 저장
                        if needs_save:
                            self.save_label(image_path)
                            print(f"Converted and saved label file: {label_path}")

                except Exception as e:
                    print(f"Error reading label file {label_path}: {e}")
                    self.labels[image_path] = []
            else:
                self.labels[image_path] = []

    def start_drawing(self):
        """라벨 그리기 모드 시작 (A 키 - 이제는 선택적 사용)"""
        # A 키를 눌러도 바로 그리기 모드로 전환 (실제로는 클릭 앤 드래그로 바로 가능)
        # 이 메서드는 호환성을 위해 유지
        pass

    def wheelEvent(self, event: QWheelEvent):
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

    def select_suit(self, suit_index):
        """문양 선택 (z/x/c/v)"""
        suits = ["♥", "♦", "♣", "♠"]
        if 0 <= suit_index < len(suits):
            self.selected_suit = suit_index
            print(
                f"Suit selected: {suits[suit_index]} (Press number/letter key to select rank)"
            )

    def select_rank(self, rank_index):
        """숫자/문자 선택 (1-0, j, k, q)"""
        suits = ["♥", "♦", "♣", "♠"]
        ranks = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]

        if 0 <= rank_index < len(ranks):
            if self.selected_suit is not None:
                # 문양이 선택된 경우: 클래스 ID = (suit_index * 13 + rank_index) + CLASS_OFFSET
                class_id = self.selected_suit * 13 + rank_index + self.CLASS_OFFSET
                self.select_class_by_id(class_id)
            else:
                print("Please select suit first (z/x/c/v)")

    def select_class_by_id(self, class_id):
        """클래스 ID로 클래스 선택"""
        suits = ["♥", "♦", "♣", "♠"]
        ranks = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]

        # edit 모드에서 박스가 선택된 상태면 클래스 변경
        if self.edit_mode and self.editing_label_index is not None:
            if (
                self.image_list
                and 0 <= self.current_index < len(self.image_list)
                and self.original_pixmap
            ):
                image_path = self.image_list[self.current_index]
                if image_path in self.labels and 0 <= self.editing_label_index < len(
                    self.labels[image_path]
                ):
                    # 실행 취소를 위한 상태 저장
                    self.save_state_for_undo()

                    # 선택된 박스의 클래스 변경
                    label = self.labels[image_path][self.editing_label_index]
                    if len(label) > 0:
                        label[0] = class_id  # 클래스 ID 변경
                        self.save_label(image_path)
                        self.show_image()
                        print(f"Changed class to {class_id}")

        # class_id에 해당하는 버튼 찾기
        for button in self.class_buttons:
            if self.class_button_group.id(button) == class_id:
                button.setChecked(True)
                self.selected_class = class_id

                # 클래스 ID에 따라 이름 생성 (문양+숫자 순서)
                if class_id == 0:
                    card_name = "chip"
                elif class_id == 1:
                    card_name = "null"
                elif class_id == 2:
                    card_name = "back"
                else:
                    # 카드: class_id = (suit_index * 13 + rank_index) + CLASS_OFFSET
                    adjusted_id = class_id - self.CLASS_OFFSET
                    suit_index = adjusted_id // 13
                    rank_index = adjusted_id % 13
                    if 0 <= suit_index < len(suits) and 0 <= rank_index < len(ranks):
                        card_name = f"{suits[suit_index]}{ranks[rank_index]}"
                    else:
                        card_name = f"Class_{class_id}"

                self.selected_class_label.setText(
                    f"Selected: {card_name} (Class {self.selected_class})"
                )
                # 문양 선택 상태 초기화
                self.selected_suit = None
                break

    def on_class_selected(self, button):
        """클래스 선택 시 호출되는 메서드"""
        suits = ["♥", "♦", "♣", "♠"]
        ranks = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]

        self.selected_class = self.class_button_group.id(button)

        # 클래스 ID에 따라 이름 생성 (문양+숫자 순서)
        if self.selected_class == 0:
            card_name = "chip"
        elif self.selected_class == 1:
            card_name = "null"
        elif self.selected_class == 2:
            card_name = "back"
        else:
            # 카드: class_id = (suit_index * 13 + rank_index) + CLASS_OFFSET
            adjusted_id = self.selected_class - self.CLASS_OFFSET
            suit_index = adjusted_id // 13
            rank_index = adjusted_id % 13
            if 0 <= suit_index < len(suits) and 0 <= rank_index < len(ranks):
                card_name = f"{suits[suit_index]}{ranks[rank_index]}"
            else:
                card_name = f"Class_{self.selected_class}"

        self.selected_class_label.setText(
            f"Selected: {card_name} (Class {self.selected_class})"
        )
        # 문양 선택 상태 초기화
        self.selected_suit = None

    def toggle_class_name_display(self):
        """클래스 이름 표시 토글"""
        self.show_class_name = not self.show_class_name
        self.show_class_name_button.setChecked(self.show_class_name)
        if self.show_class_name:
            self.show_class_name_button.setText("ON")
            print("Class name display: ON")
        else:
            self.show_class_name_button.setText("OFF")
            print("Class name display: OFF")
        self.show_image()

    def delete_correct_image(self):
        """이미지 삭제 기능 (현재는 사용하지 않음)"""
        pass

    def load_existing_model(self):
        """기존에 학습된 모델을 로드"""
        try:
            # 모델 파일 선택 다이얼로그
            model_path, _ = QFileDialog.getOpenFileName(
                self,
                "Select ONNX Model",
                "./models",  # 기본 모델 폴더
                "ONNX Model (*.onnx);;All Files (*)",
            )

            if not model_path:
                return

            # Device 선택 다이얼로그
            device_items = ["CPU", "GPU (CUDA)"]
            device, ok = QInputDialog.getItem(
                self,
                "Select Device",
                "사용할 디바이스를 선택하세요:",
                device_items,
                0,
                False,
            )

            if not ok:
                return

            # ONNX Runtime providers 설정
            if device == "CPU":
                providers = ["CPUExecutionProvider"]
                self.device = "cpu"
            else:  # GPU (CUDA)
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                self.device = "cuda"

            # ONNX 모델 로드
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )
            self.onnx_session = ort.InferenceSession(
                model_path, sess_options=sess_options, providers=providers
            )
            self.model_path = model_path
            self.model_status_label.setText(f"Model: Loaded (Device: {self.device})")
            self.auto_label_button.setEnabled(True)

            QMessageBox.information(
                self,
                "Model Loaded",
                f"ONNX model loaded successfully!\nModel path: {model_path}\nDevice: {self.device}\n\nYou can now use Auto Label Current (G) to add labels to images.",
            )

        except Exception as e:
            QMessageBox.critical(
                self, "Model Load Error", f"Failed to load ONNX model: {str(e)}"
            )

    def auto_label_current_image(self):
        """현재 이미지에 자동 라벨링 적용 (ONNX Runtime 사용)"""
        if not self.onnx_session:
            QMessageBox.warning(
                self,
                "No Model Loaded",
                "Please load an ONNX model first using 'Load Model (L)'.",
            )
            return

        if not self.image_list:
            return

        try:
            image_path = self.image_list[self.current_index]

            # 실행 취소를 위한 상태 저장
            self.save_state_for_undo()

            # 이미지 로드 및 전처리 (한글 경로 지원)
            # OpenCV의 imread는 한글 경로를 제대로 처리하지 못하므로
            # np.fromfile과 cv2.imdecode를 사용
            img_array = np.fromfile(image_path, np.uint8)
            img0 = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img0 is None:
                QMessageBox.critical(
                    self, "Error", f"Failed to load image: {image_path}"
                )
                return

            # letterbox 전처리
            img, ratio, dwdh = letterbox(
                img0, new_shape=(640, 640), auto=False, scaleFill=False
            )
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_tensor = img_rgb.transpose(2, 0, 1)
            img_tensor = np.expand_dims(img_tensor, axis=0).astype(np.float32) / 255.0

            # ONNX 추론
            pred = self.onnx_session.run(None, {"images": img_tensor})
            pred = pred[0].squeeze(0).transpose(1, 0)  # (1, N, 8400) → (8400, N)

            # 박스와 클래스 점수 분리
            boxes = pred[:, :4]
            cls_scores = pred[:, 4:]

            # Sigmoid 적용 (필요한 경우)
            if cls_scores.max() > 1.0 or cls_scores.min() < 0.0:
                cls_scores = 1 / (1 + np.exp(-cls_scores))

            # 각 앵커의 최대 클래스 점수와 클래스 ID 추출
            cls_ids = np.argmax(cls_scores, axis=1)
            scores = cls_scores[np.arange(len(cls_scores)), cls_ids]

            # Confidence threshold (기본값 0.5 사용)
            conf_threshold = 0.5
            mask = scores > conf_threshold
            boxes = boxes[mask]
            scores = scores[mask]
            cls_ids = cls_ids[mask]

            if len(boxes) == 0:
                print(
                    f"Auto Labeling: No objects detected (threshold: {conf_threshold:.2f})"
                )
                return

            # xywh → xyxy 변환
            xyxy = np.zeros_like(boxes)
            xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
            xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
            xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
            xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2

            # 원본 이미지 크기에 맞게 좌표 복원
            for i in range(xyxy.shape[0]):
                xyxy[i, [0, 2]] -= dwdh[0]
                xyxy[i, [1, 3]] -= dwdh[1]
                xyxy[i, :] /= ratio[0]
                xyxy[i, :] = np.clip(
                    xyxy[i, :],
                    0,
                    [img0.shape[1], img0.shape[0], img0.shape[1], img0.shape[0]],
                )

            # NMS
            boxes_xywh = xyxy2xywh(xyxy)
            boxes_list = boxes_xywh.tolist()
            scores_list = scores.tolist()

            indices = cv2.dnn.NMSBoxes(
                bboxes=boxes_list,
                scores=scores_list,
                score_threshold=conf_threshold,
                nms_threshold=0.45,
            )

            if not isinstance(indices, (np.ndarray, list, tuple)) or len(indices) == 0:
                print(f"Auto Labeling: No objects after NMS")
                return

            # 기존 라벨 개수 저장
            old_label_count = len(self.labels.get(image_path, []))

            # 새로운 라벨 추가 (기존 라벨은 유지)
            new_labels = []
            existing_labels = self.labels.get(image_path, []).copy()

            height, width = img0.shape[:2]

            for idx in indices:
                if isinstance(idx, (list, tuple, np.ndarray)):
                    i = idx[0]
                else:
                    i = idx

                x1, y1, x2, y2 = xyxy[i]
                cls = int(cls_ids[i])
                conf = float(scores[i])

                # YOLO 형식으로 변환 (정규화된 좌표)
                x_center = (x1 + x2) / 2 / width
                y_center = (y1 + y2) / 2 / height
                w = (x2 - x1) / width
                h = (y2 - y1) / height

                new_box = [x_center, y_center, w, h]

                # 기존 라벨과 IOU 체크
                should_add = True
                for existing_label in existing_labels:
                    if len(existing_label) >= 5:
                        # 기존 라벨의 좌표 추출
                        if len(existing_label) == 20:  # 새로운 형식
                            existing_box = existing_label[1:5]
                        elif len(existing_label) >= 6:  # confidence 포함
                            existing_box = existing_label[1:5]
                        else:  # 기본 형식
                            existing_box = existing_label[1:5]

                        # IOU 계산
                        iou = calculate_iou(new_box, existing_box)

                        # IOU가 0.7 이상이면 추가하지 않음
                        if iou >= 0.7:
                            should_add = False
                            print(f"IOU {iou:.3f} >= 0.7, skipping duplicate detection")
                            break

                # 중복되지 않는 경우에만 추가
                # chip(0), null(1), back(2)은 오프셋을 더하지 않고, 포커 카드(3-54)는 CLASS_OFFSET를 더함
                if should_add:
                    # chip(0), null(1), back(2)인 경우 오프셋 없이 그대로 사용
                    # YOLO 모델 출력: chip=0, null=1, back=2, 포커 카드=3-54
                    if cls == 0:  # YOLO 모델의 chip 클래스 (0) → 0으로 매핑
                        final_cls = 0
                    elif cls == 1:  # YOLO 모델의 null 클래스 (1) → 1로 매핑
                        final_cls = 1
                    elif cls == 2:  # YOLO 모델의 back 클래스 (2) → 2로 매핑
                        final_cls = 2
                    else:  # 포커 카드 클래스 (3-54) → CLASS_OFFSET를 더함 (6-57)
                        final_cls = cls + self.CLASS_OFFSET

                    label = [
                        final_cls,
                        x_center,
                        y_center,
                        w,
                        h,
                    ]
                    # 기존 라벨이 없으면 초기화
                    if image_path not in self.labels:
                        self.labels[image_path] = []
                    self.labels[image_path].append(label)
                    new_labels.append(label)
                    existing_labels.append(label)  # 다음 검사를 위해 복사본에 추가

            # 라벨 저장 및 이미지 다시 표시
            self.save_label(image_path)
            self.show_image()

            # 결과 메시지 제거 - 조용히 처리
            new_count = len(new_labels)
            total_count = len(self.labels.get(image_path, []))

            # 콘솔에만 간단히 출력
            if new_count > 0:
                print(
                    f"Auto Labeling: Added {new_count} new labels (Total: {total_count})"
                )
            else:
                print(f"Auto Labeling: No new objects detected (Total: {total_count})")

        except Exception as e:
            import traceback

            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Auto labeling failed: {str(e)}")

    def update_position_info(self):
        """현재 위치 정보 업데이트"""
        if self.image_list:
            current_pos = self.current_index + 1
            total_images = len(self.image_list)

            # 현재까지 라벨된 이미지 수 계산
            labeled_count = 0
            for i in range(self.current_index + 1):
                if i < len(self.image_list):
                    image_path = self.image_list[i]
                    if image_path in self.labels and self.labels[image_path]:
                        labeled_count += 1

            self.position_info_label.setText(
                f"{current_pos}/{total_images} (Labeled: {labeled_count})"
            )
        else:
            self.position_info_label.setText("0/0")

    def jump_to_image(self):
        """입력된 이미지 번호로 이동"""
        if not self.image_list:
            return

        # 입력된 텍스트에서 숫자 추출
        text = self.position_info_label.text()
        # "1/100" 형식에서 첫 번째 숫자만 추출하거나, 단순 숫자만 입력된 경우
        try:
            # "/"로 분리하여 첫 번째 숫자 추출
            if "/" in text:
                target_pos = int(text.split("/")[0].strip())
            else:
                # 단순 숫자만 입력된 경우
                target_pos = int(text.strip())

            # 1-based를 0-based로 변환
            target_index = target_pos - 1

            # 유효한 범위인지 확인
            if 0 <= target_index < len(self.image_list):
                self.current_index = target_index
                # 이미지 변경 시 실행 취소 스택 초기화
                self.undo_stack = []
                self.show_image()
                # 위치 정보 업데이트
                self.update_position_info()
            else:
                QMessageBox.warning(
                    self,
                    "Invalid Index",
                    f"이미지 번호는 1부터 {len(self.image_list)}까지 입력할 수 있습니다.",
                )
                # 현재 위치로 복원
                self.update_position_info()
        except ValueError:
            QMessageBox.warning(
                self, "Invalid Input", "올바른 숫자를 입력해주세요. (예: 1 또는 1/100)"
            )
            # 현재 위치로 복원
            self.update_position_info()

    def save_state_for_undo(self):
        """현재 상태를 실행 취소 스택에 저장"""
        if not self.image_list:
            return

        image_path = self.image_list[self.current_index]

        # 현재 이미지의 라벨 상태를 깊은 복사
        current_labels = []
        if image_path in self.labels and self.labels[image_path]:
            for label in self.labels[image_path]:
                if isinstance(label, list):
                    current_labels.append(label.copy())
                else:
                    current_labels.append(label)

        # 상태 저장
        state = {
            "image_path": image_path,
            "labels": current_labels,
        }

        # 스택에 추가
        self.undo_stack.append(state)
        print(f"State saved for undo: {len(current_labels)} labels")  # 디버깅용

        # 최대 개수 제한
        if len(self.undo_stack) > self.max_undo_history:
            self.undo_stack.pop(0)

    def undo_last_action(self):
        """마지막 작업 실행 취소"""
        if not self.undo_stack:
            print("No actions to undo")
            return

        if not self.image_list:
            return

        # 현재 이미지 경로 확인
        current_image_path = self.image_list[self.current_index]

        # 스택에서 같은 이미지의 상태를 찾기 (뒤에서부터)
        found_state = None
        temp_stack = []

        while self.undo_stack:
            state = self.undo_stack.pop()
            if state["image_path"] == current_image_path:
                found_state = state
                break
            else:
                temp_stack.append(state)

        # 다른 이미지의 상태들을 다시 스택에 넣기
        while temp_stack:
            self.undo_stack.append(temp_stack.pop())

        if found_state:
            # 라벨 복원
            restored_labels = []
            for label in found_state["labels"]:
                if isinstance(label, list):
                    restored_labels.append(label.copy())
                else:
                    restored_labels.append(label)

            self.labels[current_image_path] = restored_labels

            # 파일 저장 및 화면 업데이트
            self.save_label(current_image_path)
            self.show_image()
            print(
                f"Undo: Restored {len(restored_labels)} label(s) for {os.path.basename(current_image_path)}"
            )
        else:
            print(f"Cannot undo: No undo state found for current image")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = ImageViewer()
    viewer.show()
    sys.exit(app.exec_())
