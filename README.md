# Labeling Tools

YOLO 형식 라벨링을 위한 PyQt5 기반 GUI 툴 모음입니다.

## 📋 목차

- [개요](#개요)
- [툴 목록](#툴-목록)
- [설치](#설치)
- [사용법](#사용법)
- [기능](#기능)

## 개요

이 저장소는 다양한 목적의 이미지 라벨링 작업을 위한 GUI 툴들을 제공합니다. 모든 툴은 PyQt5로 개발되었으며, YOLO 형식의 라벨 파일을 생성/편집할 수 있습니다.

## 툴 목록

### 1. Object Detection Labeler (`object_detection.py`)
**용도**: 일반 객체 검출 라벨링 + YOLO 모델 학습

**주요 기능**:
- 바운딩 박스 라벨링 (드래그 앤 드롭)
- 라벨 편집/삭제 기능
- YOLO 모델 학습 및 자동 라벨링
- 실행 취소/다시 실행 (Ctrl+Z)
- 이전 이미지 라벨 복사 기능
- **클래스 설정 기능** (GUI에서 클래스 추가/수정/삭제 가능)

**클래스 설정**:
- 기본값: 7개 클래스 (class_0 ~ class_6)
- `config.yaml` 파일로 클래스 이름 및 개수 설정 가능
- GUI에서 "⚙️ Class Settings" 버튼으로 클래스 편집 가능
- `config.yaml.example` 파일을 참고하여 설정 파일 생성
- **즉시 적용 기능**: 클래스 변경 후 "Reload" 옵션으로 프로그램 재시작 없이 적용 가능

### 2. Image Classifier (`image_classifier.py`)
**용도**: 크롭된 이미지를 클래스별로 분류

**주요 기능**:
- 이미지를 클래스별 폴더로 이동 (0~10 클래스)
- 키보드 단축키로 빠른 분류 (0~9 숫자 키, . 키)
- 실행 취소 기능 (Ctrl+Z)
- 이미지 삭제 기능 (Del 키)

### 3. Card Labeler (`card_labeler.py`)
**용도**: 카드 이미지 전용 라벨링 (Suit, Rank 등)

**주요 기능**:
- 카드의 Suit와 Rank 라벨링
- ONNX 모델을 사용한 자동 라벨링
- 바운딩 박스 편집 기능

## 설치

### 필수 요구사항

- Python 3.7 이상
- PyQt5
- OpenCV (cv2)
- NumPy

### 설치 방법

```bash
# 저장소 클론
git clone <repository-url>
cd labeling-tools

# 의존성 설치
pip install -r requirements.txt
```

### requirements.txt

```txt
PyQt5>=5.15.0
opencv-python>=4.5.0
numpy>=1.19.0
Pillow>=8.0.0
ultralytics>=8.0.0  # Object Detection 툴용
torch>=1.9.0        # Object Detection 툴용
PyYAML>=5.4.0       # Object Detection 툴용
tqdm>=4.60.0        # 진행률 표시용
onnxruntime>=1.8.0  # Card Labeler용 (선택사항)
```

## 사용법

### Object Detection Labeler

```bash
cd labeling-tools
python tools/object_detection.py
```

**단축키**:
- `A`: 라벨 추가 모드
- `E`: 삭제 모드
- `R`: 편집 모드
- `F`: 이전 이미지 라벨 복사
- `G`: 자동 라벨링 (모델 필요)
- `T`: 모델 학습
- `L`: 모델 로드
- `0~9`: 클래스 선택 (설정된 클래스 수에 따라 동적)
- `Ctrl+Z`: 실행 취소
- `←/→`: 이전/다음 이미지

**클래스 설정 방법**:
1. **설정 파일 사용** (권장):
   ```bash
   # config.yaml.example을 복사
   cp config.yaml.example config.yaml
   
   # config.yaml 파일을 편집하여 클래스 이름 수정
   # 예시:
   classes:
     - "Camera"
     - "Phone"
     - "Lens"
   ```
   - 설정 파일 수정 후 프로그램 재시작 필요

2. **GUI에서 설정** (즉시 적용 가능):
   - 프로그램 실행 후 오른쪽 패널의 "⚙️ Class Settings" 버튼 클릭
   - 클래스 추가/삭제/이름 수정 가능
   - 변경사항은 `config.yaml`에 자동 저장
   - 저장 후 "Reload" 옵션으로 즉시 적용 (프로그램 재시작 불필요)
   - 이미지/라벨/모델 상태는 유지됨

### Image Classifier

```bash
cd labeling-tools
python tools/image_classifier.py
```

**단축키**:
- `0~9`: 클래스 0~9로 이동
- `.`: 클래스 10으로 이동
- `←/→`: 이전/다음 이미지
- `Del`: 현재 이미지 삭제
- `Ctrl+Z`: 이동 취소

### Card Labeler

```bash
cd labeling-tools
python tools/card_labeler.py
```

## 기능

### 공통 기능
- 이미지 확대/축소 (마우스 휠 + Ctrl)
- 이미지 패닝 (마우스 휠 클릭 + 드래그)
- 자연스러운 파일명 정렬
- 위치 점프 기능 (이미지 번호 입력)

### Object Detection 전용
- YOLO 모델 학습
- 자동 라벨링 (AI 지원)

## 라이선스

[LICENSE 파일 참조](LICENSE)

## 기여

이슈 리포트 및 Pull Request를 환영합니다!
