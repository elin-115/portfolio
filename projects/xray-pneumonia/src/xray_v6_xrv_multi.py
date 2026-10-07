# %% [markdown]
# # 흉부 X-ray 폐렴 분류 — v6 (사전학습 데이터가 다른 X-ray 모델 3개 추가 → 앙상블)
#
# v4(EfficientNet, ImageNet) + v5(DenseNet, X-ray 8개 데이터셋)의 순위 평균이 **0.9119**로 최고점을 기록했습니다.
# X-ray 전용 모델을 섞은 효과가 있었으므로, **서로 다른 병원 데이터로 사전학습된 X-ray 모델**을 더 추가합니다.
#
# | 이름 | 사전학습 데이터 |
# |---|---|
# | chex | CheXpert (미국 스탠퍼드 병원, 약 22만 장) |
# | pc | PadChest (스페인 병원, 약 16만 장) |
# | nih | NIH ChestX-ray14 (미국 국립보건원, 약 11만 장) |
#
# 사전학습 데이터가 다르면 모델마다 **잘 보는 것과 실수하는 사진이 달라서** 평균냈을 때 서로 보완됩니다.
#
# **v5와 다른 점:** 5-fold 대신 **train 전체로 모델당 1번만** 학습합니다 (5 에포크, 고정).
# 5-fold는 검증 점수를 얻는 게 목적이었고, 이번에는 앙상블 재료를 만드는 게 목적이라 시간을 1/5로 줄였습니다.
# 검증을 하지 않으므로 '최고 에포크 선택'도 하지 않고, 5 에포크 학습이 끝난 모델을 그대로 씁니다 (v5의 fold들도 4~5 에포크에서 최고였음).
#
# **test는 예측에만 사용**합니다. 모델당 약 30~40분, 총 약 2시간. 결과: `test_prob_xrv_chex.npy` 등, `submission_rank_v6_r58.csv`

# %% [markdown]
# ## 라이브러리 설치 (처음 한 번만)
# 흉부 X-ray 전용 모델이 들어 있는 `torchxrayvision`을 설치합니다. 설치 후에는 이 셀을 다시 실행할 필요가 없습니다.
# 설치 후 "Restart" 안내가 나오면 VS Code 위쪽의 **재시작(Restart)** 버튼을 누르고 0번 셀부터 다시 실행하세요.

# %%
%pip install torchxrayvision

# %% [markdown]
# ## 0. 데이터 준비 (VS Code / 로컬)
# 데이터 폴더(`train/`, `test/`, `train.csv`, `test.csv`, `sample_submission.csv`)가 있는 경로를 `DATA_DIR`에 적어 주세요.
# 맥(M1/M2)에서는 GPU 역할을 하는 **mps**가 자동으로 사용됩니다.
#
# VS Code에서 처음 실행할 때 오른쪽 위 **커널 선택**에서 torch가 설치된 파이썬을 고르세요. 라이브러리가 없다면 아래 명령을 터미널에서 한 번 실행합니다.
# `pip install torch torchvision scikit-learn pandas matplotlib pillow`

# %%
import os

DATA_DIR = "."   # 데이터가 들어 있는 폴더 경로 (본인 경로에 맞게 수정)
os.chdir(DATA_DIR)                                 # 작업 폴더를 데이터 폴더로 이동 → 이후 "train.csv"처럼 짧은 경로 사용 가능

print(os.getcwd())                                 # 현재 작업 폴더 확인
print(os.listdir())                                # train, test, train.csv, test.csv, sample_submission.csv 가 보여야 정상
print("train 이미지 수:", len(os.listdir("train")))   # 5216
print("test 이미지 수:", len(os.listdir("test")))     # 624

# %% [markdown]
# ## 1. 라이브러리 불러오기
# baseline과 같고, 사전학습 모델을 쓰기 위해 `torchvision.models`, 환자 묶음 단위로 5조각을 나누기 위해 `StratifiedGroupKFold`, 결과 확인용 `accuracy_score`, `confusion_matrix`를 추가했습니다.

# %%
import os, random                      # os: 파일 경로 다루기 / random: 파이썬 난수
import numpy as np, pandas as pd       # numpy: 숫자 배열 계산 / pandas: CSV 표 다루기
from PIL import Image                  # 이미지 파일 열기·자르기·붙이기

import torch                                        # PyTorch 핵심
import torchxrayvision as xrv                       # 흉부 X-ray 전용 사전학습 모델 모음
import torch.nn as nn                               # 신경망 층(Linear 등)과 손실함수
import torch.optim as optim                         # 옵티마이저(AdamW)와 학습률 스케줄러
from torch.utils.data import Dataset, DataLoader    # 데이터를 모델에 배치 단위로 넣어주는 도구
from torchvision import transforms, models          # transforms: 이미지 전처리·증강 / models: 사전학습 모델 모음

from sklearn.metrics import accuracy_score, confusion_matrix  # 정확도(대회 평가 산식), 혼동행렬

# 실행할 때마다 결과가 같도록 시드 고정
random.seed(42); np.random.seed(42); torch.manual_seed(42)   # 파이썬·numpy·torch 난수를 모두 42로 고정

# GPU가 있으면 GPU(cuda), 맥이면 mps, 없으면 cpu
if torch.cuda.is_available():                 # 코랩 GPU(T4 등)가 있으면
    device = torch.device("cuda")
elif torch.backends.mps.is_available():       # 맥(M1/M2) GPU가 있으면
    device = torch.device("mps")
else:                                         # 둘 다 없으면 CPU (매우 느림)
    device = torch.device("cpu")
print(device)                                 # 코랩에서는 'cuda'가 나와야 정상

# %% [markdown]
# ## 2. 이미지 전처리
#
# **letterbox**: 이미지를 가로세로 비율을 유지한 채 224 안에 들어가게 줄이고, 남는 부분을 검은색으로 채웁니다. train과 똑같은 모양을 만들기 위해 test(와 검증)에 사용합니다. train 이미지는 이미 224×224라서 이 함수를 거쳐도 그대로입니다.
#
# v5 모델은 **흑백 1채널** X-ray로, 픽셀 값을 **-1024 ~ 1024** 범위로 바꿔서 학습됐습니다. 그래서 `ToTensor()`로 0~1이 된 값을 `× 2048 − 1024` 해서 같은 범위로 맞춥니다.
# (ImageNet 모델처럼 3채널로 복사하거나 ImageNet 평균으로 정규화하지 않습니다.)

# %%
IMG_SIZE = 224    # 모델에 넣을 이미지 크기 (train 이미지 크기와 같게)

def letterbox(img):
    img = img.copy()                                  # 원본을 건드리지 않도록 복사
    img.thumbnail((IMG_SIZE, IMG_SIZE))              # 비율 유지하며 긴 변을 224로 줄이기 (예: 1600×1000 → 224×140)
    canvas = Image.new("L", (IMG_SIZE, IMG_SIZE), 0)  # 224×224 검은 도화지 ("L"=흑백, 0=검정)
    x = (IMG_SIZE - img.size[0]) // 2                 # 가로 여백의 절반 → 왼쪽 시작 위치
    y = (IMG_SIZE - img.size[1]) // 2                 # 세로 여백의 절반 → 위쪽 시작 위치 (예: (224-140)//2 = 42)
    canvas.paste(img, (x, y))                          # 가운데에 붙이기 → 위아래(또는 좌우)가 검은 띠
    return canvas

# 학습용: 매번 조금씩 다르게 변형(증강)
train_tf = transforms.Compose([
    transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),  # ±10도 회전, 상하좌우 5% 이동, 0.9~1.1배 확대/축소
    transforms.ColorJitter(brightness=0.2, contrast=0.2),                           # 밝기·대비를 ±20% 범위에서 무작위로 변경
    transforms.ToTensor(),                                                          # PIL 흑백 이미지 → 1채널 텐서, 픽셀 0~255 → 0~1
    transforms.Lambda(lambda x: x * 2048 - 1024),                                   # 0~1 → -1024~1024 (X-ray 전용 모델이 학습된 범위)
])

# 검증/테스트용: 변형 없이
eval_tf = transforms.Compose([
    transforms.ToTensor(),                                                  # 1채널 텐서, 0~1
    transforms.Lambda(lambda x: x * 2048 - 1024),                           # 학습 때와 같은 범위 변환 (증강은 하지 않음)
])

# %% [markdown]
# ## 3. Dataset 만들기
# baseline의 `TrainCSV`, `TestCSV`와 같은 구조입니다. 달라진 점은
# - CSV 경로 대신 **DataFrame**을 받아서, 학습용/검증용으로 나눈 표를 각각 넣을 수 있게 했고
# - 전처리(`tf`)를 바깥에서 골라 넣을 수 있게 했고
# - 이미지를 열자마자 `letterbox`를 적용합니다.

# %%
class TrainCSV(Dataset):
    def __init__(self, df, root_dir, tf):
        self.files = df["file_name"].tolist()               # 파일명 목록
        self.labels = df["label"].astype(int).tolist()      # 정답 목록 (0/1)
        self.root, self.tf = root_dir, tf                   # 이미지 폴더, 적용할 전처리
    def __len__(self): return len(self.files)               # 데이터 개수
    def __getitem__(self, i):                               # i번째 데이터를 요청하면
        img = Image.open(os.path.join(self.root, self.files[i])).convert("L")   # 이미지를 흑백으로 열고
        x = self.tf(letterbox(img))                         # letterbox로 224×224 만든 뒤 전처리
        y = self.labels[i]                                  # 정답
        return x, y                                         # (이미지 텐서, 정답) 반환

class TestCSV(Dataset):
    def __init__(self, df, root_dir, tf):
        self.files = df["file_name"].tolist()               # 파일명 목록 (test는 정답 없음)
        self.root, self.tf = root_dir, tf
    def __len__(self): return len(self.files)
    def __getitem__(self, i):
        img = Image.open(os.path.join(self.root, self.files[i])).convert("L")
        x = self.tf(letterbox(img))                         # 원본 크기 test도 train과 같은 모양으로
        return x                                            # 이미지만 반환


# %% [markdown]
# ## 4. 모델별 학습 → test 예측
# `WEIGHTS`에 적은 사전학습 모델마다 아래를 반복합니다.
# 1. X-ray 전용 DenseNet121 불러오기 → 마지막 층을 2개 클래스로 교체
# 2. train 전체(5,216장)로 5 에포크 학습 (v1~v5와 같은 클래스 가중치·옵티마이저)
# 3. test 폐렴 확률 예측 (+ 좌우반전 TTA) → `test_prob_xrv_이름.npy`로 저장
#
# 이미 확률 파일이 있는 모델은 건너뛰므로, 중간에 멈췄다가 다시 실행해도 처음부터 하지 않습니다.

# %%
train_df = pd.read_csv("train.csv")                        # 학습 표
test_df = pd.read_csv("test.csv")                          # 테스트 표
train_loader = DataLoader(TrainCSV(train_df, "train", train_tf), batch_size=32, shuffle=True)   # train 전체, 증강 O
test_loader  = DataLoader(TestCSV(test_df, "test", eval_tf), batch_size=64, shuffle=False)     # 순서 유지!

n0, n1 = (train_df["label"] == 0).sum(), (train_df["label"] == 1).sum()                       # 정상 1341, 폐렴 3875
class_weight = torch.tensor([len(train_df) / (2 * n0), len(train_df) / (2 * n1)], dtype=torch.float32).to(device)   # [1.94, 0.67]

WEIGHTS = ["chex", "pc", "nih"]                            # 추가할 사전학습 모델 (CheXpert, PadChest, NIH)
EPOCHS = 5                                                 # 모델당 5번 반복 학습

for name in WEIGHTS:
    out_file = f"test_prob_xrv_{name}.npy"                 # 이 모델의 test 확률을 저장할 파일
    if os.path.exists(out_file):                           # 이미 끝난 모델은 건너뛰기
        print(name, "이미 완료 → 건너뜀")
        continue
    print(f"========== {name} ==========")

    # 1. 모델
    model = xrv.models.DenseNet(weights=f"densenet121-res224-{name}")   # 해당 데이터로 사전학습된 DenseNet121 (처음 한 번 다운로드)
    model.classifier = nn.Linear(model.classifier.in_features, 2)        # 마지막 층 → 2개 클래스(정상/폐렴)
    model.op_threshs = None                                              # 원래 소견용 확률 보정값 끄기
    model.apply_sigmoid = False                                          # sigmoid 끄기 (CrossEntropyLoss가 처리)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weight)                 # 클래스 가중치 적용 손실함수
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # 2. 학습
    for ep in range(EPOCHS):
        model.train(); running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = criterion(model(x), y)                                # 순전파 + 손실
            optimizer.zero_grad(); loss.backward(); optimizer.step()    # 역전파 + 업데이트
            running += loss.item() * x.size(0)
        scheduler.step()
        print(f"  epoch {ep+1} | loss {running/len(train_df):.4f}")

    # 3. test 예측
    model.eval(); prob = []
    with torch.no_grad():
        for x in test_loader:
            x = x.to(device)
            p1 = torch.softmax(model(x), 1)[:, 1]                          # 원본의 폐렴 확률
            p2 = torch.softmax(model(torch.flip(x, dims=[3])), 1)[:, 1]   # 좌우반전의 폐렴 확률
            prob.extend(((p1 + p2) / 2).cpu().numpy())
    np.save(out_file, np.array(prob))                                      # 확률 저장
    torch.save(model.state_dict(), f"model_xrv_{name}.pt")                 # 가중치 저장
    del model; torch.mps.empty_cache() if device.type == "mps" else None   # 다음 모델 전에 메모리 비우기

# %% [markdown]
# ## 5. 앙상블 (순위 평균) → 제출 파일
# 모델마다 확률이 퍼지는 범위가 달라서(v5는 0.99를 넘는 사진이 0장) **확률이 아니라 순위를 평균**냅니다.
# 폐렴 판정 비율은 v4(고정 기준 0.99, 리더보드 0.9022)와 같은 약 57.9%로 맞춥니다.
# 비교를 위해 2가지 파일을 만듭니다.

# %%
probs = {"v4": np.load("test_prob_v4.npy"),                # EfficientNet-B0 (ImageNet) 5-fold
         "v5": np.load("test_prob_v5.npy")}                # DenseNet121 (X-ray 8개 데이터셋) 5-fold
for name in WEIGHTS:
    if os.path.exists(f"test_prob_xrv_{name}.npy"):
        probs[name] = np.load(f"test_prob_xrv_{name}.npy")
print("사용 가능한 모델:", list(probs.keys()))

# 모델끼리 순서가 얼마나 비슷한지 (1에 가까울수록 같은 순서 → 섞어도 효과 작음)
rank = {k: pd.Series(v).rank(pct=True).values for k, v in probs.items()}   # 모델별 순위 (0~1)
print(pd.DataFrame(rank).corr().round(3))                                  # 순위 상관 표

v4_sub = pd.read_csv("submission_v4_t099.csv")             # v4 제출 파일
r = v4_sub["label"].mean()                                 # 맞출 폐렴 비율 (약 0.579)
submission = pd.read_csv("sample_submission.csv")

def save_rank_ensemble(names, fname):
    mean_rank = np.mean([rank[k] for k in names], axis=0)              # 고른 모델들의 순위 평균
    threshold = np.quantile(mean_rank, 1 - r)                           # 폐렴 비율이 r이 되는 기준
    submission["label"] = (mean_rank > threshold).astype(int)
    submission.to_csv(fname, index=False)
    best = pd.read_csv("submission_rank_v4_v5_r58.csv")["label"]        # 현재 최고점 파일 (0.9119)
    print(f"{fname}: 모델 {names}, 폐렴 비율 {submission['label'].mean():.3f}, 최고점 파일과 다른 장수 {(submission['label'] != best).sum()}")

# (1) 전부: v4 + v5 + 새 X-ray 모델들
save_rank_ensemble(list(probs.keys()), "submission_rank_v6_all_r58.csv")
# (2) X-ray 모델 비중을 줄인 버전: v4 + (X-ray 모델들의 평균) → v4와 X-ray 쪽이 1:1
xray_names = [k for k in probs if k != "v4"]
rank["xray_mean"] = np.mean([rank[k] for k in xray_names], axis=0)
save_rank_ensemble(["v4", "xray_mean"], "submission_rank_v6_half_r58.csv")
