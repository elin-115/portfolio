# %% [markdown]
# # 흉부 X-ray 폐렴 분류 — v7 (ConvNeXt-Tiny × 5-fold → v4·v5와 앙상블)
#
# 지금 최고점은 **v4(EfficientNet, ImageNet) + v5(DenseNet, X-ray 전용)의 순위 평균 = 0.9119** 입니다.
# 두 모델이 **서로 다른 관점**(순위 상관 0.87)이라 섞었을 때 효과가 있었습니다.
# 반대로 v6에서 X-ray 모델을 더 넣었을 때는 v5와 너무 비슷해서(상관 0.93~0.95) 효과가 없었습니다.
#
# 그래서 v4·v5와 **구조가 다른** 세 번째 모델을 같은 방식(5-fold)으로 만듭니다.
#
# | | v4 | v5 | **v7** |
# |---|---|---|---|
# | 모델 | EfficientNet-B0 | DenseNet121 | **ConvNeXt-Tiny** (2022년, CNN을 Transformer 방식으로 다시 설계한 구조) |
# | 사전학습 | ImageNet | 흉부 X-ray | ImageNet |
# | 학습 | 5-fold × 6 에포크 | 5-fold × 5 에포크 | 5-fold × **4 에포크** (모델이 커서 시간 절약) |
#
# 전처리·클래스 가중치·증강은 v4와 같습니다. **test는 예측에만 사용**합니다.
# ConvNeXt-Tiny는 EfficientNet-B0보다 계산량이 약 10배라 맥(mps)에서 **약 3~4시간** 걸립니다.
# 결과: `best_model_v7_fold1~5.pt`, `test_prob_v7.npy`, `submission_rank_v4_v5_v7_r58.csv` 등

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
import torch.nn as nn                               # 신경망 층(Linear 등)과 손실함수
import torch.optim as optim                         # 옵티마이저(AdamW)와 학습률 스케줄러
from torch.utils.data import Dataset, DataLoader    # 데이터를 모델에 배치 단위로 넣어주는 도구
from torchvision import transforms, models          # transforms: 이미지 전처리·증강 / models: 사전학습 모델 모음

from sklearn.model_selection import StratifiedGroupKFold      # 묶음(group) 단위 + 정상/폐렴 비율 유지하며 k조각으로 나누기
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
# 사전학습 모델은 컬러(3채널) 사진으로 학습됐기 때문에 `Grayscale(3)`으로 흑백 한 장을 3채널로 복사하고, ImageNet 평균/표준편차로 정규화합니다.

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
    transforms.Grayscale(3),                                                        # 흑백 1채널 → 같은 값 3채널 (사전학습 모델은 RGB 입력)
    transforms.ToTensor(),                                                          # PIL 이미지 → 텐서, 픽셀 0~255 → 0~1
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),             # ImageNet 평균/표준편차로 정규화 (사전학습 때와 같은 기준)
])

# 검증/테스트용: 변형 없이
eval_tf = transforms.Compose([
    transforms.Grayscale(3),                                                # 학습 때와 똑같이 3채널로
    transforms.ToTensor(),                                                  # 텐서 변환
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),     # 같은 정규화 (증강은 하지 않음 → 매번 같은 결과)
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
# ## 4. 5-fold 나누기 (환자 묶음 단위)
# 연속된 10장을 한 묶음(`group`)으로 보고, 묶음이 쪼개지지 않게 5조각으로 나눕니다.
# `StratifiedGroupKFold`는 **같은 묶음은 항상 같은 조각**에 넣으면서, 조각마다 정상/폐렴 비율도 비슷하게 맞춰 줍니다.

# %%
train_df = pd.read_csv("train.csv")                        # 학습 표 (file_name, label)
test_df = pd.read_csv("test.csv")                          # 테스트 표 (file_name)
group = train_df.index // 10                               # 연속 10장씩 같은 묶음 번호 (0~9번 → 0, 10~19번 → 1 ...)

skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)   # 5조각, 섞어서 나누기, 시드 고정
folds = list(skf.split(train_df, train_df["label"], groups=group))     # [(학습 행 번호, 검증 행 번호), ...] 5쌍

for k, (tr_idx, val_idx) in enumerate(folds):
    print(f"fold {k+1}: 학습 {len(tr_idx)}장, 검증 {len(val_idx)}장, 검증 폐렴 비율 {train_df['label'][val_idx].mean():.3f}")

# test는 예측에만 사용 (fold마다 같은 loader 재사용)
test_loader = DataLoader(TestCSV(test_df, "test", eval_tf), batch_size=64, shuffle=False)   # 순서 유지!

# %% [markdown]
# ## 5. fold별 학습 → 검증 → test 예측
# fold마다 아래를 반복합니다. v4와 같고 1번(모델)만 ConvNeXt-Tiny로 바뀌었습니다.
# 1. 새 ConvNeXt-Tiny 불러오기 (fold마다 처음부터)
# 2. 클래스 가중치 계산 (그 fold의 학습 데이터로만)
# 3. 6 에포크 학습, 검증 정확도가 가장 높은 에포크의 가중치 저장
# 4. 가장 좋았던 가중치로 test 폐렴 확률 예측 (+ 좌우반전 TTA)
#
# 끝난 fold의 test 확률은 매번 `test_probs_v7_folds.npy`에 저장됩니다.

# %%
EPOCHS = 4                                                 # fold마다 4번 반복 학습 (모델이 커서 v4보다 적게)
test_probs = []                                            # fold별 test 폐렴 확률 (5개 쌓임)
fold_acc = []                                              # fold별 최고 검증 정확도
oof_prob = np.zeros(len(train_df))                         # 검증용으로 쓰였을 때의 폐렴 확률 (모든 train 사진이 한 번씩 채워짐)

for k, (tr_idx, val_idx) in enumerate(folds):
    print(f"========== fold {k+1} / 5 ==========")
    tr_df, val_df = train_df.iloc[tr_idx], train_df.iloc[val_idx]          # 이번 fold의 학습/검증 표
    train_loader = DataLoader(TrainCSV(tr_df, "train", train_tf), batch_size=32, shuffle=True)    # 학습: 증강 O, 섞기
    val_loader   = DataLoader(TrainCSV(val_df, "train", eval_tf), batch_size=64, shuffle=False)  # 검증: 증강 X

    # 1. 모델 (fold마다 새로 불러와야 이전 fold의 학습 내용이 섞이지 않음)
    model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)   # ImageNet 사전학습 ConvNeXt-Tiny (처음 한 번 약 110MB 다운로드)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, 2)               # ConvNeXt의 마지막 층은 classifier[2] (768 → 2개 클래스로 교체)
    model = model.to(device)

    # 2. 클래스 가중치 + 손실함수 + 옵티마이저 (v1과 동일, label smoothing 없음)
    n0, n1 = (tr_df["label"] == 0).sum(), (tr_df["label"] == 1).sum()                     # 이번 fold 학습용 정상/폐렴 개수
    class_weight = torch.tensor([len(tr_df) / (2 * n0), len(tr_df) / (2 * n1)], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weight)                                   # 정상을 틀리면 더 큰 벌점
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)                # 학습률 0.0001
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)              # 학습률을 점점 줄이기

    # 3. 학습 + 검증
    best_acc = 0
    for ep in range(EPOCHS):
        model.train(); running = 0.0                                   # 학습 모드
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            loss = criterion(model(x), y)                              # 순전파 + 손실 계산
            optimizer.zero_grad(); loss.backward(); optimizer.step()  # 역전파 + 가중치 업데이트
            running += loss.item() * x.size(0)
        scheduler.step()

        model.eval(); val_prob = []                                    # 평가 모드
        with torch.no_grad():
            for x, y in val_loader:
                val_prob.extend(torch.softmax(model(x.to(device)), 1)[:, 1].cpu().numpy())   # 검증 폐렴 확률
        val_prob = np.array(val_prob)
        val_acc = accuracy_score(val_df["label"], val_prob > 0.5)     # 검증 정확도
        print(f"  epoch {ep+1} | loss {running/len(tr_df):.4f} | val acc {val_acc:.4f}")
        if val_acc > best_acc:                                         # 최고 기록이면
            best_acc = val_acc
            oof_prob[val_idx] = val_prob                               # 이 fold 검증 사진들의 확률 기록
            torch.save(model.state_dict(), f"best_model_v7_fold{k+1}.pt")   # 가중치 저장

    fold_acc.append(best_acc)
    print(f"  fold {k+1} 최고 검증 정확도: {best_acc:.4f}")

    # 4. 가장 좋았던 가중치로 test 예측
    model.load_state_dict(torch.load(f"best_model_v7_fold{k+1}.pt", map_location=device))
    model.eval(); prob = []
    with torch.no_grad():
        for x in test_loader:
            x = x.to(device)
            p1 = torch.softmax(model(x), 1)[:, 1]                          # 원본의 폐렴 확률
            p2 = torch.softmax(model(torch.flip(x, dims=[3])), 1)[:, 1]   # 좌우반전의 폐렴 확률
            prob.extend(((p1 + p2) / 2).cpu().numpy())                   # 평균
    test_probs.append(np.array(prob))                                      # 이번 fold의 test 확률 저장
    np.save("test_probs_v7_folds.npy", np.array(test_probs))              # 중간에 멈춰도 여기까지 결과는 남도록 매 fold 저장

# %% [markdown]
# ## 6. 교차검증 결과
# 모든 train 사진이 한 번씩 검증용으로 쓰였으므로, 이 정확도가 **train 전체에 대한 정직한 성능 추정치**입니다.
# (단, test는 train과 분포가 달라서 리더보드는 이보다 낮게 나옵니다.)

# %%
print("fold별 검증 정확도:", np.round(fold_acc, 4))
print("평균:", np.mean(fold_acc).round(4))
print("전체 OOF 정확도:", round(accuracy_score(train_df["label"], oof_prob > 0.5), 4))   # OOF = 검증용으로 예측한 확률 모음
print(confusion_matrix(train_df["label"], oof_prob > 0.5))                              # [[정상→정상, 정상→폐렴], [폐렴→정상, 폐렴→폐렴]]

# %% [markdown]
# ## 7. v4 · v5 · v7 순위 평균 → 제출 파일
# 모델마다 확률이 퍼지는 범위가 달라서 **확률이 아니라 순위를 평균**냅니다.
# 폐렴 판정 비율은 v4(고정 기준 0.99, 리더보드 0.9022)와 같은 약 57.9%로 맞춥니다.

# %%
test_prob = np.mean(test_probs, axis=0)                    # 5개 fold 모델 확률의 평균
np.save("test_prob_v7.npy", test_prob)                     # 저장

probs = {"v4": np.load("test_prob_v4.npy"),                # EfficientNet-B0 (ImageNet) 5-fold
         "v5": np.load("test_prob_v5.npy"),                # DenseNet121 (X-ray) 5-fold
         "v7": test_prob}                                  # ConvNeXt-Tiny (ImageNet) 5-fold
rank = {k: pd.Series(v).rank(pct=True).values for k, v in probs.items()}   # 모델별 순위 (0~1)
print("순위 상관 (1에 가까울수록 같은 판단 → 섞어도 효과 작음)")
print(pd.DataFrame(rank).corr().round(3))

v4_sub = pd.read_csv("submission_v4_t099.csv")             # v4 제출 파일
r = v4_sub["label"].mean()                                 # 맞출 폐렴 비율 (약 0.579)
best = pd.read_csv("submission_rank_v4_v5_r58.csv")["label"]   # 현재 최고점 파일 (0.9119)
submission = pd.read_csv("sample_submission.csv")

for names, fname in [(["v4", "v5", "v7"], "submission_rank_v4_v5_v7_r58.csv"),   # 세 모델 같은 비중 (추천)
                     (["v7"], "submission_v7_r58.csv")]:                          # v7 단독 (참고용)
    mean_rank = np.mean([rank[k] for k in names], axis=0)                         # 고른 모델들의 순위 평균
    threshold = np.quantile(mean_rank, 1 - r)                                      # 폐렴 비율이 r이 되는 기준
    submission["label"] = (mean_rank > threshold).astype(int)
    submission.to_csv(fname, index=False)
    print(f"{fname}: 폐렴 비율 {submission['label'].mean():.3f}, 최고점 파일과 다른 장수 {(submission['label'] != best).sum()}")
