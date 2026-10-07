# %% [markdown]
# # 흉부 X-ray 폐렴 분류 — v10 (검은 띠·가로세로 비율 단서 없애기)
#
# ## 전처리를 다시 점검해서 찾은 문제
# train 사진을 라벨별로 보면 **모양이 다릅니다.**
#
# | | 가로÷세로 비율 (중앙값) | 모습 |
# |---|---|---|
# | 정상 | 1.18 | 정사각형에 가까움 → 위아래 **검은 띠가 얇음** |
# | 폐렴 | 1.44 | 가로로 김 → 위아래 **검은 띠가 두꺼움** |
#
# - 폐를 전혀 보지 않고 **가로세로 비율만으로도 AUC 0.88**이 나옵니다. train에서 비율 1.5 이상인 사진은 99% 이상이 폐렴입니다.
# - 그런데 test는 624장 중 263장(42%)이 비율 1.5 이상입니다. 즉 **test에는 가로로 긴 정상 사진이 많은데, train에는 거의 없습니다.**
# - 모델은 "검은 띠가 두꺼우면 폐렴"을 배웠고, 그래서 test의 가로로 긴 정상 사진을 폐렴으로 판정했습니다. 지금까지의 오답 패턴(정상→폐렴)과 정확히 일치합니다.
# - v8·v9(폐 자르기)가 효과 있었던 것도 검은 띠가 잘려 나갔기 때문으로 보입니다.
#
# ## v10에서 바꾼 전처리
# | 단계 | 내용 |
# |---|---|
# | 1. **검은 띠 잘라내기** | letterbox 후 내용이 있는 부분만 남김 (`crop_content`) |
# | 2. **정사각형으로 늘리기** | 잘라낸 내용을 224×224에 꽉 채움 → 띠가 전혀 없음 |
# | 3. **가로세로 비율 흔들기** (학습 때만) | `RandomResizedCrop(ratio 0.75~1.33)`으로 매번 다른 비율로 늘이고 줄임 → 가슴이 납작한지 길쭉한지로 맞히지 못하게 함 |
#
# 모델·학습 설정은 v4와 같습니다 (EfficientNet-B0, 5-fold, 6 에포크). 끝난 fold는 `v10_fold번호.npz`로 저장해 이어서 실행할 수 있습니다.
#
# ## test 없이 효과를 확인하는 방법
# train 안에도 **가로로 긴 정상 사진**(비율 1.35 초과)이 100장쯤 있습니다. 단서와 정답이 어긋나는 이 사진들의 검증(OOF) 정확도를 따로 봅니다.
# 7번 셀에서 저장된 v4 모델(기존 전처리)과 v10을 같은 사진으로 비교합니다.
#
# test는 예측에만 사용합니다. 예상 시간 약 1시간 40분. 결과: `test_prob_v10.npy`, `submission_rank_v10_v5_v8_v9_r58.csv`

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
# **letterbox**: test 원본을 train과 같은 224×224(검은 띠 포함) 모양으로 만듭니다. train 이미지는 이미 이 모양입니다.
#
# **crop_content (v10 추가)**: 그 다음 train·test 모두 **검은 띠를 잘라내고** 내용만 224×224로 늘립니다. train과 test가 같은 순서(letterbox → crop_content)를 거치므로 같은 방식으로 처리됩니다.
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

def crop_content(img):
    a = np.asarray(img)                                    # 224×224 숫자 배열
    bright = a > 8                                         # 검정(0~8)이 아닌 픽셀
    rows = np.where(bright.mean(axis=1) > 0.05)[0]         # 밝은 픽셀이 5% 넘는 행 = 내용이 있는 행
    cols = np.where(bright.mean(axis=0) > 0.05)[0]         # 내용이 있는 열
    if len(rows) < 20 or len(cols) < 20:                   # 내용을 못 찾으면 그대로
        return img
    img = img.crop((cols[0], rows[0], cols[-1] + 1, rows[-1] + 1))      # 내용 부분만 자르기 (검은 띠 제거)
    return img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)             # 224×224에 꽉 채우기 (가로세로 비율은 무시하고 늘림)

# 학습용: 매번 조금씩 다르게 변형(증강)
train_tf = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.75, 1.0), ratio=(0.75, 1.33)),  # 75~100% 영역을 가로세로 비율 0.75~1.33으로 무작위로 잘라 224로 → 비율 단서 흔들기
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
        x = self.tf(crop_content(letterbox(img)))           # letterbox → 검은 띠 제거·늘리기 → 전처리
        y = self.labels[i]                                  # 정답
        return x, y                                         # (이미지 텐서, 정답) 반환

class TestCSV(Dataset):
    def __init__(self, df, root_dir, tf):
        self.files = df["file_name"].tolist()               # 파일명 목록 (test는 정답 없음)
        self.root, self.tf = root_dir, tf
    def __len__(self): return len(self.files)
    def __getitem__(self, i):
        img = Image.open(os.path.join(self.root, self.files[i])).convert("L")
        x = self.tf(crop_content(letterbox(img)))           # test도 똑같이 letterbox → 검은 띠 제거·늘리기
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
# fold마다 아래를 반복합니다. 모델·학습 설정은 v4와 같고 전처리만 다릅니다.
# 1. 새 EfficientNet-B0 불러오기 (fold마다 처음부터)
# 2. 클래스 가중치 계산 (그 fold의 학습 데이터로만)
# 3. 6 에포크 학습, 검증 정확도가 가장 높은 에포크의 가중치 저장
# 4. 가장 좋았던 가중치로 test 폐렴 확률 예측 (+ 좌우반전 TTA)

# %%
EPOCHS = 6                                                 # fold마다 6번 반복 학습
test_probs = []                                            # fold별 test 폐렴 확률 (5개 쌓임)
fold_acc = []                                              # fold별 최고 검증 정확도
oof_prob = np.zeros(len(train_df))                         # 검증용으로 쓰였을 때의 폐렴 확률 (모든 train 사진이 한 번씩 채워짐)

for k, (tr_idx, val_idx) in enumerate(folds):
    print(f"========== fold {k+1} / 5 ==========")
    done_file = f"v10_fold{k+1}.npz"                                        # 이 fold의 결과 저장 파일
    if os.path.exists(done_file):                                           # 이미 끝난 fold면 불러오고 건너뛰기
        saved = np.load(done_file)
        oof_prob[val_idx] = saved["val_prob"]; fold_acc.append(float(saved["best_acc"])); test_probs.append(saved["test_prob"])
        print(f"  이미 완료 (검증 정확도 {float(saved['best_acc']):.4f}) → 건너뜀")
        continue
    tr_df, val_df = train_df.iloc[tr_idx], train_df.iloc[val_idx]          # 이번 fold의 학습/검증 표
    train_loader = DataLoader(TrainCSV(tr_df, "train", train_tf), batch_size=32, shuffle=True)    # 학습: 증강 O, 섞기
    val_loader   = DataLoader(TrainCSV(val_df, "train", eval_tf), batch_size=64, shuffle=False)  # 검증: 증강 X

    # 1. 모델 (fold마다 새로 불러와야 이전 fold의 학습 내용이 섞이지 않음)
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)   # ImageNet 사전학습 가중치
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)                   # 마지막 층 → 2개 클래스
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
            torch.save(model.state_dict(), f"best_model_v10_fold{k+1}.pt")   # 가중치 저장

    fold_acc.append(best_acc)
    print(f"  fold {k+1} 최고 검증 정확도: {best_acc:.4f}")

    # 4. 가장 좋았던 가중치로 test 예측
    model.load_state_dict(torch.load(f"best_model_v10_fold{k+1}.pt", map_location=device))
    model.eval(); prob = []
    with torch.no_grad():
        for x in test_loader:
            x = x.to(device)
            p1 = torch.softmax(model(x), 1)[:, 1]                          # 원본의 폐렴 확률
            p2 = torch.softmax(model(torch.flip(x, dims=[3])), 1)[:, 1]   # 좌우반전의 폐렴 확률
            prob.extend(((p1 + p2) / 2).cpu().numpy())                   # 평균
    test_probs.append(np.array(prob))                                      # 이번 fold의 test 확률 저장
    np.savez(done_file, val_prob=oof_prob[val_idx], best_acc=best_acc, test_prob=np.array(prob))   # 이 fold 결과 저장 → 다시 실행하면 건너뜀
    del model                                                              # 다음 fold 전에 메모리 비우기
    if device.type == "mps": torch.mps.empty_cache()

# %% [markdown]
# ## 6. 교차검증 결과 (전체)
# 모든 train 사진이 한 번씩 검증용으로 쓰였으므로, 이 정확도가 **train 전체에 대한 정직한 성능 추정치**입니다.
# (단, test는 train과 분포가 달라서 리더보드는 이보다 낮게 나옵니다.)

# %%
print("fold별 검증 정확도:", np.round(fold_acc, 4))
print("평균:", np.mean(fold_acc).round(4))
print("전체 OOF 정확도:", round(accuracy_score(train_df["label"], oof_prob > 0.5), 4))   # OOF = 검증용으로 예측한 확률 모음
print(confusion_matrix(train_df["label"], oof_prob > 0.5))                              # [[정상→정상, 정상→폐렴], [폐렴→정상, 폐렴→폐렴]]

# %% [markdown]
# ## 7. 단서와 정답이 어긋나는 사진에서의 정확도 (test 없이 확인)
# train 사진마다 원래 가로세로 비율(검은 띠를 뺀 내용의 가로÷세로)을 구하고, 아래 두 그룹의 검증(OOF) 정확도를 봅니다.
# - **가로로 긴 정상** (비율 > 1.35): 띠 단서대로라면 '폐렴'으로 틀리기 쉬운 사진
# - **정사각에 가까운 폐렴** (비율 < 1.2): 띠 단서대로라면 '정상'으로 틀리기 쉬운 사진
#
# 비교를 위해 **기존 전처리(v4)** 모델도 같은 fold의 검증 사진으로 예측합니다. v4 모델 파일(`best_model_fold1~5.pt`)이 없으면 v10만 출력합니다.

# %%
def content_ratio(fname):
    a = np.asarray(Image.open(os.path.join("train", fname)).convert("L"))
    bright = a > 8
    rows = np.where(bright.mean(axis=1) > 0.05)[0]; cols = np.where(bright.mean(axis=0) > 0.05)[0]
    return (cols[-1] - cols[0] + 1) / (rows[-1] - rows[0] + 1)           # 내용의 가로 ÷ 세로

train_df["ratio"] = [content_ratio(f) for f in train_df["file_name"]]    # 사진별 가로세로 비율
wide_normal = (train_df["label"] == 0) & (train_df["ratio"] > 1.35)       # 가로로 긴 정상
square_pneu = (train_df["label"] == 1) & (train_df["ratio"] < 1.2)        # 정사각에 가까운 폐렴
print("가로로 긴 정상:", wide_normal.sum(), "장 / 정사각에 가까운 폐렴:", square_pneu.sum(), "장")

def report(name, prob):
    pred = (prob > 0.5).astype(int)                                       # 0.5 기준 예측
    print(f"{name}: 전체 {accuracy_score(train_df['label'], pred):.4f} | "
          f"가로로 긴 정상 {(pred[wide_normal] == 0).mean():.4f} | 정사각 폐렴 {(pred[square_pneu] == 1).mean():.4f}")

report("v10 (새 전처리)", oof_prob)

# 기존 전처리(v4) 모델로 같은 검증 사진 예측
if all(os.path.exists(f"best_model_fold{k+1}.pt") for k in range(5)):
    old_tf = transforms.Compose([transforms.Grayscale(3), transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])   # v4의 검증용 전처리 (검은 띠 그대로)
    old_oof = np.zeros(len(train_df))
    for k, (tr_idx, val_idx) in enumerate(folds):
        m = models.efficientnet_b0(); m.classifier[1] = nn.Linear(m.classifier[1].in_features, 2)        # v4와 같은 구조
        m.load_state_dict(torch.load(f"best_model_fold{k+1}.pt", map_location="cpu")); m = m.to(device).eval()
        files = train_df["file_name"].iloc[val_idx].tolist(); out = []
        with torch.no_grad():
            for i in range(0, len(files), 64):
                x = torch.stack([old_tf(Image.open(os.path.join("train", f)).convert("L")) for f in files[i:i + 64]]).to(device)
                out.extend(torch.softmax(m(x), 1)[:, 1].cpu().numpy())
        old_oof[val_idx] = out
    report("v4  (기존 전처리)", old_oof)

# %% [markdown]
# ## 8. 앙상블 → 제출 파일
# 지금 최고점(0.9215)은 v4 + v5 + v8 + v9의 순위 평균입니다. 여기서 **v4만 v10으로 바꾼** 파일을 만듭니다.
# 바뀐 것이 하나뿐이라 점수 차이는 새 전처리의 효과를 보여줍니다. 폐렴 판정 비율은 지금까지와 같은 약 57.9%입니다.

# %%
test_prob = np.mean(test_probs, axis=0)                    # 5개 fold 모델 확률의 평균
np.save("test_prob_v10.npy", test_prob)

probs = {"v10": test_prob,                                 # EfficientNet, 검은 띠 제거 + 비율 흔들기
         "v5": np.load("test_prob_v5.npy"),                # X-ray DenseNet, 전체 사진
         "v8": np.load("test_prob_v8.npy"),                # EfficientNet, 폐만
         "v9": np.load("test_prob_v9.npy")}                # X-ray DenseNet, 폐만
rank = {k: pd.Series(v).rank(pct=True).values for k, v in probs.items()}   # 모델별 순위 (0~1)
print("순위 상관")
print(pd.DataFrame(rank).corr().round(3))

r = pd.read_csv("submission_v4_t099.csv")["label"].mean()                  # 맞출 폐렴 비율 (약 0.579)
best = pd.read_csv("submission_rank_v4_v5_v8_v9_r58.csv")["label"]         # 현재 최고점 파일 (0.9215)
submission = pd.read_csv("sample_submission.csv")

for names, fname in [(["v10", "v5", "v8", "v9"], "submission_rank_v10_v5_v8_v9_r58.csv"),   # v4 → v10 교체 (제출용)
                     (["v10"], "submission_v10_r58.csv")]:                                   # v10 단독 (참고용)
    mean_rank = np.mean([rank[k] for k in names], axis=0)
    threshold = np.quantile(mean_rank, 1 - r)
    submission["label"] = (mean_rank > threshold).astype(int)
    submission.to_csv(fname, index=False)
    print(f"{fname}: 폐렴 비율 {submission['label'].mean():.3f}, 최고점 파일과 다른 장수 {(submission['label'] != best).sum()}")
