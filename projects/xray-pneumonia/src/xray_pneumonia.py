# %% [markdown]
# # 흉부 X-ray 폐렴 분류
#
# 흉부 X-ray 사진으로 **정상(0) / 폐렴(1)** 을 분류함. 평가 산식은 Accuracy.
#
# ## 핵심 아이디어
# - train 사진은 원본을 비율을 유지한 채 224×224에 넣고 남는 곳을 검게 채운 형태임
# - 폐렴 사진이 정상 사진보다 가로로 길어서 **위아래 검은 띠가 더 두꺼움**
# - 띠를 그대로 두면 모델이 폐가 아니라 **띠의 두께**로 폐렴을 맞히게 됨
# - 그래서 **검은 띠를 잘라내고**, 학습할 때 **가로세로 비율을 무작위로 흔들어** 사진 모양으로 맞히지 못하게 함
#
# ## 구성
# | 단계 | 내용 |
# |---|---|
# | 전처리 | letterbox → 검은 띠 제거 → 224×224 |
# | 증강 | 비율 흔들기, 회전·이동·확대, 밝기·대비 |
# | 검증 | 환자 묶음 단위 5-fold |
# | 모델 | ImageNet 사전학습 **EfficientNet-B0**, **ResNet50**, **DenseNet121** |
# | 예측 | 모델별 5-fold 평균 → 세 모델의 폐렴 확률 평균 → 기준 0.5 |
# | 분석 | Baseline과 비교, 틀린 사진 분석, Grad-CAM, 위험도, 서비스용 결과 데이터 |
#
# - test 데이터는 예측에만 사용함 (학습·통계 계산에 쓰지 않음)
# - 제출 점수: **0.9583**

# %% [markdown]
# ## 0. 데이터 준비
# - `DATA_DIR`에 `train/`, `test/`, `train.csv`, `test.csv`, `sample_submission.csv`가 있는 폴더 경로를 적음
# - 맥(M1/M2)에서는 GPU 역할을 하는 **mps**가 자동으로 사용됨
# - 필요한 라이브러리: `pip install torch torchvision scikit-learn pandas matplotlib pillow`

# %%
import os

DATA_DIR = "."   # 데이터가 들어 있는 폴더 경로 (본인 경로에 맞게 수정)
os.chdir(DATA_DIR)                                     # 작업 폴더를 데이터 폴더로 이동 → 이후 "train.csv"처럼 짧은 경로 사용

print("train 이미지 수:", len(os.listdir("train")))    # 5216
print("test 이미지 수:", len(os.listdir("test")))      # 624

# %% [markdown]
# ## 1. 라이브러리 불러오기

# %%
import random                          # 파이썬 난수
import numpy as np, pandas as pd       # numpy: 숫자 배열 계산 / pandas: CSV 표 다루기
from PIL import Image                  # 이미지 파일 열기·자르기·붙이기
import matplotlib.pyplot as plt        # 그림 그리기

import torch                                        # PyTorch 핵심
import torch.nn as nn                               # 신경망 층(Linear 등)과 손실함수
import torch.optim as optim                         # 옵티마이저(AdamW)와 학습률 스케줄러
from torch.utils.data import Dataset, DataLoader    # 데이터를 모델에 배치 단위로 넣어주는 도구
from torchvision import transforms, models          # transforms: 이미지 전처리·증강 / models: 사전학습 모델 모음

from sklearn.model_selection import StratifiedGroupKFold      # 묶음(group) 단위 + 정상/폐렴 비율 유지하며 k조각으로 나누기
from sklearn.metrics import accuracy_score, confusion_matrix  # 정확도(대회 평가 산식), 혼동행렬

SEED = 42                                     # 난수 시드 (실행할 때마다 결과가 비슷하게 나오도록 고정)

if torch.cuda.is_available():                 # 코랩 GPU(T4 등)가 있으면
    device = torch.device("cuda")
elif torch.backends.mps.is_available():       # 맥(M1/M2) GPU가 있으면
    device = torch.device("mps")
else:                                         # 둘 다 없으면 CPU (매우 느림)
    device = torch.device("cpu")
print(device)

# %% [markdown]
# ## 2. 데이터 살펴보기
# - 라벨은 0 = 정상, 1 = 폐렴
# - train 사진은 전부 224×224, test 사진은 원본 크기 그대로임

# %%
train_df = pd.read_csv("train.csv")        # 학습 표: file_name(파일명), label(0=정상, 1=폐렴)
test_df = pd.read_csv("test.csv")          # 테스트 표: file_name만 있음 (정답 없음)
print(train_df["label"].value_counts())    # 폐렴 3875, 정상 1341 → 폐렴이 약 3배 많음

for f in train_df["file_name"][:3]:
    print("train", f, Image.open("train/" + f).size)   # (가로, 세로) → 전부 (224, 224)
for f in test_df["file_name"][:3]:
    print("test ", f, Image.open("test/" + f).size)    # 원본 크기 (제각각)

# %% [markdown]
# ### 사진 모양이 라벨에 따라 다름
# train 사진에서 **검은 띠를 뺀 내용 부분**의 가로 ÷ 세로 비율을 구해 라벨별로 비교함

# %%
def content_box(a):
    bright = a > 8                                         # 검정(0~8)이 아닌 픽셀
    rows = np.where(bright.mean(axis=1) > 0.05)[0]         # 밝은 픽셀이 5% 넘는 행 = 내용이 있는 행
    cols = np.where(bright.mean(axis=0) > 0.05)[0]         # 내용이 있는 열
    return rows[0], rows[-1] + 1, cols[0], cols[-1] + 1    # 내용의 위, 아래, 왼쪽, 오른쪽 경계

ratios = []
for f in train_df["file_name"]:
    a = np.asarray(Image.open("train/" + f).convert("L"))  # 사진을 흑백 숫자 배열로
    top, bottom, left, right = content_box(a)              # 내용 영역의 경계
    ratios.append((right - left) / (bottom - top))         # 내용의 가로 ÷ 세로
train_df["ratio"] = ratios

print(train_df.groupby("label")["ratio"].median().round(2))                      # 정상 1.18, 폐렴 1.44
print("비율 1.5 초과인 사진 중 폐렴 비율:", train_df[train_df["ratio"] > 1.5]["label"].mean().round(3))

# %%
# 정상 4장(윗줄), 폐렴 4장(아랫줄) 그려보기 → 폐렴 쪽의 위아래 검은 띠가 더 두꺼움
fig, axes = plt.subplots(2, 4, figsize=(14, 7))
normal_files = train_df[train_df["label"] == 0]["file_name"].tolist()   # 정상 파일 목록
pneu_files = train_df[train_df["label"] == 1]["file_name"].tolist()     # 폐렴 파일 목록
for i in range(4):
    axes[0, i].imshow(Image.open("train/" + normal_files[i * 300]), cmap="gray"); axes[0, i].set_title("normal")
    axes[1, i].imshow(Image.open("train/" + pneu_files[i * 900]), cmap="gray"); axes[1, i].set_title("pneumonia")
for ax in axes.ravel():
    ax.axis("off")          # 눈금 숨기기
plt.show()

# %% [markdown]
# - 정상은 비율 1.18(정사각형에 가까움), 폐렴은 1.44(가로로 김)임
# - 비율이 1.5를 넘는 train 사진은 거의 전부 폐렴임
# - 즉 **폐를 보지 않고 검은 띠의 두께만 봐도** 폐렴을 꽤 맞힐 수 있음 → 모델이 이 단서를 배우지 못하게 해야 함

# %% [markdown]
# ## 3. 이미지 전처리
# 1. **letterbox**: test 원본을 train과 같은 모양(비율 유지 + 검은 띠, 224×224)으로 만듦. train은 이미 이 모양이라 그대로임
# 2. **crop_content**: train·test 모두 검은 띠를 잘라내고 내용만 224×224에 꽉 채움
# 3. 학습할 때만 **비율 흔들기**(`RandomResizedCrop`)로 매번 다른 가로세로 비율로 늘이고 줄임
#
# 사전학습 모델은 컬러(3채널) 사진으로 학습됐기 때문에 흑백 한 장을 3채널로 복사하고, ImageNet 평균/표준편차로 정규화함

# %%
IMG_SIZE = 224    # 모델에 넣을 이미지 크기

def letterbox(img):
    img = img.copy()                                  # 원본을 건드리지 않도록 복사
    img.thumbnail((IMG_SIZE, IMG_SIZE))              # 비율 유지하며 긴 변을 224로 줄이기 (예: 1600×1000 → 224×140)
    canvas = Image.new("L", (IMG_SIZE, IMG_SIZE), 0)  # 224×224 검은 도화지 ("L"=흑백, 0=검정)
    x = (IMG_SIZE - img.size[0]) // 2                 # 왼쪽 시작 위치 (가운데 정렬)
    y = (IMG_SIZE - img.size[1]) // 2                 # 위쪽 시작 위치 (가운데 정렬)
    canvas.paste(img, (x, y))                          # 가운데에 붙이기 → 남는 곳은 검은 띠
    return canvas

def crop_content(img):
    top, bottom, left, right = content_box(np.asarray(img))            # 내용 영역의 경계 (2번에서 만든 함수)
    if bottom - top < 20 or right - left < 20:                         # 내용을 제대로 못 찾으면 그대로
        return img
    img = img.crop((left, top, right, bottom))                         # 내용 부분만 자르기 (검은 띠 제거)
    return img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)            # 224×224에 꽉 채우기

# 학습용: 매번 조금씩 다르게 변형(증강)
train_tf = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.75, 1.0), ratio=(0.75, 1.33)),  # 75~100% 영역을 비율 0.75~1.33으로 무작위로 잘라 224로 → 비율 흔들기
    transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),  # ±10도 회전, 상하좌우 5% 이동, 0.9~1.1배 확대/축소
    transforms.ColorJitter(brightness=0.2, contrast=0.2),                           # 밝기·대비를 ±20% 범위에서 무작위로 변경
    transforms.Grayscale(3),                                                        # 흑백 1채널 → 같은 값 3채널
    transforms.ToTensor(),                                                          # PIL 이미지 → 텐서, 픽셀 0~255 → 0~1
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),             # ImageNet 평균/표준편차로 정규화
])

# 검증/테스트용: 변형 없이
eval_tf = transforms.Compose([
    transforms.Grayscale(3),                                                # 학습 때와 똑같이 3채널로
    transforms.ToTensor(),                                                  # 텐서 변환
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),     # 같은 정규화 (증강 없음 → 매번 같은 결과)
])

# 전처리 결과 확인: test 원본 → letterbox → 검은 띠 제거
img = Image.open("test/" + test_df["file_name"][0]).convert("L")     # test 첫 장을 흑백으로 열기
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(img, cmap="gray"); axes[0].set_title("test original " + str(img.size))
axes[1].imshow(letterbox(img), cmap="gray"); axes[1].set_title("letterbox")
axes[2].imshow(crop_content(letterbox(img)), cmap="gray"); axes[2].set_title("crop_content (model input)")
for ax in axes:
    ax.axis("off")
plt.show()

# %% [markdown]
# ## 4. Dataset 만들기
# - `TrainCSV`: (이미지 텐서, 정답)을 돌려줌
# - `TestCSV`: 이미지 텐서만 돌려줌 (정답 없음)
# - 두 클래스 모두 이미지를 열면 `letterbox → crop_content → 전처리(tf)` 순서로 처리함

# %%
class TrainCSV(Dataset):
    def __init__(self, df, root_dir, tf):
        self.files = df["file_name"].tolist()               # 파일명 목록
        self.labels = df["label"].astype(int).tolist()      # 정답 목록 (0/1)
        self.root, self.tf = root_dir, tf                   # 이미지 폴더, 적용할 전처리
    def __len__(self): return len(self.files)               # 데이터 개수
    def __getitem__(self, i):                               # i번째 데이터를 요청하면
        img = Image.open(os.path.join(self.root, self.files[i])).convert("L")   # 이미지를 흑백으로 열고
        x = self.tf(crop_content(letterbox(img)))           # letterbox → 검은 띠 제거 → 전처리
        y = self.labels[i]                                  # 정답
        return x, y                                         # (이미지 텐서, 정답) 반환

class TestCSV(Dataset):
    def __init__(self, df, root_dir, tf):
        self.files = df["file_name"].tolist()               # 파일명 목록
        self.root, self.tf = root_dir, tf                   # 이미지 폴더, 적용할 전처리
    def __len__(self): return len(self.files)               # 데이터 개수
    def __getitem__(self, i):
        img = Image.open(os.path.join(self.root, self.files[i])).convert("L")   # 이미지를 흑백으로 열고
        x = self.tf(crop_content(letterbox(img)))           # train과 똑같이 letterbox → 검은 띠 제거 → 전처리
        return x                                            # 이미지만 반환

# %% [markdown]
# ## 5. 5-fold 나누기 (환자 묶음 단위)
# - 파일 번호가 환자 순서대로 되어 있어서, **옆 번호끼리는 같은 환자의 사진**인 경우가 많음
# - 사진을 한 장씩 무작위로 나누면 같은 환자가 학습과 검증에 섞여 검증 점수가 부풀려짐 (데이터 누수)
# - 그래서 **연속된 10장을 한 묶음**으로 보고, 묶음이 쪼개지지 않게 5조각으로 나눔
# - 5조각 중 4조각으로 학습하고 1조각으로 검증하기를 5번 반복 → 모델 5개, 모든 train 사진이 한 번씩 검증에 쓰임

# %%
group = train_df.index // 10                               # 연속 10장씩 같은 묶음 번호 (0~9번 → 0, 10~19번 → 1 ...)

skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)   # 5조각, 같은 묶음은 같은 조각에, 정상/폐렴 비율 유지
folds = list(skf.split(train_df, train_df["label"], groups=group))       # [(학습 행 번호, 검증 행 번호), ...] 5쌍

for k, (tr_idx, val_idx) in enumerate(folds):
    print(f"fold {k+1}: 학습 {len(tr_idx)}장, 검증 {len(val_idx)}장, 검증 폐렴 비율 {train_df['label'][val_idx].mean():.3f}")

test_loader = DataLoader(TestCSV(test_df, "test", eval_tf), batch_size=64, shuffle=False)   # test는 예측에만 사용, 순서 유지

# %% [markdown]
# ## 6. 학습 → 검증 → test 예측
# 모델 3개(EfficientNet-B0, ResNet50, DenseNet121)를 각각 5-fold로 학습함. fold마다 아래를 반복함
# 1. ImageNet 사전학습 모델을 불러와 마지막 층을 2개 클래스로 교체
# 2. 클래스 가중치 계산 (정상이 적으므로 정상을 틀리면 더 큰 벌점)
# 3. 학습하면서 매 에포크 검증 정확도를 계산하고, 가장 높았던 에포크의 가중치를 저장
# 4. 그 가중치로 test 폐렴 확률 예측 (원본 + 좌우반전 평균)
#
# - 끝난 fold는 `모델이름_fold번호.npz`로 저장함 → 중간에 멈췄다가 다시 실행하면 끝난 fold는 건너뜀
# - 맥(mps) 기준 EfficientNet-B0 약 3시간, ResNet50 약 6~8시간, DenseNet121 약 6시간 걸림

# %%
EPOCHS = {"efficientnet_b0": 10, "resnet50": 8, "densenet121": 8}   # 모델별 에포크 수 (무거운 두 모델은 조금 적게)
oof = {}                                                   # 모델 이름 → 검증용으로 쓰였을 때의 폐렴 확률 (train 5216장)
test_prob = {}                                             # 모델 이름 → test 폐렴 확률 (5-fold 평균, 624장)

for name in ["efficientnet_b0", "resnet50", "densenet121"]:
    print(f"########## {name} ##########")
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)   # 모델마다 같은 시드에서 시작
    oof[name] = np.zeros(len(train_df))                    # 이 모델의 검증 확률을 담을 배열
    fold_test = []                                         # fold별 test 폐렴 확률 (5개 쌓임)

    for k, (tr_idx, val_idx) in enumerate(folds):
        print(f"===== fold {k+1} / 5 =====")
        done_file = f"{name}_fold{k+1}.npz"                                 # 이 fold의 결과 저장 파일
        if os.path.exists(done_file):                                       # 이미 끝난 fold면 불러오고 건너뛰기
            saved = np.load(done_file)
            oof[name][val_idx] = saved["val_prob"]; fold_test.append(saved["test_prob"])
            print(f"  이미 완료 (검증 정확도 {float(saved['best_acc']):.4f}) → 건너뜀")
            continue

        tr_df, val_df = train_df.iloc[tr_idx], train_df.iloc[val_idx]      # 이번 fold의 학습/검증 표
        train_loader = DataLoader(TrainCSV(tr_df, "train", train_tf), batch_size=32, shuffle=True)    # 학습: 증강 O, 섞기
        val_loader   = DataLoader(TrainCSV(val_df, "train", eval_tf), batch_size=64, shuffle=False)  # 검증: 증강 X

        # 1. 모델 (fold마다 새로 불러와야 이전 fold의 학습 내용이 섞이지 않음)
        if name == "efficientnet_b0":
            model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)   # ImageNet 사전학습 EfficientNet-B0
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)                   # 마지막 층 1000개 클래스 → 2개(정상/폐렴)
        elif name == "resnet50":
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)                 # ImageNet 사전학습 ResNet50
            model.fc = nn.Linear(model.fc.in_features, 2)                                          # ResNet의 마지막 층 이름은 fc → 2개 클래스
        else:
            model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)           # ImageNet 사전학습 DenseNet121
            model.classifier = nn.Linear(model.classifier.in_features, 2)                          # DenseNet의 마지막 층 이름은 classifier → 2개 클래스
        model = model.to(device)                                                                   # 모델을 GPU로

        # 2. 클래스 가중치 + 손실함수 + 옵티마이저
        n0, n1 = (tr_df["label"] == 0).sum(), (tr_df["label"] == 1).sum()                         # 학습용 정상/폐렴 개수
        class_weight = torch.tensor([len(tr_df) / (2 * n0), len(tr_df) / (2 * n1)], dtype=torch.float32).to(device)   # 약 [1.94, 0.67]
        criterion = nn.CrossEntropyLoss(weight=class_weight)                                       # 정상을 틀리면 더 큰 벌점
        optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)                    # 학습률 0.0001
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS[name])            # 학습률을 점점 줄이기

        # 3. 학습 + 검증
        best_acc = 0; best_ep = 0                                          # 최고 검증 정확도와 그때의 에포크
        for ep in range(EPOCHS[name]):
            model.train(); running = 0.0                                   # 학습 모드, 손실 합계 초기화
            for x, y in train_loader:                                      # 32장씩 꺼내서
                x, y = x.to(device), y.to(device)                          # GPU로 옮기고
                loss = criterion(model(x), y)                              # 순전파 + 손실 계산
                optimizer.zero_grad(); loss.backward(); optimizer.step()  # 기울기 초기화 → 역전파 → 가중치 업데이트
                running += loss.item() * x.size(0)                         # 평균 손실 계산용 누적
            scheduler.step()                                               # 에포크마다 학습률 조금 줄이기

            model.eval(); val_prob = []                                    # 평가 모드
            with torch.no_grad():                                          # 기울기 계산 안 함
                for x, y in val_loader:
                    val_prob.extend(torch.softmax(model(x.to(device)), 1)[:, 1].cpu().numpy())   # 검증 사진의 폐렴 확률
            val_prob = np.array(val_prob)
            val_acc = accuracy_score(val_df["label"], val_prob > 0.5)     # 검증 정확도
            print(f"  epoch {ep+1} | loss {running/len(tr_df):.4f} | val acc {val_acc:.4f}")
            if val_acc > best_acc:                                         # 최고 기록이면
                best_acc = val_acc; best_ep = ep + 1                       # 기록 갱신
                oof[name][val_idx] = val_prob                              # 이 fold 검증 사진들의 확률 기록
                torch.save(model.state_dict(), f"best_model_{name}_fold{k+1}.pt")   # 그 시점의 가중치 저장
        print(f"  fold {k+1} 최고 검증 정확도: {best_acc:.4f} (에포크 {best_ep})")

        # 4. 가장 좋았던 가중치로 test 예측 (원본 + 좌우반전 평균)
        model.load_state_dict(torch.load(f"best_model_{name}_fold{k+1}.pt", map_location=device))   # 최고 에포크의 가중치로 되돌리기
        model.eval(); prob = []
        with torch.no_grad():
            for x in test_loader:                                          # 64장씩 (순서 그대로)
                x = x.to(device)
                p1 = torch.softmax(model(x), 1)[:, 1]                          # 원본의 폐렴 확률
                p2 = torch.softmax(model(torch.flip(x, dims=[3])), 1)[:, 1]   # 좌우반전의 폐렴 확률
                prob.extend(((p1 + p2) / 2).cpu().numpy())                   # 두 확률의 평균
        fold_test.append(np.array(prob))                                   # 이번 fold의 test 확률 저장
        np.savez(done_file, val_prob=oof[name][val_idx], best_acc=best_acc, test_prob=np.array(prob))   # fold 결과 저장 → 다시 실행하면 건너뜀
        del model                                                          # 다음 fold 전에 메모리 비우기
        if device.type == "mps": torch.mps.empty_cache()

    test_prob[name] = np.mean(fold_test, axis=0)           # 5개 fold 모델의 test 확률 평균

# %% [markdown]
# ## 7. 검증 결과
# - 모든 train 사진이 한 번씩 검증에 쓰였으므로, 그때의 예측을 모으면 train 전체에 대한 검증 결과가 됨
# - 세 모델의 **폐렴 확률을 평균**낸 결과도 함께 확인함
# - **가로로 긴 정상 사진**(비율 1.35 초과)의 정확도를 따로 봄. 검은 띠 단서대로라면 '폐렴'으로 틀리기 쉬운 사진이라, 여기서 잘 맞히면 모델이 띠가 아닌 폐를 보고 있다는 뜻임

# %%
oof["평균"] = (oof["efficientnet_b0"] + oof["resnet50"] + oof["densenet121"]) / 3   # 세 모델의 검증 확률 평균
y_true = train_df["label"].values                                 # 정답
wide_normal = (y_true == 0) & (train_df["ratio"].values > 1.35)   # 가로로 긴 정상 사진
print("가로로 긴 정상 사진:", wide_normal.sum(), "장")

for name in ["efficientnet_b0", "resnet50", "densenet121", "평균"]:
    pred = (oof[name] > 0.5).astype(int)                          # 기준 0.5로 예측
    cm = confusion_matrix(y_true, pred)                           # 혼동행렬
    print(f"{name:16s} 검증 정확도 {accuracy_score(y_true, pred):.4f} | 정상→폐렴 {cm[0, 1]}장, 폐렴→정상 {cm[1, 0]}장 | "
          f"가로로 긴 정상 {(pred[wide_normal] == 0).mean():.3f}")

for a, b in [("efficientnet_b0", "resnet50"), ("efficientnet_b0", "densenet121"), ("resnet50", "densenet121")]:
    corr = pd.Series(oof[a]).rank().corr(pd.Series(oof[b]).rank())   # 두 모델 검증 확률의 순위 상관
    print(f"순위 상관 {a} - {b}: {corr:.3f}")

# %% [markdown]
# - 세 모델의 순위 상관이 0.80~0.88로 1보다 꽤 낮음 → 서로 다르게 판단하는 사진이 있어서 평균내는 의미가 있음
# - 모델마다 틀리는 방향이 다름. EfficientNet-B0는 폐렴을 가장 많이 놓치고(191장), ResNet50은 정상을 폐렴으로 가장 많이 잘못 판정함(31장)
# - DenseNet121은 가로로 긴 정상 사진을 가장 잘 맞힘(0.971)
# - 평균은 정상→폐렴 오답을 18장으로 낮게 유지하면서, 폐렴을 놓치는 경우는 EfficientNet-B0보다 크게 줄임
# - 가로로 긴 정상 사진도 대부분 정상으로 맞힘 → 검은 띠 단서에 의존하지 않음

# %% [markdown]
# ## 8. 제출 파일 만들기
# - 세 모델의 test 폐렴 확률을 평균내고, **기준 0.5**를 넘으면 폐렴(1)으로 판정함
# - 기준값이나 폐렴 비율을 따로 조정하지 않음

# %%
final_prob = (test_prob["efficientnet_b0"] + test_prob["resnet50"] + test_prob["densenet121"]) / 3   # 세 모델의 test 폐렴 확률 평균
np.save("test_prob.npy", final_prob)                                       # 확률 저장

submission = pd.read_csv("sample_submission.csv")          # 제출 양식 (test 파일명 순서 그대로)
submission["label"] = (final_prob > 0.5).astype(int)       # 0.5 초과 → 1(폐렴), 아니면 0(정상)
submission.to_csv("submission.csv", index=False)           # index=False: 행 번호 없이 저장 → 이 파일을 제출
print("폐렴으로 판정한 비율:", submission["label"].mean().round(3))
submission.head()

# %% [markdown]
# ## 9. Before vs After: Baseline과 비교
# - **Baseline**: 대회에서 준 기본 모델(SimpleCNN). 사진을 128×128로 그냥 늘려 넣고, 증강 없이 2 에포크만 학습함
# - **최종 모델**: 위에서 만든 세 모델(EfficientNet-B0, ResNet50, DenseNet121)의 평균
# - 공정하게 비교하려고 Baseline도 **같은 5-fold**로 학습해서 train 전체에 대한 검증 예측을 만듦
# - 정확도 말고도 **민감도**(실제 폐렴 중 폐렴으로 맞힌 비율)와 **특이도**(실제 정상 중 정상으로 맞힌 비율)를 같이 봄
# - Baseline은 가벼워서 5-fold 전체가 몇 분이면 끝남. 결과는 `baseline.npz`에 저장해서 다시 실행하면 건너뜀

# %%
from sklearn.metrics import roc_auc_score, ConfusionMatrixDisplay   # AUC(확률 순서가 얼마나 정확한지), 혼동행렬 그림

class SimpleCNN(nn.Module):                                # 대회 Baseline 모델
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),    # 흑백 1채널 → 16채널, 128 → 64
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 16채널 → 32채널, 64 → 32
        )
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(32 * 32 * 32, 2))  # 한 줄로 펴서 2개 클래스로
    def forward(self, x):
        return self.fc(self.conv(x))

base_tf = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])   # 128×128로 늘리고 텐서로 (검은 띠 그대로)

class BaseCSV(Dataset):                                    # Baseline용 Dataset: letterbox·띠 제거 없이 그대로 읽음
    def __init__(self, df, root_dir):
        self.files = df["file_name"].tolist()              # 파일명 목록
        self.root = root_dir                               # 이미지 폴더
    def __len__(self): return len(self.files)              # 데이터 개수
    def __getitem__(self, i):
        img = Image.open(os.path.join(self.root, self.files[i])).convert("L")   # 흑백으로 열기
        return base_tf(img)                                # 이미지 텐서만 반환 (정답은 표에서 따로 꺼냄)

def base_predict(model, df, root_dir):
    loader = DataLoader(BaseCSV(df, root_dir), batch_size=64, shuffle=False)    # 순서 그대로 64장씩
    model.eval(); prob = []
    with torch.no_grad():
        for x in loader:
            prob.extend(torch.softmax(model(x.to(device)), 1)[:, 1].cpu().numpy())   # 폐렴 확률
    return np.array(prob)

if os.path.exists("baseline.npz"):                         # 이미 학습했으면 불러오기
    saved = np.load("baseline.npz")
    base_oof, base_test = saved["oof"], saved["test"]
    print("이미 완료 → 건너뜀")
else:
    base_oof = np.zeros(len(train_df)); base_test = []     # 검증 확률(5216장), fold별 test 확률
    for k, (tr_idx, val_idx) in enumerate(folds):          # 최종 모델과 같은 5-fold
        random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)       # 시드 고정
        tr_df, val_df = train_df.iloc[tr_idx], train_df.iloc[val_idx]          # 이번 fold의 학습/검증 표
        ds = BaseCSV(tr_df, "train")                                           # 학습 이미지
        labels = torch.tensor(tr_df["label"].values)                           # 학습 정답
        model = SimpleCNN().to(device)
        criterion = nn.CrossEntropyLoss()                                      # 클래스 가중치 없음
        optimizer = optim.Adam(model.parameters(), lr=1e-3)                    # 학습률 0.001
        for ep in range(2):                                                    # 2 에포크
            model.train()
            order = torch.randperm(len(ds))                                    # 매 에포크 순서 섞기
            for s in range(0, len(ds), 32):                                    # 32장씩
                idx = order[s:s + 32]
                x = torch.stack([ds[int(i)] for i in idx]).to(device)          # 이미지 묶음
                y = labels[idx].to(device)                                     # 정답 묶음
                loss = criterion(model(x), y)                                  # 손실 계산
                optimizer.zero_grad(); loss.backward(); optimizer.step()      # 역전파 → 가중치 업데이트
        base_oof[val_idx] = base_predict(model, val_df, "train")               # 검증 사진의 폐렴 확률
        base_test.append(base_predict(model, test_df, "test"))                 # test 사진의 폐렴 확률
        print(f"fold {k+1} 검증 정확도 {accuracy_score(val_df['label'], base_oof[val_idx] > 0.5):.4f}")
    base_test = np.mean(base_test, axis=0)                 # 5개 fold의 test 확률 평균
    np.savez("baseline.npz", oof=base_oof, test=base_test)  # 저장 → 다시 실행하면 건너뜀

base_sub = pd.read_csv("sample_submission.csv")            # Baseline 제출 파일 (제출 점수를 비교하고 싶을 때 사용)
base_sub["label"] = (base_test > 0.5).astype(int)
base_sub.to_csv("submission_baseline.csv", index=False)

# %%
final_oof = oof["평균"]                                     # 최종 모델의 검증 확률 (세 모델 평균)
rows = []
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, (title, name, prob) in zip(axes, [("Before: Baseline (SimpleCNN)", "Baseline", base_oof),
                                          ("After: 3-model average", "최종 모델", final_oof)]):
    pred = (prob > 0.5).astype(int)                         # 기준 0.5로 예측
    cm = confusion_matrix(y_true, pred)                     # 혼동행렬
    tn, fp, fn, tp = cm.ravel()                             # 정상 맞힘, 정상→폐렴, 폐렴→정상, 폐렴 맞힘
    rows.append([name, accuracy_score(y_true, pred), roc_auc_score(y_true, prob),
                 tp / (tp + fn), tn / (tn + fp), fp, fn,     # 민감도, 특이도, FP, FN
                 (pred[wide_normal] == 0).mean()])           # 가로로 긴 정상 사진의 정확도
    ConfusionMatrixDisplay(cm, display_labels=["NORMAL(0)", "PNEUMONIA(1)"]).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title)
plt.tight_layout(); plt.show()

compare_df = pd.DataFrame(rows, columns=["모델", "검증 정확도", "AUC", "민감도", "특이도", "FP", "FN", "가로로 긴 정상"]).round(4)
print(compare_df.to_string(index=False))
print("Baseline이 test에서 폐렴으로 판정한 비율:", base_sub["label"].mean().round(3))

# %% [markdown]
# ### Before vs After 정리
# | | Baseline | 최종 모델 |
# |---|---|---|
# | 검증 정확도 | 0.9425 | **0.9724** |
# | AUC | 0.9819 | **0.9971** |
# | 민감도 (폐렴을 폐렴으로) | **0.9757** | 0.9675 |
# | 특이도 (정상을 정상으로) | 0.8464 | **0.9866** |
# | FP / FN | 206 / 94 | 18 / 126 |
# | 가로로 긴 정상 사진 정확도 | 0.457 | **0.952** |
# | 제출 점수 | (제출하면 채움) | **0.9583** |
#
# - Baseline도 검증 정확도는 0.94로 높아 보이지만, **가로로 긴 정상 사진은 절반도 못 맞힘(0.457)** → 폐가 아니라 사진 모양(검은 띠)으로 판단한다는 증거임
# - 최종 모델은 같은 사진을 0.952 맞힘 → 검은 띠 제거와 비율 흔들기가 효과가 있었음
# - 가장 크게 좋아진 것은 **특이도**(0.846 → 0.987)임. 정상을 폐렴으로 잘못 판정한 사진이 206장에서 18장으로 줄었음
# - 반대로 **민감도는 조금 내려감**(0.976 → 0.968). 폐렴을 놓친 사진이 94장에서 126장으로 늘었음
# - Baseline은 정상을 폐렴이라고 많이 판정해서 민감도가 높게 나온 것이라, 민감도만 보고 Baseline이 낫다고 할 수는 없음. 다만 최종 모델의 약점이 **폐렴을 놓치는 쪽**이라는 점은 분명함 → 13번의 위험도 기준에서 이 점을 반영함
# - Baseline의 제출 점수는 `submission_baseline.csv`를 제출하면 확인할 수 있음

# %% [markdown]
# ## 10. 최종 모델이 틀린 사진 보기
# - 틀린 사진을 **FP**(정상인데 폐렴으로 판정)와 **FN**(폐렴인데 정상으로 판정)으로 나눔
# - 각각 **가장 확신하면서 틀린 사진** 6장씩 그려봄 (FP는 확률이 높은 순, FN은 확률이 낮은 순)
# - 그림은 모델에 실제로 들어가는 모양(검은 띠 제거 후)으로 그림

# %%
final_pred = (final_oof > 0.5).astype(int)                 # 최종 모델의 예측
fp_idx = np.where((final_pred == 1) & (y_true == 0))[0]    # 정상 → 폐렴으로 틀린 사진 번호
fn_idx = np.where((final_pred == 0) & (y_true == 1))[0]    # 폐렴 → 정상으로 틀린 사진 번호
print(f"틀린 사진 수: {len(fp_idx) + len(fn_idx)}장 (FP {len(fp_idx)}장, FN {len(fn_idx)}장) / 전체 {len(y_true)}장")

fp_top = fp_idx[np.argsort(-final_oof[fp_idx])][:6]        # FP 중 폐렴 확률이 가장 높은 6장
fn_top = fn_idx[np.argsort(final_oof[fn_idx])][:6]         # FN 중 폐렴 확률이 가장 낮은 6장

def model_input(i):
    img = Image.open("train/" + train_df["file_name"][i]).convert("L")   # i번째 train 사진
    return crop_content(letterbox(img))                                  # 모델에 들어가는 모양으로

fig, axes = plt.subplots(2, 6, figsize=(18, 7.5))
for row, (kind, idxs) in enumerate([("FP (Normal -> Pneumonia)", fp_top), ("FN (Pneumonia -> Normal)", fn_top)]):
    for ax, i in zip(axes[row], idxs):
        ax.imshow(model_input(i), cmap="gray")
        ax.set_title(f"{kind}\nprob {final_oof[i]:.2f} | ratio {train_df['ratio'][i]:.2f}", fontsize=9)
for ax in axes.ravel():
    ax.axis("off")
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 11. 틀린 사진의 공통점
# - **확신도** = 폐렴 확률이 기준 0.5에서 얼마나 떨어져 있는지 (0이면 완전히 애매, 0.5면 완전히 확신)
# - 틀린 사진이 애매해서 틀린 건지, 확신하면서 틀린 건지 확인함
# - 사진 모양(가로 ÷ 세로)이 반대 클래스와 비슷해서 틀린 건지도 확인함

# %%
info = pd.DataFrame({"label": y_true, "prob": final_oof, "pred": final_pred, "ratio": train_df["ratio"].values})
info["정답여부"] = np.where(info["pred"] == info["label"], "맞음", "틀림")
info["확신도"] = (info["prob"] - 0.5).abs()                 # 기준 0.5에서 떨어진 거리

print("정답여부별 확신도 평균")
print(info.groupby("정답여부")["확신도"].mean().round(3).to_string())

print("\n라벨·정답여부별 가로÷세로 비율 중앙값")
print(info.groupby(["label", "정답여부"])["ratio"].median().round(2).to_string())

fn_prob = info[(info["label"] == 1) & (info["pred"] == 0)]["prob"]     # 놓친 폐렴 사진들의 폐렴 확률
print(f"\n놓친 폐렴(FN) {len(fn_prob)}장의 확률: 0.3~0.5 {((fn_prob >= 0.3)).sum()}장, 0.1~0.3 {((fn_prob >= 0.1) & (fn_prob < 0.3)).sum()}장, 0.1 미만 {(fn_prob < 0.1).sum()}장")

plt.figure(figsize=(7, 3.5))
plt.hist(info[info["정답여부"] == "틀림"]["prob"], bins=20, range=(0, 1), color="tomato")   # 틀린 사진의 폐렴 확률 분포
plt.axvline(0.5, color="k", linestyle="--")                # 판정 기준 0.5
plt.xlabel("pneumonia probability (wrong predictions)"); plt.ylabel("count"); plt.show()

# %% [markdown]
# ### 실패 케이스 분석
# **틀린 사진 수**: 5,216장 중 144장 (FP 18장, FN 126장)
#
# **1. 대부분 폐렴을 놓친 경우였음**
# - 틀린 144장 중 126장(88%)이 폐렴을 정상으로 판정한 FN임
# - 의료에서는 환자를 놓치는 쪽이 더 위험하므로, 판정 결과만 내보내지 말고 애매한 사진을 따로 표시할 필요가 있음
#
# **2. 애매해서 틀린 사진이 많지만, 확신하면서 틀린 사진도 있었음**
# - 확신도 평균은 맞힌 사진 0.476, 틀린 사진 0.253임 → 틀린 사진은 대체로 기준 0.5에 더 가까웠음
# - 놓친 폐렴 126장 중 55장은 확률 0.3~0.5(기준 바로 아래), 37장은 0.1~0.3임
# - 하지만 34장은 확률 0.1 미만으로, 모델이 **정상이라고 확신하면서** 놓쳤음
# - 이 사진들은 그림에서 폐가 비교적 깨끗해 보임 (눈으로 본 인상이고, 의학적 판단은 아님). 증상이 약한 사진이거나 라벨이 애매한 사진일 가능성이 있음
#
# **3. 사진 모양의 영향이 조금 남아 있음**
# | | 맞힌 사진의 비율 | 틀린 사진의 비율 |
# |---|---|---|
# | 정상 | 1.18 | 1.26 (조금 더 가로로 김) |
# | 폐렴 | 1.45 | 1.35 (조금 더 정사각형에 가까움) |
# - 틀린 사진은 반대 클래스 쪽 모양에 조금 더 가까웠음
# - 다만 차이가 크지 않고(0.08~0.10), 비율 자체가 실제로 병과 관련 있을 수도 있어서 모델이 모양을 본다고 단정할 수는 없음
#
# **4. 정상을 폐렴으로 본 사진(FP)에는 기울어진 사진이 많았음**
# - 확신하면서 틀린 FP 6장 중 여러 장이 사진이 돌아가 있어서 모서리에 검은 삼각형이 보임
# - 폐가 아니라 사진 상태(기울기, 밝기)에 영향을 받았을 가능성이 있음. 18장뿐이라 확정하기는 어려움

# %% [markdown]
# ## 12. Grad-CAM: 모델이 사진의 어디를 보는지
# - **Grad-CAM**은 모델이 판단할 때 사진의 어느 부분을 많이 썼는지 색으로 보여줌 (빨간색일수록 많이 봄)
# - 마지막 합성곱 층의 출력에, "그 출력이 예측에 얼마나 영향을 줬는지(기울기)"를 곱해서 만듦
# - 세 모델이 같은 사진에서 어디를 보는지 나란히 비교함
# - **fold 1 모델**과 **fold 1에서 검증에 쓰인 사진**만 사용함 → 모델이 학습 때 보지 못한 사진임

# %%
import torch.nn.functional as F                            # 히트맵 크기를 키울 때 사용 (interpolate)

def load_model(name, k):
    if name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=None)                            # 구조만 만들고 (사전학습 가중치는 필요 없음)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, 2)         # 마지막 층을 2개 클래스로
    elif name == "resnet50":
        m = models.resnet50(weights=None)
        m.fc = nn.Linear(m.fc.in_features, 2)
    else:
        m = models.densenet121(weights=None)
        m.classifier = nn.Linear(m.classifier.in_features, 2)
    m.load_state_dict(torch.load(f"best_model_{name}_fold{k}.pt", map_location=device))   # 6번에서 저장한 가중치 불러오기
    return m.to(device).eval()                                              # GPU로 옮기고 평가 모드

cam_names = ["efficientnet_b0", "resnet50", "densenet121"]                       # Grad-CAM을 볼 모델 3개
cam_models = {name: load_model(name, 1) for name in cam_names}                   # 모델마다 fold 1 가중치 불러오기
cam_layers = {"efficientnet_b0": cam_models["efficientnet_b0"].features[-1],     # EfficientNet의 마지막 합성곱 블록
              "resnet50": cam_models["resnet50"].layer4[-1],                     # ResNet50의 마지막 합성곱 블록
              "densenet121": cam_models["densenet121"].features.denseblock4}     # DenseNet121의 마지막 합성곱 블록

def grad_cam(name, img):
    model, layer = cam_models[name], cam_layers[name]
    store = {}                                                              # 층의 출력과 기울기를 담아둘 곳
    def hook(module, inp, out):
        store["feat"] = out                                                 # 층의 출력 (7×7 크기의 특징 지도)
        out.register_hook(lambda g: store.update(grad=g))                   # 역전파 때 이 출력의 기울기도 저장
    h = layer.register_forward_hook(hook)                                   # 층에 갈고리 걸기
    x = eval_tf(img).unsqueeze(0).to(device)                                # 이미지 1장을 배치 모양으로
    out = model(x)                                                          # 순전파
    h.remove()                                                              # 갈고리 제거
    prob = torch.softmax(out, 1)[0, 1].item()                               # 폐렴 확률
    model.zero_grad()
    out[0, out.argmax(1).item()].backward()                                 # 모델이 고른 클래스의 점수를 역전파
    weight = store["grad"].mean(dim=(2, 3), keepdim=True)                   # 채널별 중요도 (기울기 평균)
    cam = torch.relu((weight * store["feat"]).sum(dim=1, keepdim=True))     # 중요도를 곱해 더하고 음수는 0으로
    cam = F.interpolate(cam, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)[0, 0]   # 7×7 → 224×224
    cam = cam / (cam.max() + 1e-8)                                          # 0~1로 맞추기
    return cam.detach().cpu().numpy(), prob

val1 = folds[0][1]                                                          # fold 1에서 검증에 쓰인 사진 번호
rng = np.random.RandomState(0)                                              # 사진 고르기용 난수 (고정)
ok_normal = rng.choice(val1[(final_pred[val1] == 0) & (y_true[val1] == 0)], 2, replace=False)   # 정상을 맞힌 사진 2장
ok_pneu = rng.choice(val1[(final_pred[val1] == 1) & (y_true[val1] == 1)], 2, replace=False)     # 폐렴을 맞힌 사진 2장
bad_fp = [i for i in fp_idx[np.argsort(-final_oof[fp_idx])] if i in val1][:2]                   # fold 1의 FP 중 확신이 큰 2장
bad_fn = [i for i in fn_idx[np.argsort(final_oof[fn_idx])] if i in val1][:2]                    # fold 1의 FN 중 확신이 큰 2장
picked = [("Correct Normal", i) for i in ok_normal] + [("Correct Pneumonia", i) for i in ok_pneu] + \
         [("Wrong: FP", i) for i in bad_fp] + [("Wrong: FN", i) for i in bad_fn]

fig, axes = plt.subplots(len(picked), 4, figsize=(13, 3.4 * len(picked)))
for row, (kind, i) in enumerate(picked):
    img = model_input(i)                                                    # 모델에 들어가는 모양의 사진
    axes[row, 0].imshow(img, cmap="gray"); axes[row, 0].set_title(kind)
    for col, name in enumerate(cam_names, start=1):
        cam, prob = grad_cam(name, img)
        axes[row, col].imshow(img, cmap="gray"); axes[row, col].imshow(cam, cmap="jet", alpha=0.4)   # 사진 위에 히트맵 겹치기
        axes[row, col].set_title(f"{name}\nprob {prob:.2f}")
    for ax in axes[row]:
        ax.axis("off")
plt.tight_layout(); plt.show()

# %% [markdown]
# ### 가운데를 보는지 숫자로 확인
# - 사진 몇 장만 눈으로 보면 우연일 수 있어서, fold 1 검증 사진 200장으로 숫자를 계산함
# - 사진의 **가운데 70%×70% 영역**(면적으로는 전체의 약 50%)에 히트맵이 얼마나 몰려 있는지 봄
# - 50%보다 크면 가장자리(어깨, 배, 구석의 글자 표시)보다 가슴 가운데를 더 본다는 뜻임

# %%
center = np.zeros((IMG_SIZE, IMG_SIZE), dtype=bool)                         # 가운데 영역 표시용
m = int(IMG_SIZE * 0.15)                                                    # 바깥 15%씩 제외
center[m:IMG_SIZE - m, m:IMG_SIZE - m] = True                               # 가운데 70%×70%만 True

sample = rng.choice(val1, 200, replace=False)                               # fold 1 검증 사진 중 200장
share = {name: [] for name in cam_names}                                    # 모델별 '가운데 비중' 목록
for i in sample:
    img = model_input(i)
    for name in share:
        cam, _ = grad_cam(name, img)
        share[name].append(cam[center].sum() / (cam.sum() + 1e-8))          # 히트맵 전체 중 가운데 영역의 비중

cam_df = pd.DataFrame(share)
cam_df["label"] = y_true[sample]                                            # 정답
cam_df["정답여부"] = np.where(final_pred[sample] == y_true[sample], "맞음", "틀림")
print(f"가운데 영역의 면적 비율: {center.mean():.2f}")
print("히트맵 중 가운데 영역 비중 (평균)")
print(cam_df[cam_names].mean().round(3).to_string())
print("\n라벨별")
print(cam_df.groupby("label")[cam_names].mean().round(3).to_string())

# %% [markdown]
# ### Grad-CAM 분석
# **1. 세 모델 모두 사진의 가장자리가 아니라 가슴 가운데를 봄**
# - 가운데 영역은 면적의 50%인데, 히트맵은 EfficientNet-B0 86%, ResNet50 73%, DenseNet121 71%가 가운데에 몰려 있었음
# - 구석의 글자 표시(R)나 사진 테두리를 보고 판단하는 것은 아님
#
# **2. 세 모델이 보는 방식이 다름**
# - EfficientNet-B0는 가슴 가운데에 큰 덩어리 하나로 나타남
# - ResNet50은 작은 점 여러 개로 나뉘어 나타나고, 폐렴 사진에서는 가운데 비중이 0.70으로 더 낮음 (아래쪽이나 옆쪽에도 점이 생김)
# - DenseNet121은 가장 넓게 퍼져서 폐 영역 전체를 덮는 모양임
# - 7번에서 모델끼리 순위 상관이 0.80~0.88로 낮았던 것과 맞는 결과임 → 서로 다른 곳을 보기 때문에 평균내는 효과가 있음
#
# **3. 폐의 병변을 정확히 본다고 말할 수는 없음**
# - 정상을 맞힌 사진에서도 히트맵은 폐보다 가운데(심장·척추 부근)에 있는 경우가 많았음
# - 틀린 사진(FP, FN)이라고 해서 히트맵 위치가 눈에 띄게 다르지 않았음 → Grad-CAM만으로는 왜 틀렸는지 설명하기 어려움
#
# **한계**
# - Grad-CAM은 7×7 크기의 지도를 224×224로 키운 것이라 위치를 대략적으로만 보여줌
# - 그림으로는 8장, 숫자로는 200장만 확인했고 모델도 fold 1 하나씩만 봄
# - 모델마다 히트맵을 뽑는 층의 성격이 달라서, 퍼진 정도의 차이가 전부 "보는 곳의 차이"라고 할 수는 없음
# - 따라서 히트맵은 "모델이 엉뚱한 곳(테두리, 글자)을 보지 않는다"는 확인 용도로만 쓰고, 진단 근거로 그대로 믿으면 안 됨

# %% [markdown]
# ## 13. 위험도(Risk Level) 정하기
# - 폐렴 확률을 **Low / Medium / High** 3단계로 나눔
# - **High**: 0.5 이상 (모델이 폐렴으로 판정) → 빠른 확인 필요
# - **Low**와 **Medium**을 나누는 기준은 검증 결과를 보고 정함. 기준을 여러 개 넣어보고 아래 두 가지를 비교함
#   - Low에 들어간 **실제 폐렴** 수 → 적을수록 좋음 (안전하다고 한 사람 중에 환자가 있으면 안 됨)
#   - **Medium** 장수 → 적을수록 좋음 (의사가 직접 확인해야 하는 사진)

# %%
for t in [0.02, 0.05, 0.1, 0.15, 0.2, 0.3]:                                 # Low 기준 후보
    low = final_oof < t                                                     # Low로 분류되는 사진
    med = (final_oof >= t) & (final_oof < 0.5)                              # Medium으로 분류되는 사진
    print(f"Low 기준 {t:<4} | Low 속 폐렴 {(low & (y_true == 1)).sum():3d}장 | "
          f"Medium {med.sum():3d}장 ({med.mean() * 100:.1f}%), 그중 폐렴 {(med & (y_true == 1)).sum():3d}장")

# %%
def get_risk_level(prob):
    if prob < 0.05:               # 폐렴 확률 5% 미만
        return "Low"              # 정상 가능성 높음
    elif prob < 0.5:              # 5% 이상 50% 미만
        return "Medium"           # 애매함 → 의사 확인 필요
    else:                         # 50% 이상
        return "High"             # 폐렴 가능성 높음 → 빠른 확인 필요

risk_df = pd.DataFrame({"실제": np.where(y_true == 1, "폐렴", "정상"), "prob": final_oof})
risk_df["risk"] = risk_df["prob"].apply(get_risk_level)                     # 사진마다 위험도 붙이기
table = pd.crosstab(risk_df["risk"], risk_df["실제"]).reindex(["Low", "Medium", "High"])   # 위험도별 실제 정상/폐렴 수
print(table)
print(f"\n놓친 폐렴(FN) {len(fn_idx)}장 중 Medium에 들어간 수: {(final_oof[fn_idx] >= 0.05).sum()}장")

# %% [markdown]
# ### 위험도 기준
# | 위험도 | 폐렴 확률 | 의미 |
# |---|---|---|
# | Low | 0.05 미만 | 정상 가능성 높음 |
# | Medium | 0.05 이상 0.5 미만 | 애매함 → 의사 확인 필요 |
# | High | 0.5 이상 | 폐렴 가능성 높음 → 빠른 확인 필요 |
#
# **검증 결과 (train 5,216장)**
# | 위험도 | 실제 정상 | 실제 폐렴 |
# |---|---|---|
# | Low | 1,194 | 21 |
# | Medium | 129 | 105 |
# | High | 18 | 3,749 |
#
# **기준을 이렇게 정한 이유**
# - 이 모델의 약점은 폐렴을 놓치는 것(FN 126장)이므로, 놓친 환자가 **Low가 아니라 Medium**에 들어가게 하는 것이 목표임
# - Low 기준을 0.05로 잡으면 놓친 폐렴 126장 중 **105장이 Medium**에 들어가 의사 확인을 거치게 됨
# - 의사가 직접 봐야 하는 Medium은 234장으로 전체의 4.5%임
#
# **남는 문제**
# - Low에 실제 폐렴이 **21장** 남음 (전체 폐렴 3,875장의 0.5%). 모델이 확률 0.05 미만으로 확신한 사진이라 기준을 조정해서는 걸러내기 어려움
# - 기준을 0.02로 더 낮춰도 10장이 남고, Medium은 336장(6.4%)으로 늘어남
# - 반대로 기준을 0.15로 높이면 Medium은 131장으로 줄지만 Low 속 폐렴이 44장으로 늘어남
# - 즉 **Low라고 해서 폐렴이 아니라고 보장할 수는 없음**. 서비스에서는 Low도 "정상 확정"이 아니라 "우선순위 낮음"으로 표시해야 함
#
# **주의할 점**
# - 이 기준은 train 데이터의 검증 결과로 정했음. train은 폐렴이 74%인데 실제 병원은 비율이 다르므로 그대로 쓰면 안 되고 다시 확인해야 함
# - test 사진은 Low 138장, Medium 80장(13%), High 406장으로 나뉨. Medium 비율이 검증(4.5%)보다 높음 → 모델이 test 사진에서 더 자주 애매해한다는 뜻이고, 사진이 달라지면 기준도 다시 봐야 한다는 예시임

# %% [markdown]
# ## 14. 서비스에 저장할 결과 데이터
# - 사진 한 장을 예측했을 때 서비스(DB)에 저장할 항목을 정하고, 실제로 만들어봄
# - 새로 들어온 사진 역할은 **test 사진**이 맡음 (정답 없이 예측만 하는 상황과 같음)
# - Grad-CAM은 모델 15개의 평균으로는 한 장으로 만들기 어려워서 **ResNet50 fold 1 모델**로 만듦

# %%
import datetime                                                             # 예측 시간 기록용

os.makedirs("results", exist_ok=True)                                       # Grad-CAM 그림을 저장할 폴더

def make_result(i):
    f = test_df["file_name"][i]                                             # i번째 test 사진 파일명
    prob = float(final_prob[i])                                             # 예측 확률: 8번에서 만든 세 모델 평균
    img = crop_content(letterbox(Image.open("test/" + f).convert("L")))     # 모델에 들어가는 모양으로 전처리
    cam, _ = grad_cam("resnet50", img)                                      # Grad-CAM 히트맵 (12번 함수)
    cam_path = "results/gradcam_" + f                                       # 히트맵 그림 저장 경로
    plt.figure(figsize=(3, 3))
    plt.imshow(img, cmap="gray"); plt.imshow(cam, cmap="jet", alpha=0.4); plt.axis("off")
    plt.savefig(cam_path, bbox_inches="tight", pad_inches=0); plt.close()   # 그림 파일로 저장하고 닫기
    return {
        "image_id": f,                                                      # 사진 이름
        "original_image": "test/" + f,                                      # 원본 이미지 경로
        "gradcam_image": cam_path,                                          # Grad-CAM 이미지 경로
        "pred_label": int(prob > 0.5),                                      # 예측 결과: 0 = 정상, 1 = 폐렴
        "pred_name": "PNEUMONIA" if prob > 0.5 else "NORMAL",               # 예측 결과 이름
        "probability": round(prob, 4),                                      # 폐렴 확률
        "risk_level": get_risk_level(prob),                                 # 위험도 (13번 함수)
        "model_version": "effb0+resnet50+densenet121_5fold",                  # 어떤 모델로 예측했는지
        "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),   # 예측한 시간
    }

test_risk = np.array([get_risk_level(p) for p in final_prob])              # test 사진들의 위험도
print(pd.Series(test_risk).value_counts().reindex(["Low", "Medium", "High"]).to_string())   # 위험도별 test 사진 수

examples = [np.where(test_risk == r)[0][0] for r in ["Low", "Medium", "High"]]   # 위험도별로 1장씩 고르기
result_df = pd.DataFrame([make_result(i) for i in examples])               # 결과 데이터 3건
result_df.to_csv("results/result_examples.csv", index=False)               # 표로 저장
result_df

# %% [markdown]
# ### 모델 결과 데이터 정의
# | 데이터 | 코드에서 만들어지는 위치 | 의미 |
# |---|---|---|
# | 원본 이미지 (`original_image`) | 14번 `make_result` | 사용자가 올린 X-ray 사진. 나중에 다시 확인하거나 재학습할 때 필요 |
# | Grad-CAM 이미지 (`gradcam_image`) | 12번 `grad_cam` → 14번에서 저장 | 모델이 어디를 보고 판단했는지 보여주는 히트맵. 엉뚱한 곳을 본 것은 아닌지 확인할 때 사용 |
# | 예측 결과 (`pred_label`, `pred_name`) | 14번 `prob > 0.5` | 정상(0) / 폐렴(1) |
# | 예측 확률 (`probability`) | 8번 `final_prob` | 폐렴일 가능성. 같은 "폐렴"이어도 0.55와 0.99는 의미가 다름 |
# | 위험도 (`risk_level`) | 13번 `get_risk_level` | 확률을 Low / Medium / High로 나눈 것. 의사가 볼 순서를 정할 때 사용 |
# | 모델 버전 (`model_version`) | 14번 | 어떤 모델로 예측했는지. 모델이 바뀌어도 예전 결과를 추적할 수 있음 |
# | 예측 시간 (`created_at`) | 14번 | 언제 예측했는지 |
#
# - 예측 확률은 모델 15개(3종류 × 5-fold)의 평균이고, Grad-CAM은 그중 ResNet50 fold 1 모델 하나로 만든 것임 → 히트맵이 최종 판단 전체를 설명하지는 않음
# - 예시 3건은 `results/result_examples.csv`, 히트맵 그림은 `results/` 폴더에 저장됨
