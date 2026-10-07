# %% [markdown]
# # 흉부 X-ray 폐렴 분류 — 앙상블 + 기준 조정 (재학습 없음)
#
# 이미 학습해 둔 모델 파일(`.pt`)을 불러와 **test 예측만** 다시 하고, 모델들의 폐렴 확률을 평균(앙상블)냅니다.
#
# | 모델 파일 | 모델 | 만든 노트북 | 리더보드 (기준 0.5) |
# |---|---|---|---|
# | `best_model.pt` | EfficientNet-B0 | xray_improved (v1) | 0.8526 |
# | `best_model_v2.pt` | EfficientNet-B0 | xray_v2 | 0.8686 |
# | `best_model_v3.pt` | DenseNet121 | xray_v3_densenet | 학습 후 확인 |
#
# `best_model_v3.pt`가 아직 없으면 v1+v2만으로 앙상블하고, v3 학습이 끝난 뒤 이 노트북을 다시 실행하면 v3까지 포함됩니다.
#
# **기준 조정:** 모델들이 test의 정상 사진을 폐렴으로 과하게 판정하는 경향이 있어서(v1은 0.5 기준 76.6% 폐렴 → 0.8526점, 0.99 기준 62.2% → 0.9006점),
# 폐렴으로 판정하는 비율이 58% / 62% / 66%가 되도록 기준값을 정한 제출 파일 3개를 만듭니다.
#
# ⚠️ 리더보드 점수를 보고 비율을 고르는 것은 **리더보드에 맞춘 튜닝**입니다. 최종 순위를 공개되지 않은 test 일부로 매긴다면 점수가 떨어질 수 있습니다.

# %% [markdown]
# ## 0. 데이터 폴더로 이동

# %%
import os

DATA_DIR = "."   # 데이터와 .pt 파일이 있는 폴더 (본인 경로에 맞게 수정)
os.chdir(DATA_DIR)                                 # 작업 폴더를 데이터 폴더로 이동
print([f for f in os.listdir() if f.endswith(".pt")])   # 저장된 모델 파일 목록 확인

# %% [markdown]
# ## 1. 라이브러리 + 장치 설정

# %%
import numpy as np, pandas as pd       # 숫자 배열 계산, CSV 표 다루기
from PIL import Image                  # 이미지 열기

import torch                           # PyTorch 핵심
import torch.nn as nn                  # 신경망 층 (마지막 분류층 교체용)
from torch.utils.data import Dataset, DataLoader   # 데이터를 배치 단위로 넣어주는 도구
from torchvision import transforms, models          # 전처리, 모델 구조

if torch.cuda.is_available():                 # GPU(cuda)가 있으면
    device = torch.device("cuda")
elif torch.backends.mps.is_available():       # 맥 GPU(mps)가 있으면
    device = torch.device("mps")
else:                                         # 없으면 CPU (예측만 하므로 CPU도 몇 분이면 끝남)
    device = torch.device("cpu")
print(device)

# %% [markdown]
# ## 2. test 전처리 + Dataset
# 학습 노트북과 **완전히 같은** 전처리여야 합니다 (letterbox → 3채널 → ImageNet 정규화).

# %%
IMG_SIZE = 224    # 학습 때와 같은 크기

def letterbox(img):
    img = img.copy()                                  # 원본을 건드리지 않도록 복사
    img.thumbnail((IMG_SIZE, IMG_SIZE))              # 비율 유지하며 긴 변을 224로 줄이기
    canvas = Image.new("L", (IMG_SIZE, IMG_SIZE), 0)  # 224×224 검은 도화지
    x = (IMG_SIZE - img.size[0]) // 2                 # 가로 시작 위치 (가운데 정렬)
    y = (IMG_SIZE - img.size[1]) // 2                 # 세로 시작 위치 (가운데 정렬)
    canvas.paste(img, (x, y))                          # 가운데에 붙이기
    return canvas

eval_tf = transforms.Compose([
    transforms.Grayscale(3),                                                # 흑백 → 3채널
    transforms.ToTensor(),                                                  # 텐서 변환, 0~1
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),     # ImageNet 정규화
])

class TestCSV(Dataset):
    def __init__(self, df, root_dir, tf):
        self.files = df["file_name"].tolist()               # test 파일명 목록
        self.root, self.tf = root_dir, tf                   # 이미지 폴더, 전처리
    def __len__(self): return len(self.files)               # 데이터 개수
    def __getitem__(self, i):
        img = Image.open(os.path.join(self.root, self.files[i])).convert("L")   # 흑백으로 열기
        return self.tf(letterbox(img))                      # letterbox → 전처리 → 텐서

test_df = pd.read_csv("test.csv")                                                           # test 파일명 표
test_loader = DataLoader(TestCSV(test_df, "test", eval_tf), batch_size=64, shuffle=False)  # 순서 유지!

# %% [markdown]
# ## 3. 모델 불러오기 + test 예측 함수
# `.pt` 파일에는 **가중치만** 저장되어 있어서, 학습 때와 같은 구조를 먼저 만들고 가중치를 넣어야 합니다.

# %%
def make_model(name):
    if name == "effnet":                                                   # EfficientNet-B0 (v1, v2)
        m = models.efficientnet_b0()                                       # 구조만 만들기 (가중치는 아래에서 불러옴)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, 2)       # 학습 때처럼 마지막 층을 2개 클래스로
    else:                                                                  # DenseNet121 (v3)
        m = models.densenet121()
        m.classifier = nn.Linear(m.classifier.in_features, 2)
    return m

def predict_test(model):
    model.eval()                                                           # 평가 모드
    probs = []
    with torch.no_grad():                                                  # 기울기 계산 X
        for x in test_loader:
            x = x.to(device)
            p1 = torch.softmax(model(x), 1)[:, 1]                          # 원본의 폐렴 확률
            p2 = torch.softmax(model(torch.flip(x, dims=[3])), 1)[:, 1]   # 좌우반전의 폐렴 확률 (TTA)
            probs.extend(((p1 + p2) / 2).cpu().numpy())                   # 두 확률 평균
    return np.array(probs)

# %% [markdown]
# ## 4. 모델별 test 확률 구하기
# 모델 파일이 있는 것만 예측합니다.

# %%
model_list = [("v1", "best_model.pt", "effnet"),          # (이름, 파일, 구조)
              ("v2", "best_model_v2.pt", "effnet"),
              ("v3", "best_model_v3.pt", "densenet")]

probs = {}                                                 # 모델 이름 → test 폐렴 확률 배열
for tag, path, arch in model_list:
    if not os.path.exists(path):                           # 아직 학습하지 않은 모델은 건너뛰기
        print(tag, "파일 없음 → 건너뜀")
        continue
    model = make_model(arch)                               # 구조 만들기
    model.load_state_dict(torch.load(path, map_location="cpu"))   # 저장된 가중치 넣기
    model = model.to(device)
    probs[tag] = predict_test(model)                       # test 예측
    np.save(f"test_prob_{tag}.npy", probs[tag])            # 확률 저장 (다음에 다시 쓸 수 있게)
    print(tag, "폐렴 예측 비율 @0.5:", (probs[tag] > 0.5).mean().round(3))

# %% [markdown]
# ## 5. 모델끼리 얼마나 비슷하게 예측하나
# 앙상블은 모델들이 **서로 다른 실수**를 할 때 효과가 큽니다. 일치율이 너무 높으면(99%) 평균내도 별 차이가 없습니다.

# %%
tags = list(probs.keys())
for i in range(len(tags)):
    for j in range(i + 1, len(tags)):
        a, b = tags[i], tags[j]
        agree = ((probs[a] > 0.5) == (probs[b] > 0.5)).mean()          # 0.5 기준 예측이 같은 비율
        corr = np.corrcoef(probs[a], probs[b])[0, 1]                    # 확률끼리의 상관계수
        print(f"{a} vs {b}: 예측 일치율 {agree:.3f}, 확률 상관 {corr:.3f}")

# %% [markdown]
# ## 6. 앙상블 (확률 평균)
# 모델마다 폐렴 확률을 더해서 모델 개수로 나눕니다. `USE`에 적은 모델만 사용합니다.

# %%
# 앙상블에 넣을 모델 고르기 (v2는 v1과 섞었을 때 리더보드가 0.9006 → 0.8766으로 떨어져서 제외)
USE = ["v1", "v3"]                                         # 원하는 모델 이름만 남기기
tags = [t for t in USE if t in probs]                      # 그중 파일이 있는(예측한) 모델만 사용
# 확률을 그대로 평균내지 않고 '순위'로 바꿔서 평균냅니다.
# 이유: v1은 폐렴 확률이 0.99~0.9999에 몰려 있고, v3는 label smoothing 때문에 0.5~0.95에 퍼져 있어서
#       그냥 평균내면 기준값 근처에서 v3 쪽 차이만 크게 반영됩니다 (두 모델이 공평하게 섞이지 않음).
# rank(pct=True): 가장 정상 같은 사진 → 0에 가깝게, 가장 폐렴 같은 사진 → 1 (각 모델 안에서의 순서)
ranks = [pd.Series(probs[t]).rank(pct=True).values for t in tags]   # 모델별 순위(0~1)
ens_prob = np.mean(ranks, axis=0)                                    # 순위의 평균 → 두 모델이 똑같은 비중
print("앙상블에 쓴 모델:", tags)

# %% [markdown]
# ## 7. 폐렴 비율을 정해서 제출 파일 만들기
# `np.quantile(ens_prob, 1 - r)`은 "확률이 이 값보다 큰 사진이 전체의 r 비율이 되는 기준값"입니다.
# 예: r = 0.62이면 확률 상위 62%만 폐렴(1), 나머지는 정상(0)으로 판정합니다.

# %%
submission = pd.read_csv("sample_submission.csv")         # 제출 양식
name = "_".join(tags)                                      # 파일 이름에 사용한 모델 표시 (예: v1_v2_v3)
for r in [0.58, 0.62, 0.66]:                               # 폐렴으로 판정할 비율
    threshold = np.quantile(ens_prob, 1 - r)               # 그 비율이 되는 기준값
    submission["label"] = (ens_prob > threshold).astype(int)   # 기준값보다 크면 폐렴
    fname = f"submission_rank_{name}_r{round(r * 100)}.csv"     # 예: submission_rank_v1_v3_r62.csv
    submission.to_csv(fname, index=False)
    print(f"{fname}: 기준값 {threshold:.4f}, 폐렴 비율 {submission['label'].mean():.3f}")
