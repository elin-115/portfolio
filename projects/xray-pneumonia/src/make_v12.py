# xray_v11_epochs.py 를 바탕으로 xray_v12_xrv_nobars.py 를 만든다
# (v11과 같은 전처리·fold, 모델만 X-ray 전용 DenseNet121로 교체 + v11과 평균)
src = open('xray_v11_epochs.py').read()

def sub(old, new, count=1):
    global src
    assert src.count(old) >= 1, old[:60]      # 바꿀 문장이 없으면 바로 멈춤
    src = src.replace(old, new) if count == 0 else src.replace(old, new, count)

# ── 머리말 ──────────────────────────────────────────────
head_end = src.index('# %% [markdown]\n# ## 0. 데이터 준비')
src = '''# %% [markdown]
# # 흉부 X-ray 폐렴 분류 — v12 (X-ray 전용 모델 × 검은 띠 제거)
#
# v11(검은 띠 제거 + 10 에포크)은 **모델 하나, 기준 0.5, 보정 없이** 리더보드 **0.9471**이 나왔습니다.
# 반대로 v10을 예전 모델들(v5·v8·v9)과 섞은 파일은 0.9183으로 더 낮았습니다. 예전 모델들은 검은 띠 단서를 배운 모델이라, 섞으면 그 단서가 다시 들어오기 때문으로 보입니다.
#
# 그래서 v12는 **검은 띠를 제거한 모델끼리만** 섞을 수 있도록 두 번째 모델을 만듭니다.
#
# | 버전 | 모델 | 전처리 | 리더보드 |
# |---|---|---|---|
# | v5 | DenseNet121 (X-ray 전용 사전학습) | 검은 띠 있음 | v4와 섞어 0.9119 |
# | v11 | EfficientNet-B0 (ImageNet) | **검은 띠 제거** | **0.9471** |
# | **v12** | DenseNet121 (**X-ray 전용 사전학습**) | **검은 띠 제거** | 이번 노트북 |
#
# ## v12에서 바꾼 점 (v11 기준)
# - **모델만 교체**: EfficientNet-B0 → 흉부 X-ray로 사전학습된 DenseNet121 (v5·v9에서 쓴 모델).
# - 그에 맞춰 입력 형식 변경: 흑백 **1채널**, 픽셀 값 **-1024 ~ 1024** (v5와 동일).
# - 에포크 10 → **8** (DenseNet이 더 느려서. fold마다 가장 좋았던 에포크를 출력합니다).
# - 전처리(검은 띠 제거·비율 흔들기), fold 나누기, 손실함수, 학습률은 v11과 **완전히 같습니다.**
#
# ## 제출 파일 (둘 다 기준 0.5, 비율 보정 없음)
# - `submission_v12_t05.csv` — v12 단독
# - `submission_v11_v12_t05.csv` — v11과 v12의 폐렴 확률 평균
#
# 어느 쪽이 나은지는 7번 셀에서 **test를 쓰지 않고** train 검증(OOF) 결과로 먼저 확인합니다.
#
# 끝난 fold는 `v12_fold번호.npz`로 저장해 이어서 실행할 수 있습니다. test는 예측에만 사용합니다.
# 예상 시간 약 4시간 (v9가 5 에포크에 약 2시간 30분).

# %% [markdown]
# ## 라이브러리 설치 (처음 한 번만)
# v5·v8·v9에서 이미 설치했다면 건너뛰어도 됩니다.

# %%
%pip install torchxrayvision

''' + src[head_end:]

# ── 1. 라이브러리 ───────────────────────────────────────
sub('baseline과 같고, 사전학습 모델을 쓰기 위해 `torchvision.models`,',
    'baseline과 같고, X-ray 전용 사전학습 모델을 쓰기 위해 `torchxrayvision`,')
sub('import torch.nn as nn                               # 신경망 층(Linear 등)과 손실함수',
    'import torchxrayvision as xrv                       # 흉부 X-ray 전용 사전학습 모델 모음\n'
    'import torch.nn as nn                               # 신경망 층(Linear 등)과 손실함수')
sub('from torchvision import transforms, models          # transforms: 이미지 전처리·증강 / models: 사전학습 모델 모음',
    'from torchvision import transforms                  # 이미지 전처리·증강')

# ── 2. 전처리 ───────────────────────────────────────────
sub('**crop_content (v10 추가)**', '**crop_content (v10과 동일)**')
sub('# 사전학습 모델은 컬러(3채널) 사진으로 학습됐기 때문에 `Grayscale(3)`으로 흑백 한 장을 3채널로 복사하고, ImageNet 평균/표준편차로 정규화합니다.',
    '# X-ray 전용 모델은 **흑백 1채널**, 픽셀 값 **-1024 ~ 1024** 범위로 학습됐습니다. 그래서 `ToTensor()`로 0~1이 된 값을 `× 2048 − 1024` 해서 같은 범위로 맞춥니다 (v5와 동일).\n'
    '# v11에 있던 `Grayscale(3)`(3채널 복사)과 ImageNet 정규화는 쓰지 않습니다.')
sub('''    transforms.Grayscale(3),                                                        # 흑백 1채널 → 같은 값 3채널 (사전학습 모델은 RGB 입력)
    transforms.ToTensor(),                                                          # PIL 이미지 → 텐서, 픽셀 0~255 → 0~1
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),             # ImageNet 평균/표준편차로 정규화 (사전학습 때와 같은 기준)
''', '''    transforms.ToTensor(),                                                          # PIL 흑백 이미지 → 1채널 텐서, 픽셀 0~255 → 0~1
    transforms.Lambda(lambda x: x * 2048 - 1024),                                   # 0~1 → -1024~1024 (X-ray 전용 모델이 학습된 범위)
''')
sub('''    transforms.Grayscale(3),                                                # 학습 때와 똑같이 3채널로
    transforms.ToTensor(),                                                  # 텐서 변환
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),     # 같은 정규화 (증강은 하지 않음 → 매번 같은 결과)
''', '''    transforms.ToTensor(),                                                  # 1채널 텐서, 0~1
    transforms.Lambda(lambda x: x * 2048 - 1024),                           # 학습 때와 같은 범위 변환 (증강은 하지 않음 → 매번 같은 결과)
''')

# ── 5. 학습 ─────────────────────────────────────────────
sub('''# fold마다 아래를 반복합니다. 모델·학습 설정은 v4와 같고 전처리만 다릅니다.
# 1. 새 EfficientNet-B0 불러오기 (fold마다 처음부터)
# 2. 클래스 가중치 계산 (그 fold의 학습 데이터로만)
# 3. 10 에포크 학습, 검증 정확도가 가장 높은 에포크의 가중치 저장''',
'''# fold마다 아래를 반복합니다. v11과 같고 1번(모델)만 X-ray 전용 DenseNet121로 바뀌었습니다.
# 0. 이미 끝난 fold면(`v12_fold번호.npz` 파일이 있으면) 저장된 결과를 불러오고 건너뛰기
# 1. 새 X-ray 전용 DenseNet121 불러오기 (fold마다 처음부터)
# 2. 클래스 가중치 계산 (그 fold의 학습 데이터로만)
# 3. 8 에포크 학습, 검증 정확도가 가장 높은 에포크의 가중치 저장''')
sub('EPOCHS = 10                                                # fold마다 10번 반복 학습 (v10은 6번)',
    'EPOCHS = 8                                                 # fold마다 8번 반복 학습 (v11은 10번, v5·v9는 5번)')
sub('''    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)   # ImageNet 사전학습 가중치
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)                   # 마지막 층 → 2개 클래스
''', '''    model = xrv.models.DenseNet(weights="densenet121-res224-all")   # 흉부 X-ray 8개 데이터셋으로 사전학습된 DenseNet121
    model.classifier = nn.Linear(model.classifier.in_features, 2)   # 원래 18개 소견 예측층 → 2개 클래스(정상/폐렴)로 교체
    model.op_threshs = None                                          # 원래 소견용 확률 보정값 끄기
    model.apply_sigmoid = False                                      # 모델 안에서 sigmoid 적용 안 함 (CrossEntropyLoss가 처리)
''')
sub('v11_fold', 'v12_fold', 0)                 # 결과 저장 파일 이름 (아래 7번 셀에서 v11 것은 따로 다시 적음)

# ── 7·8. 비교와 제출 (새로 작성) ─────────────────────────
tail_start = src.index('# %% [markdown]\n# ## 7. v10과 비교')
src = src[:tail_start] + '''# %% [markdown]
# ## 7. v11과 비교, 그리고 섞었을 때 (test 없이 확인)
# 저장된 v11의 검증 확률(`v11_fold1~5.npz`)을 불러와 **같은 사진**으로 비교합니다. fold 나누기가 v11과 같아서 그대로 비교할 수 있습니다.
# - **가로로 긴 정상** (비율 > 1.35): 검은 띠 단서대로라면 틀리기 쉬운 사진
# - **정사각에 가까운 폐렴** (비율 < 1.2): 마찬가지로 틀리기 쉬운 사진
#
# 마지막 줄 `v11+v12 평균`의 정확도가 둘 중 나은 쪽보다 높으면 섞을 가치가 있다는 뜻입니다.
# 두 모델의 순위 상관도 함께 출력합니다 (낮을수록 서로 다른 것을 보고 있어서 섞는 효과가 큽니다. 지금까지는 0.87 근처에서 효과가 있었고 0.93 이상이면 효과가 없었습니다).

# %%
def content_ratio(fname):
    a = np.asarray(Image.open(os.path.join("train", fname)).convert("L"))   # train 사진을 흑백 숫자 배열로
    bright = a > 8                                                           # 검정이 아닌 픽셀
    rows = np.where(bright.mean(axis=1) > 0.05)[0]; cols = np.where(bright.mean(axis=0) > 0.05)[0]   # 내용이 있는 행·열
    return (cols[-1] - cols[0] + 1) / (rows[-1] - rows[0] + 1)           # 내용의 가로 ÷ 세로

train_df["ratio"] = [content_ratio(f) for f in train_df["file_name"]]    # 사진별 가로세로 비율
wide_normal = ((train_df["label"] == 0) & (train_df["ratio"] > 1.35)).values   # 가로로 긴 정상
square_pneu = ((train_df["label"] == 1) & (train_df["ratio"] < 1.2)).values    # 정사각에 가까운 폐렴

def report(name, prob):
    pred = (prob > 0.5).astype(int)                                       # 0.5 기준 예측
    cm = confusion_matrix(train_df["label"], pred)                        # 혼동행렬
    print(f"{name}: 전체 {accuracy_score(train_df['label'], pred):.4f} | 정상→폐렴 {cm[0, 1]}장, 폐렴→정상 {cm[1, 0]}장 | "
          f"가로로 긴 정상 {(pred[wide_normal] == 0).mean():.3f} | 정사각 폐렴 {(pred[square_pneu] == 1).mean():.3f}")

has_v11 = all(os.path.exists(f"v11_fold{k+1}.npz") for k in range(5))    # v11 결과 파일 5개가 모두 있는지
if has_v11:
    v11_oof = np.zeros(len(train_df))                                     # v11의 검증 확률을 담을 배열
    for k, (tr_idx, val_idx) in enumerate(folds):
        v11_oof[val_idx] = np.load(f"v11_fold{k+1}.npz")["val_prob"]     # v11의 검증 확률 불러오기
    report("v11 (EfficientNet)  ", v11_oof)
report("v12 (X-ray DenseNet)", oof_prob)
if has_v11:
    report("v11+v12 평균        ", (v11_oof + oof_prob) / 2)              # 두 모델의 폐렴 확률 평균
    corr = pd.Series(v11_oof).rank().corr(pd.Series(oof_prob).rank())    # 검증 확률의 순위 상관
    print(f"v11-v12 순위 상관 (검증): {corr:.3f}")

# %% [markdown]
# ## 8. 제출 파일
# v10부터는 기준 0.5에서 폐렴 판정 비율이 자연스럽게 나오므로, **비율을 맞추는 보정 없이 기준 0.5**를 그대로 씁니다.
# - `submission_v12_t05.csv` — v12 단독
# - `submission_v11_v12_t05.csv` — v11과 v12의 폐렴 확률 평균
#
# 둘 중 무엇을 낼지는 **7번 셀의 검증 결과**로 정합니다. 평균 쪽 정확도가 더 높았다면 평균 파일을, 아니면 더 나은 단독 파일을 냅니다.
# (예전에는 모델마다 확률의 눈금이 달라서 순위 평균을 썼지만, 그러면 폐렴 판정 비율을 따로 정해야 합니다. v11과 v12는 손실함수가 같아 확률을 그대로 평균냅니다.)

# %%
test_prob = np.mean(test_probs, axis=0)                    # 5개 fold 모델 확률의 평균
np.save("test_prob_v12.npy", test_prob)                    # v12의 test 폐렴 확률 저장

submission = pd.read_csv("sample_submission.csv")          # 제출 양식
submission["label"] = (test_prob > 0.5).astype(int)        # 기준 0.5
submission.to_csv("submission_v12_t05.csv", index=False)   # v12 단독 제출 파일
print(f"submission_v12_t05.csv: 폐렴 비율 {submission['label'].mean():.3f}")

if os.path.exists("test_prob_v11.npy") and os.path.exists("submission_v11_t05.csv"):
    v11_prob = np.load("test_prob_v11.npy")                                  # v11의 test 폐렴 확률
    v11_sub = pd.read_csv("submission_v11_t05.csv")["label"]                 # v11 제출 파일 (0.9471)
    print(f"  v11 파일과 다른 장수 {(submission['label'] != v11_sub).sum()}")
    corr = pd.Series(v11_prob).rank().corr(pd.Series(test_prob).rank())      # test 확률의 순위 상관 (정답은 쓰지 않음)
    print(f"v11-v12 순위 상관 (test): {corr:.3f}")
    submission["label"] = ((v11_prob + test_prob) / 2 > 0.5).astype(int)     # 두 모델 확률 평균, 기준 0.5
    submission.to_csv("submission_v11_v12_t05.csv", index=False)             # 평균 제출 파일
    print(f"submission_v11_v12_t05.csv: 폐렴 비율 {submission['label'].mean():.3f}, v11 파일과 다른 장수 {(submission['label'] != v11_sub).sum()}")
'''

assert 'efficientnet' not in src and 'Grayscale' not in src.split('## 3. Dataset')[1]
open('xray_v12_xrv_nobars.py', 'w').write(src)
print('xray_v12_xrv_nobars.py 생성')
