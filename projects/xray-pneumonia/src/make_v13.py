# xray_v11_epochs.py 를 바탕으로 xray_v13_resnet50.py 를 만든다
# (v11과 같은 전처리·fold, 모델만 ResNet50으로 교체 + v11·v12와 평균)
src = open('xray_v11_epochs.py').read()

def sub(old, new, count=1):
    global src
    assert src.count(old) >= 1, old[:60]      # 바꿀 문장이 없으면 바로 멈춤
    src = src.replace(old, new) if count == 0 else src.replace(old, new, count)

head_end = src.index('# %% [markdown]\n# ## 0. 데이터 준비')
src = '''# %% [markdown]
# # 흉부 X-ray 폐렴 분류 — v13 (ResNet50 × 검은 띠 제거)
#
# 현재 1위(0.9567)의 기록을 참고했습니다. 1위도 같은 원인(검은 테두리·사진 모양)을 찾았고, 최고점은 **테두리를 제거한, 구조가 다른 모델끼리의 앙상블**에서 나왔습니다.
#
# | | 1위 | 우리 |
# |---|---|---|
# | 테두리 제거 + 비율 흔들기 단일 모델 | ResNet50 0.9519 / DenseNet121 0.9455 | EfficientNet-B0 (v11) 0.9471 |
# | 위 모델들의 앙상블 | **0.9567** | v12·v13을 만들어 시도 |
#
# 1위의 단일 최고 모델이 ResNet50이어서, 같은 전처리로 ResNet50을 추가합니다.
#
# ## v13에서 바꾼 점 (v11 기준, 하나만)
# - **모델만 교체**: EfficientNet-B0 → ResNet50 (ImageNet 사전학습).
# - 에포크 10 → **8** (ResNet50이 훨씬 무거워서. v11은 최고 에포크가 6~9였음).
# - 전처리(검은 띠 제거·비율 흔들기), fold 나누기, 클래스 가중치, 손실함수, 학습률은 v11과 **완전히 같습니다.**
# - 1위 기록에서 효과가 없었던 TTA 추가, train 100% 재학습은 하지 않습니다.
#
# ## 제출 파일 (모두 기준 0.5, 비율 보정 없음)
# - `submission_v13_t05.csv` — v13 단독
# - `submission_v11_v13_t05.csv` — v11 + v13 확률 평균
# - `submission_v11_v12_v13_t05.csv` — v12 결과가 있으면 세 모델 평균
#
# 어느 것을 낼지는 7번 셀에서 **test를 쓰지 않고** train 검증(OOF) 결과로 먼저 확인합니다.
#
# 끝난 fold는 `v13_fold번호.npz`로 저장해 이어서 실행할 수 있습니다. test는 예측에만 사용합니다.
# ⚠️ ResNet50은 EfficientNet-B0보다 계산량이 약 10배라 맥(mps)에서 **6~8시간** 걸릴 수 있습니다. fold 하나(1시간 남짓)씩 나눠 돌려도 됩니다.

''' + src[head_end:]

sub('''# fold마다 아래를 반복합니다. 모델·학습 설정은 v4와 같고 전처리만 다릅니다.
# 1. 새 EfficientNet-B0 불러오기 (fold마다 처음부터)
# 2. 클래스 가중치 계산 (그 fold의 학습 데이터로만)
# 3. 10 에포크 학습, 검증 정확도가 가장 높은 에포크의 가중치 저장''',
'''# fold마다 아래를 반복합니다. v11과 같고 1번(모델)만 ResNet50으로 바뀌었습니다.
# 0. 이미 끝난 fold면(`v13_fold번호.npz` 파일이 있으면) 저장된 결과를 불러오고 건너뛰기
# 1. 새 ResNet50 불러오기 (fold마다 처음부터)
# 2. 클래스 가중치 계산 (그 fold의 학습 데이터로만)
# 3. 8 에포크 학습, 검증 정확도가 가장 높은 에포크의 가중치 저장''')
sub('EPOCHS = 10                                                # fold마다 10번 반복 학습 (v10은 6번)',
    'EPOCHS = 8                                                 # fold마다 8번 반복 학습 (v11은 10번)')
sub('''    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)   # ImageNet 사전학습 가중치
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)                   # 마지막 층 → 2개 클래스
''', '''    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)   # ImageNet 사전학습 ResNet50 (처음 한 번 약 100MB 다운로드)
    model.fc = nn.Linear(model.fc.in_features, 2)                            # ResNet의 마지막 층 이름은 fc (2048 → 2개 클래스로 교체)
''')
sub('v11_fold', 'v13_fold', 0)

tail_start = src.index('# %% [markdown]\n# ## 7. v10과 비교')
src = src[:tail_start] + '''# %% [markdown]
# ## 7. v11·v12와 비교, 그리고 섞었을 때 (test 없이 확인)
# 저장된 v11(`v11_fold1~5.npz`)과 v12(`v12_fold1~5.npz`, 있으면)의 검증 확률을 불러와 **같은 사진**으로 비교합니다.
# - **가로로 긴 정상** (비율 > 1.35), **정사각에 가까운 폐렴** (비율 < 1.2): 검은 띠 단서대로라면 틀리기 쉬운 사진
#
# `평균` 줄의 정확도가 단독 모델보다 높으면 섞을 가치가 있다는 뜻입니다.

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

oofs = {"v13": oof_prob}                                                   # 모델 이름 → 검증 확률
for tag in ["v11", "v12"]:
    if all(os.path.exists(f"{tag}_fold{k+1}.npz") for k in range(5)):     # 결과 파일 5개가 모두 있는 모델만
        o = np.zeros(len(train_df))
        for k, (tr_idx, val_idx) in enumerate(folds):
            o[val_idx] = np.load(f"{tag}_fold{k+1}.npz")["val_prob"]      # 저장된 검증 확률 불러오기
        oofs[tag] = o

for tag in sorted(oofs):
    report(f"{tag} 단독        ", oofs[tag])
if "v11" in oofs:
    report("v11+v13 평균    ", (oofs["v11"] + oofs["v13"]) / 2)
if "v11" in oofs and "v12" in oofs:
    report("v11+v12+v13 평균", (oofs["v11"] + oofs["v12"] + oofs["v13"]) / 3)
print("검증 확률의 순위 상관 (낮을수록 서로 다른 것을 봄)")
print(pd.DataFrame(oofs).rank().corr().round(3))

# %% [markdown]
# ## 8. 제출 파일
# **비율을 맞추는 보정 없이 기준 0.5**를 그대로 씁니다. 무엇을 낼지는 7번 셀의 검증 결과로 정합니다.

# %%
test_prob = np.mean(test_probs, axis=0)                    # 5개 fold 모델 확률의 평균
np.save("test_prob_v13.npy", test_prob)                    # v13의 test 폐렴 확률 저장

submission = pd.read_csv("sample_submission.csv")          # 제출 양식
v11_sub = pd.read_csv("submission_v11_t05.csv")["label"] if os.path.exists("submission_v11_t05.csv") else None   # v11 제출 파일 (0.9471)

def save(prob, fname):
    submission["label"] = (prob > 0.5).astype(int)         # 기준 0.5
    submission.to_csv(fname, index=False)
    diff = "" if v11_sub is None else f", v11 파일과 다른 장수 {(submission['label'] != v11_sub).sum()}"
    print(f"{fname}: 폐렴 비율 {submission['label'].mean():.3f}{diff}")

save(test_prob, "submission_v13_t05.csv")                                  # v13 단독
if os.path.exists("test_prob_v11.npy"):
    p11 = np.load("test_prob_v11.npy")                                     # v11의 test 폐렴 확률
    save((p11 + test_prob) / 2, "submission_v11_v13_t05.csv")              # v11 + v13 평균
    if os.path.exists("test_prob_v12.npy"):
        p12 = np.load("test_prob_v12.npy")                                 # v12의 test 폐렴 확률
        save((p11 + p12 + test_prob) / 3, "submission_v11_v12_v13_t05.csv")   # 세 모델 평균
'''
assert 'efficientnet' not in src
open('xray_v13_resnet50.py', 'w').write(src)
print('xray_v13_resnet50.py 생성')
