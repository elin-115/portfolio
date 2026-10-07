# xray_v10_nobars.py → xray_v11_epochs.py (에포크만 6 → 10)
s=open('xray_v10_nobars.py').read()
a=s.index("# %% [markdown]\n# ## 0.")
header='''# %% [markdown]
# # 흉부 X-ray 폐렴 분류 — v11 (v10 + 에포크 늘리기)
#
# v10(검은 띠 제거 + 비율 흔들기)은 **모델 하나, 기준 0.5, 보정 없이** 리더보드 0.9183이 나왔습니다 (같은 조건의 v1은 0.8526).
#
# 그런데 v10의 검증(OOF) 결과를 보면 오답 방향이 바뀌었습니다.
#
# | | 정상 → 폐렴 (오답) | 폐렴 → 정상 (오답) | 검증 정확도 |
# |---|---|---|---|
# | v10 | 20장 | **226장** | 0.953 |
#
# 검은 띠 단서가 사라지고 증강이 강해지면서(무작위로 자르고 늘림), **6 에포크로는 폐렴을 충분히 배우지 못한 것**으로 보입니다.
#
# ## v11에서 바꾼 점 (하나만)
# - **에포크 6 → 10.** 전처리·모델·나머지 설정은 v10과 완전히 같습니다.
# - fold마다 **몇 번째 에포크가 가장 좋았는지** 출력합니다. 마지막 에포크가 최고라면 아직 더 배울 여지가 있다는 뜻입니다.
#
# 끝난 fold는 `v11_fold번호.npz`로 저장해 이어서 실행할 수 있습니다. test는 예측에만 사용합니다.
# 예상 시간 약 2시간 45분. 결과: `test_prob_v11.npy`, `submission_v11_t05.csv`

'''
s=header+s[a:]
R=[
("EPOCHS = 6                                                 # fold마다 6번 반복 학습","EPOCHS = 10                                                # fold마다 10번 반복 학습 (v10은 6번)"),
("v10_fold","v11_fold"),
("    best_acc = 0\n","    best_acc = 0; best_ep = 0\n"),
("            best_acc = val_acc\n","            best_acc = val_acc; best_ep = ep + 1                 # 최고 정확도와 그때의 에포크 번호\n"),
('    print(f"  fold {k+1} 최고 검증 정확도: {best_acc:.4f}")','    print(f"  fold {k+1} 최고 검증 정확도: {best_acc:.4f} (에포크 {best_ep})")'),
("# 3. 6 에포크 학습","# 3. 10 에포크 학습"),
]
for x,y in R:
    assert x in s, x[:40]; s=s.replace(x,y)
b=s.index("# %% [markdown]\n# ## 7.")
s=s[:b]+'''# %% [markdown]
# ## 7. v10과 비교 (test 없이 확인)
# 저장된 v10의 검증 확률(`v10_fold1~5.npz`)과 같은 사진으로 비교합니다.
# - **가로로 긴 정상** (비율 > 1.35): 검은 띠 단서대로라면 틀리기 쉬운 사진
# - **정사각에 가까운 폐렴** (비율 < 1.2): 마찬가지로 틀리기 쉬운 사진

# %%
def content_ratio(fname):
    a = np.asarray(Image.open(os.path.join("train", fname)).convert("L"))
    bright = a > 8
    rows = np.where(bright.mean(axis=1) > 0.05)[0]; cols = np.where(bright.mean(axis=0) > 0.05)[0]
    return (cols[-1] - cols[0] + 1) / (rows[-1] - rows[0] + 1)           # 내용의 가로 ÷ 세로

train_df["ratio"] = [content_ratio(f) for f in train_df["file_name"]]    # 사진별 가로세로 비율
wide_normal = ((train_df["label"] == 0) & (train_df["ratio"] > 1.35)).values   # 가로로 긴 정상
square_pneu = ((train_df["label"] == 1) & (train_df["ratio"] < 1.2)).values    # 정사각에 가까운 폐렴

def report(name, prob):
    pred = (prob > 0.5).astype(int)                                       # 0.5 기준 예측
    cm = confusion_matrix(train_df["label"], pred)
    print(f"{name}: 전체 {accuracy_score(train_df['label'], pred):.4f} | 정상→폐렴 {cm[0, 1]}장, 폐렴→정상 {cm[1, 0]}장 | "
          f"가로로 긴 정상 {(pred[wide_normal] == 0).mean():.3f} | 정사각 폐렴 {(pred[square_pneu] == 1).mean():.3f}")

if all(os.path.exists(f"v10_fold{k+1}.npz") for k in range(5)):
    v10_oof = np.zeros(len(train_df))
    for k, (tr_idx, val_idx) in enumerate(folds):
        v10_oof[val_idx] = np.load(f"v10_fold{k+1}.npz")["val_prob"]     # v10의 검증 확률 불러오기
    report("v10 (6 에포크) ", v10_oof)
report("v11 (10 에포크)", oof_prob)

# %% [markdown]
# ## 8. 제출 파일
# v10부터는 기준 0.5에서 폐렴 판정 비율이 자연스럽게 나오므로(v10: 63.1%), **비율을 맞추는 보정 없이 기준 0.5**를 그대로 씁니다.

# %%
test_prob = np.mean(test_probs, axis=0)                    # 5개 fold 모델 확률의 평균
np.save("test_prob_v11.npy", test_prob)

submission = pd.read_csv("sample_submission.csv")
submission["label"] = (test_prob > 0.5).astype(int)        # 기준 0.5
submission.to_csv("submission_v11_t05.csv", index=False)
v10_sub = pd.read_csv("submission_v10_t05.csv")["label"]   # v10 제출 파일 (0.9183)
print(f"submission_v11_t05.csv: 폐렴 비율 {submission['label'].mean():.3f}, v10 파일과 다른 장수 {(submission['label'] != v10_sub).sum()}")
'''
open('xray_v11_epochs.py','w').write(s)
