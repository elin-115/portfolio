# %%
# ============================================================
# 재난현장 중증도 분류 — 데이터 분석
# 1) 데이터 불러오기  2) 전처리  3) 탐색·층별화  4) 모델링
# ============================================================

import pandas as pd                      # 표 형태 데이터를 다루는 도구
import numpy as np                       # 숫자 계산 도구
import matplotlib.pyplot as plt          # 그래프 그리는 도구
from scipy import stats                  # 통계 검정 도구

# 한글이 네모(□)로 깨지지 않게 폰트 설정하기
import platform                                 # 지금 어떤 운영체제인지 확인하는 도구
os_name = platform.system()                       # 'Darwin'(맥) / 'Windows' / 'Linux'(코랩)

if os_name == 'Darwin':                           # 맥이면
    plt.rcParams['font.family'] = 'AppleGothic'
elif os_name == 'Windows':                        # 윈도우면
    plt.rcParams['font.family'] = 'Malgun Gothic'
else:                                           # 코랩(리눅스)이면 나눔고딕을 설치해서 씀
    import os
    if not os.path.exists('/usr/share/fonts/truetype/nanum/NanumGothic.ttf'):
        os.system('apt-get -qq -y install fonts-nanum > /dev/null')   # 폰트 설치
    import matplotlib.font_manager as fm
    fm.fontManager.addfont('/usr/share/fonts/truetype/nanum/NanumGothic.ttf')
    plt.rcParams['font.family'] = 'NanumGothic'

plt.rcParams['axes.unicode_minus'] = False      # 마이너스 기호가 깨지지 않게
print('폰트 설정 완료 :', plt.rcParams['font.family'][0])

# %%
# ============================================================
# 1. 데이터 불러오기
# ============================================================

# 코랩이면 왼쪽 폴더 아이콘을 눌러 CSV를 업로드한 뒤 아래 이름 그대로 쓰면 됩니다.
path = 'SOMS_야전응급처치_타임라인_가상데이터.csv'

df = pd.read_csv(path, encoding='utf-8-sig')       # CSV 파일을 읽어서 표로 만들기

print('행 개수 :', df.shape[0])                     # 데이터가 몇 줄인지
print('열 개수 :', df.shape[1])                     # 변수가 몇 개인지
df.head()                                           # 위에서 5줄만 미리보기

# %%
# ============================================================
# 2. 데이터 구조 확인
# ============================================================

df.info()                                # 각 열의 타입과 결측 개수 확인

# %%
# 열마다 고유값이 몇 개인지 세어보기 (범주형인지 연속형인지 판단용)
for col in df.columns:                     # 모든 열을 하나씩 돌면서
    n_uniq = df[col].nunique()               # 그 열에 서로 다른 값이 몇 개인지
    print(f'{col:<15} 고유값 {n_uniq:>6}개')  # 열 이름과 개수를 나란히 출력

# %%
# ============================================================
# 3. 목표 변수 확인 — 우리가 맞히려는 것은 '분류색상'
# ============================================================

print(df['분류색상'].value_counts())              # 등급별로 몇 건인지
print()
print(df['분류색상'].value_counts(normalize=True).round(3))   # 비율로도 보기

# 환자평가는 분류색상과 같은 정보인지 교차표로 확인
print()
print(pd.crosstab(df['분류색상'], df['환자평가']))

# %%
# ============================================================
# 4. 전처리 ① — 활력징후 0 은 '측정불가'이므로 결측으로 바꾸기
# ============================================================

VITALS = ['호흡수(회/분)', '맥박수(회/분)', '수축기혈압(mmHg)']   # 활력징후 3개 열 이름

# 3개가 전부 0인 행 찾기 (측정 자체가 불가능했던 환자)
all_zero = (df[VITALS] == 0).all(axis=1)          # axis=1 은 '한 줄씩 가로로' 검사한다는 뜻

print('활력징후가 전부 0인 행 :', all_zero.sum(), '건')
print('그 행들의 분류색상     :', df[all_zero]['분류색상'].value_counts().to_dict())

df['측정불가'] = all_zero                        # 나중에 쓸 수 있게 표시를 남겨두고
df.loc[all_zero, VITALS] = np.nan                  # 0 을 결측(NaN)으로 바꾸기 (평균이 왜곡되지 않게)

# %%
# ============================================================
# 5. 전처리 ② — '-' 는 결측이 아니라 '해당 없음'이므로 이름을 붙여주기
# ============================================================

# 열마다 '-' 가 무슨 뜻인지 정리한 사전
na_meaning = {'후송우선순위': '미결정', '후송수단': '해당없음', '부상부위': '부위무관',
        '손등번호': '미부여',   '다음권고조치': '없음',  '음성근거': '발화없음'}

for col, meaning in na_meaning.items():                # 사전을 하나씩 꺼내면서
    n_dash = (df[col] == '-').sum()           # '-' 가 몇 개인지 세고
    df[col] = df[col].replace('-', meaning)       # '-' 를 알아보기 쉬운 말로 바꾸기
    print(f'{col:<10} : {n_dash:>6}건 → "{meaning}"')

# %%
# ============================================================
# 6. 전처리 ③ — 시각 문자열을 진짜 시간으로 바꾸기
# ============================================================
# '03:21:53' 은 글자일 뿐이라 뺄셈이 안 됩니다. 날짜와 합쳐서 시간 타입으로 만듭니다.

def to_seconds(text):                      # '03:21:53' → 12113 (초)
    parts = text.split(':')                  # ':' 기준으로 쪼개면 ['03','21','53']
    parts = [int(v) for v in parts]           # 숫자로 변환
    if len(parts) == 3:                      # 시:분:초 형태면
        return parts[0]*3600 + parts[1]*60 + parts[2]
    return parts[0]*60 + parts[1]             # 분:초 형태면 (영상 구간용)

date_col = pd.to_datetime(df['날짜'])           # 날짜 열을 날짜 타입으로

# 날짜 + 시각 을 합쳐서 완전한 시간 만들기
df['식별_시각']   = date_col + pd.to_timedelta(df['식별시간'].apply(to_seconds),   unit='s')
df['조치시작_시각'] = date_col + pd.to_timedelta(df['조치시작시간'].apply(to_seconds), unit='s')
df['조치완료_시각'] = date_col + pd.to_timedelta(df['조치완료시간'].apply(to_seconds), unit='s')

# 자정을 넘긴 처치는 완료 시각이 시작보다 앞서 보임 → 하루를 더해서 고침
past_midnight = df['조치완료_시각'] < df['조치시작_시각']
print('자정을 넘긴 행 :', past_midnight.sum(), '건')
df.loc[past_midnight, '조치완료_시각'] = df.loc[past_midnight, '조치완료_시각'] + pd.Timedelta(days=1)

# 제대로 바뀌었는지 검산 — 계산한 소요시간이 기록된 값과 같은지
calc = (df['조치완료_시각'] - df['조치시작_시각']).dt.total_seconds()
print('기록값과 다른 행 :', (calc != df['조치소요(초)']).sum(), '건')

# %%
# ============================================================
# 7. 전처리 ④ — 분석에 쓸 새 변수 만들기
# ============================================================

# 쇼크지수 = 맥박 ÷ 수축기혈압. 0.9 를 넘으면 쇼크를 의심하는 임상 지표입니다.
df['쇼크지수'] = df['맥박수(회/분)'] / df['수축기혈압(mmHg)']

# 의식수준을 숫자로 (A가 가장 좋고 U가 가장 나쁨)
avpu_map = {'A': 3, 'V': 2, 'P': 1, 'U': 0}
df['의식점수'] = df['의식수준(AVPU)'].map(avpu_map)

# 환자별로 '처음 발견된 시각'을 구해서, 각 처치까지 몇 분 지났는지 계산
df['환자키'] = df['시나리오ID'] + '_' + df['부상자ID']       # 환자 한 명을 구분하는 열쇠
first_seen = df.groupby('환자키')['식별_시각'].transform('min')   # 환자별 최초 시각
df['경과_분'] = (df['식별_시각'] - first_seen).dt.total_seconds() / 60

print(df[['쇼크지수', '의식점수', '경과_분']].describe().round(2))

# %%
# ============================================================
# 7-2. 전처리 ⑤ — 복합 범주 쪼개서 원-핫으로 만들기
# ============================================================
# '부상기전'에는 "둔상(압박)+압궤성 질식" 처럼 '+'로 이어붙은 값이 있습니다.
# 그대로 두면 조합마다 다른 값이 되어버리므로, 쪼개서 각각을 열로 만듭니다.

split_list = df['부상기전'].str.split('+')            # '+' 기준으로 나누기 → 리스트가 됨

all_mech = []                                     # 등장한 기전을 전부 모을 목록
for items in split_list:                                # 행마다 리스트를 꺼내서
    for mech in items:                              # 그 안의 항목을 하나씩
        all_mech.append(mech.strip())              # 앞뒤 공백 떼고 담기

freq = pd.Series(all_mech).value_counts()          # 기전별로 몇 번 나왔는지
keep_mech = freq[freq >= 30].index                # 30건 이상 나온 것만 쓰기
print('기전 종류', len(freq), '개 중', len(keep_mech), '개 사용')

for mech in keep_mech:                            # 기전마다 새 열을 하나씩 만들기
    df['기전_' + mech] = split_list.apply(lambda items: mech in [x.strip() for x in items])

print(df[['기전_' + g for g in keep_mech[:5]]].sum())   # 앞 5개만 확인

# %%
# ============================================================
# 8. 전처리 ⑥ — 공공(재난·테러)만 남기기
# ============================================================
# 군(전투·훈련사고)과 공공은 처치 단계 체계가 완전히 다르므로 섞지 않습니다.

df['영역'] = df['상황구분'].replace({'전투': '군', '훈련사고': '군',
                                    '재난': '공공', '테러': '공공'})
print(df['영역'].value_counts())

pub = df[df['영역'] == '공공'].copy()        # 공공만 골라서 새 표로
print('\n공공 데이터 :', pub.shape[0], '행')
print('환자 수     :', pub['환자키'].nunique(), '명')
print('시나리오    :', pub['시나리오ID'].nunique(), '개')

# %%
# ============================================================
# 9. 탐색 ① — 분류색상별 활력징후 차이 보기
# ============================================================

print(pub.groupby('분류색상')[VITALS + ['쇼크지수']].median().round(2))   # 등급별 중앙값

# %%
# 상자그림으로 등급별 분포 비교하기
ORDER = ['녹색', '황색', '적색']                        # 가벼운 순서대로 놓기

fig, axes = plt.subplots(1, 4, figsize=(16, 4))        # 그래프 4개를 가로로
for i, col in enumerate(VITALS + ['쇼크지수']):            # 활력징후 3개 + 쇼크지수
    box_data = [pub[pub['분류색상'] == g][col].dropna() for g in ORDER]  # 등급별로 값 모으기
    axes[i].boxplot(box_data, tick_labels=ORDER)         # 상자그림 그리기
    axes[i].set_title(col)                               # 제목은 열 이름
    axes[i].grid(axis='y', alpha=0.3)                   # 가로 눈금선 연하게
plt.tight_layout()                                      # 겹치지 않게 간격 정리
plt.show()

# %%
# ============================================================
# 10. 탐색 ② — 의식수준(AVPU)과 분류색상의 관계
# ============================================================

ctab = pd.crosstab(pub['의식수준(AVPU)'], pub['분류색상'])
print('건수')
print(ctab.reindex(['A', 'V', 'P', 'U']))            # A→U 순서로 정렬

print('\n행 비율(%)')
ratio = pd.crosstab(pub['의식수준(AVPU)'], pub['분류색상'], normalize='index') * 100
print(ratio.reindex(['A', 'V', 'P', 'U']).round(1))

# %%
# ============================================================
# 10-2. 탐색 ③ — 활력징후끼리 얼마나 겹치는가 (다중공선성)
# ============================================================
# 서로 너무 비슷한 변수를 여러 개 넣으면 모델이 헷갈립니다. 상관계수로 확인합니다.

corr = pub[VITALS + ['쇼크지수', '의식점수']].corr().round(2)   # 변수끼리 상관계수 계산
print(corr)

plt.figure(figsize=(6, 5))
plt.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)               # 색으로 표시
plt.colorbar(label='상관계수')
plt.xticks(range(len(corr)), corr.columns, rotation=45, ha='right')
plt.yticks(range(len(corr)), corr.index)
for i in range(len(corr)):                                      # 칸마다 숫자 써넣기
    for j in range(len(corr)):
        plt.text(j, i, corr.iloc[i, j], ha='center', va='center', fontsize=9)
plt.title('활력징후 간 상관계수')
plt.tight_layout()
plt.show()

print('\n맥박수와 수축기혈압이 음(-)의 상관 → 둘을 나눈 쇼크지수 하나로 요약할 수 있습니다')

# %%
# ============================================================
# 11. 층별화 ① — 재난 유형별로 중증도 구성이 다른가
# ============================================================
# 환자 한 명당 한 줄만 남겨서 세야 정확합니다 (처치 이벤트는 환자마다 개수가 달라서)

triage = pub[pub['유형'] == '환자분류'].copy()      # 환자분류 이벤트 = 환자당 1건
print('분류 시점 행 :', len(triage), '건')

by_type = pd.crosstab(triage['상황'], triage['분류색상'], normalize='index') * 100
print()
print(by_type.round(1).sort_values('적색', ascending=False))

# %%
# 카이제곱 검정 — 재난 유형에 따라 중증도 구성이 정말 다른지 확인
table = pd.crosstab(triage['상황'], triage['분류색상'])
chi2, p_val, dof, expected = stats.chi2_contingency(table)
print(f'카이제곱 = {chi2:.1f}, p값 = {p_val:.3g}')
print('→ p값이 0.05보다 작으면 "재난 유형에 따라 중증도 구성이 다르다"고 볼 수 있습니다')

# %%
# ============================================================
# 12. 층별화 ② — 구역(Zone)별로 어떤 처치를 하는가
# ============================================================

pub['구역'] = pub['구분'].replace({'CUF': 'Hot Zone', 'TFC': 'Warm Zone', 'TEC': 'Cold Zone'})

by_zone = pd.crosstab(pub['구역'], pub['유형'], normalize='index') * 100
print(by_zone.round(1)[['대량출혈', '기도', '호흡', '순환', '재평가', '후송']])

# %%
# ============================================================
# 13. 층별화 ③ — 역할별로 어떤 처치를 맡는가
# ============================================================

for role in pub['조치자_역할'].unique():                      # 역할을 하나씩 돌면서
    sub = pub[pub['조치자_역할'] == role]                    # 그 역할의 행만 고르고
    top3 = sub['유형'].value_counts(normalize=True).head(3)    # 많이 한 처치 3개
    desc = ', '.join([f'{k} {v*100:.0f}%' for k, v in top3.items()])
    print(f'{role:<18} {desc}')

# %%
# ============================================================
# 13-2. 층별화 ④ — 군집분석으로 환자군 나누기 (K-means)
# ============================================================
# 정답(분류색상)을 모른다고 가정하고, 활력징후만으로 환자를 3무리로 묶어봅니다.
# 그 결과가 실제 분류색상과 얼마나 맞는지 보면 활력징후의 설명력을 알 수 있습니다.

from sklearn.cluster import KMeans              # 군집 나누는 도구
from sklearn.preprocessing import StandardScaler  # 단위를 맞춰주는 도구

cl_data = triage[VITALS + ['의식점수']].dropna()   # 결측 있는 행은 빼고

# 혈압은 100단위, 호흡수는 20단위라 그대로 두면 혈압만 반영됩니다 → 단위 맞추기
cl_scaled = StandardScaler().fit_transform(cl_data)

kmeans = KMeans(n_clusters=3, random_state=0, n_init=10)   # 3무리로 나누기
cl_id = kmeans.fit_predict(cl_scaled)                     # 각 환자가 몇 번 무리인지

# 무리별 평균 활력징후 보기
cl_df = cl_data.copy()                          # 원본 복사
cl_df['군집'] = cl_id                          # 무리 번호 붙이기
print('무리별 평균')
print(cl_df.groupby('군집').mean().round(1))

# 무리와 실제 분류색상을 비교
print('\n무리 × 실제 분류색상')
print(pd.crosstab(cl_id, triage.loc[cl_data.index, '분류색상']))

# %%
# ============================================================
# 14. 가설 검정 — 등급 간 활력징후 차이가 통계적으로 유의한가
# ============================================================

for col in VITALS + ['쇼크지수']:
    # 먼저 정규분포인지 확인 (정규분포면 ANOVA, 아니면 비모수 검정)
    sample = triage[col].dropna().sample(min(500, len(triage)), random_state=0)
    p_norm = stats.shapiro(sample)[1]

    # 등급별로 값을 모아서 세 집단 비교
    groups = [triage[triage['분류색상'] == g][col].dropna() for g in ORDER]
    p_kw = stats.kruskal(*groups)[1]                # *집단 은 리스트를 하나씩 풀어서 넣는다는 뜻

    norm_txt = '정규 아님' if p_norm < 0.05 else '정규'
    print(f'{col:<15} 정규성 p={p_norm:.3g} ({norm_txt})   등급간 차이 p={p_kw:.3g}')

# %%
# ============================================================
# 15. 모델링 준비 ① — 입력 변수와 목표 변수 정하기
# ============================================================
# 주의: 처치가 끝난 뒤에 생기는 값은 넣으면 안 됩니다(정답을 미리 아는 셈이 되니까요).
#   넣으면 안 되는 것 : 환자평가(분류색상과 같은 정보), 유형, MARCH코드, 조치내용,
#                      조치결과, 사용도구, 다음권고조치, 손등번호,
#                      후송우선순위, 후송수단, 인지→조치(초), 조치소요(초)
#   넣어도 되는 것    : 분류하는 그 순간에 알 수 있는 값

FEATURES = ['의식점수',                 # 의식수준을 숫자로 바꾼 것
          '호흡수(회/분)',
          '맥박수(회/분)',
          '수축기혈압(mmHg)',
          '쇼크지수',                 # 맥박 ÷ 혈압
          '경과_분']                  # 발견 후 지금까지 걸린 시간(미래 정보가 아님)

TARGET = '분류색상'

# 흑색은 15건뿐이라 학습이 어려워 제외하고 3등급으로 진행합니다
data = triage[triage['분류색상'] != '흑색'].copy()
print('학습에 쓸 환자 :', len(data), '명')
print(data[TARGET].value_counts())

# %%
# ============================================================
# 16. 모델링 준비 ② — 학습용과 검증용 나누기
# ============================================================
# 무작위로 나누면 같은 시나리오(대본)가 양쪽에 섞여서 점수가 부풀려집니다.
# 그래서 시나리오 템플릿 단위로 통째로 갈라놓습니다.

templates = data['세부상황'].unique()          # 공공 템플릿 8종
print('템플릿 종류 :', len(templates), '개')
for t in templates:
    print('  -', t[:40])

# 뒤쪽 2종을 검증용으로 떼어놓기
test_templates = templates[-2:]
print('\n검증용으로 뺀 템플릿 :', [t[:25] for t in test_templates])

train_df = data[~data['세부상황'].isin(test_templates)]   # ~ 는 '아닌 것'
test_df = data[data['세부상황'].isin(test_templates)]

X_train = train_df[FEATURES]                          # 학습용 입력
y_train = train_df[TARGET]                          # 학습용 정답
X_test  = test_df[FEATURES]                          # 검증용 입력
y_test  = test_df[TARGET]                          # 검증용 정답

print(f'\n학습 {len(X_train)}명 / 검증 {len(X_test)}명')

# %%
# ============================================================
# 17. 모델 학습 — 여러 모델을 비교하기
# ============================================================

from sklearn.tree import DecisionTreeClassifier          # 의사결정나무
from sklearn.ensemble import RandomForestClassifier      # 랜덤포레스트
from sklearn.linear_model import LogisticRegression      # 로지스틱 회귀
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 결측(측정불가)은 각 열의 중앙값으로 채우기 — 모델이 빈칸을 못 읽기 때문
X_train = X_train.fillna(X_train.median())
X_test  = X_test.fillna(X_train.median())                # 검증도 '학습의' 중앙값으로 채움

models = {
    '의사결정나무':   DecisionTreeClassifier(max_depth=5, random_state=0),
    '랜덤포레스트':   RandomForestClassifier(n_estimators=300, random_state=0),
    '로지스틱 회귀':  LogisticRegression(max_iter=1000),
}

# XGBoost 는 코랩에 기본으로 깔려 있습니다. 없으면 이 부분만 건너뜁니다.
try:
    from xgboost import XGBClassifier                      # 부스팅 계열 모델
    models['XGBoost'] = XGBClassifier(n_estimators=300, max_depth=4,
                                     learning_rate=0.1, random_state=0)
except ImportError:
    print('(xgboost 가 설치돼 있지 않아 건너뜁니다)')

to_num = {'녹색': 0, '황색': 1, '적색': 2}          # XGBoost 는 글자 정답을 못 받아서
to_label = {0: '녹색', 1: '황색', 2: '적색'}          # 숫자로 바꿨다가 다시 글자로

preds = {}                                        # 모델별 예측을 담아둘 곳
for name, model in models.items():                 # 모델을 하나씩 꺼내서
    if name == 'XGBoost':                         # XGBoost 만 숫자 정답으로 학습
        model.fit(X_train, y_train.map(to_num))
        pred = pd.Series(model.predict(X_test)).map(to_label).values
    else:
        model.fit(X_train, y_train)                # 나머지는 글자 그대로
        pred = model.predict(X_test)               # 검증 데이터로 예측
    preds[name] = pred                             # 나중에 쓰려고 저장
    print(f'{name:<12} 정확도 {accuracy_score(y_test, pred)*100:.2f}%')

# %%
# ============================================================
# 18. 평가 — 정확도보다 '어느 방향으로 틀렸는가'가 중요합니다
# ============================================================
# 위중한 환자를 가볍게 본 것(과소분류)은 사망으로 이어지지만,
# 가벼운 환자를 위중하게 본 것(과대분류)은 자원 낭비에 그칩니다.

LEVEL = {'녹색': 0, '황색': 1, '적색': 2}       # 숫자로 바꿔서 크기를 비교하려고

for name, pred in preds.items():
    y_num = y_test.map(LEVEL).values          # 실제 등급을 숫자로
    p_num = pd.Series(pred).map(LEVEL).values  # 예측 등급을 숫자로

    under = (p_num < y_num).mean() * 100       # 실제보다 낮게 본 비율
    over = (p_num > y_num).mean() * 100       # 실제보다 높게 본 비율

    is_red = (y_test == '적색')                   # 진짜 적색인 환자
    red_miss = (is_red & (pd.Series(pred, index=y_test.index) != '적색')).sum()

    print(f'{name:<12} 과소분류 {under:5.1f}%   과대분류 {over:5.1f}%   '
          f'적색 놓침 {red_miss}명 / {is_red.sum()}명')

print('\n※ 미국외과학회 기준 : 과소분류 5% 미만, 과대분류 25~35%')

# %%
# 가장 좋은 모델의 혼동행렬 자세히 보기
best = '랜덤포레스트'
cm = confusion_matrix(y_test, preds[best], labels=ORDER)
print(pd.DataFrame(cm, index=['실제_'+g for g in ORDER], columns=['예측_'+g for g in ORDER]))
print()
print(classification_report(y_test, preds[best], digits=3))

# %%
# ============================================================
# 19. 어떤 변수가 중요했는가
# ============================================================

rf = models['랜덤포레스트']                                    # 학습된 랜덤포레스트 꺼내기
importance = pd.Series(rf.feature_importances_, index=FEATURES)      # 변수별 중요도
importance = importance.sort_values(ascending=False)                   # 큰 순서로 정렬

print(importance.round(3))

plt.figure(figsize=(7, 4))
plt.barh(importance.index[::-1], importance.values[::-1], color='#0E8E7E')   # [::-1]은 순서 뒤집기
plt.xlabel('중요도')
plt.title('중증도 분류에 기여한 변수')
plt.tight_layout()
plt.show()

# %%
# ============================================================
# 19-2. 변수 선별 — 상호정보량과 순열 중요도 비교
# ============================================================
# 상호정보량은 "정답과 얼마나 관련 있나"만 봅니다.
# 순열 중요도는 "그 변수를 망가뜨리면 성능이 얼마나 떨어지나"를 봅니다.
# 둘이 다르면, 관련은 있어 보여도 실제로는 도움이 안 되는 변수입니다.

from sklearn.feature_selection import mutual_info_classif    # 상호정보량
from sklearn.inspection import permutation_importance        # 순열 중요도

# 상호정보량 계산 (학습 데이터 기준)
mi = mutual_info_classif(X_train, y_train, random_state=0)
mi = pd.Series(mi, index=FEATURES)

# 순열 중요도 계산 (검증 데이터에서 변수를 섞어보며 성능 하락폭 측정)
perm = permutation_importance(rf, X_test, y_test, n_repeats=20, random_state=0)
perm = pd.Series(perm.importances_mean, index=FEATURES)

compare = pd.DataFrame({'상호정보량': mi.round(3),
                    '순열 중요도': perm.round(3),
                    '모델 중요도': importance.round(3)})
print(compare.sort_values('순열 중요도', ascending=False))
print('\n순열 중요도가 0 이하면 그 변수는 빼는 편이 낫습니다')

# %%
# ============================================================
# 20. START 규칙 — 표준 분류법을 그대로 코드로 옮기기
# ============================================================
# 다수사상자 현장에서 쓰는 국제 표준 분류법입니다.
# 모델과 달리 '왜 그렇게 판단했는지' 설명할 수 있습니다.

def start_triage(avpu, rr, pr, sbp):
    if pd.isna(rr) or rr == 0:           # 숨을 안 쉬면
        return '흑색'
    if rr > 30 or rr < 10:               # 호흡수가 비정상이면
        return '적색'
    if sbp < 90:                            # 혈압이 낮으면 (관류 불량)
        return '적색'
    if pr > 120:                           # 맥박이 너무 빠르면
        return '적색'
    if avpu in ['P', 'U']:                   # 명령을 따르지 못하면
        return '적색'
    # 여기까지 걸리지 않았고 모든 수치가 정상 범위면 녹색
    if avpu == 'A' and 10 <= rr <= 26 and pr <= 110 and sbp >= 95:
        return '녹색'
    return '황색'                            # 나머지는 황색

# 검증 데이터에 규칙 적용하기
rule_pred = []
for i in test_df.index:                          # 검증 데이터를 한 줄씩 돌면서
    res = start_triage(test_df.loc[i, '의식수준(AVPU)'],
                   test_df.loc[i, '호흡수(회/분)'],
                   test_df.loc[i, '맥박수(회/분)'],
                   test_df.loc[i, '수축기혈압(mmHg)'])
    rule_pred.append(res)                       # 결과를 목록에 담기
rule_pred = pd.Series(rule_pred, index=test_df.index)

print('START 규칙 정확도 :', f'{accuracy_score(y_test, rule_pred)*100:.2f}%')

# %%
# ============================================================
# 21. 규칙 + 모델 결합 — 둘 중 더 위중한 쪽을 고르기
# ============================================================
# 규칙과 모델은 틀리는 방식이 다릅니다. 안전한 쪽(더 위중한 판정)을 택하면
# 두 방법의 약점을 서로 덮어줍니다.

model_pred = pd.Series(preds['랜덤포레스트'], index=test_df.index)

combo_pred = []
for i in test_df.index:                                      # 환자를 한 명씩 보면서
    rule_lv = LEVEL.get(rule_pred[i], 3)                 # 규칙이 매긴 등급의 숫자
    model_lv = LEVEL.get(model_pred[i], 3)                 # 모델이 매긴 등급의 숫자
    higher = max(rule_lv, model_lv)                        # 더 위중한 쪽 고르기
    back = {0: '녹색', 1: '황색', 2: '적색', 3: '흑색'}    # 숫자를 다시 등급으로
    combo_pred.append(back[higher])
combo_pred = pd.Series(combo_pred, index=test_df.index)

# 세 방법을 한 표로 비교하기
summary = []
for name, pred in [('START 규칙', rule_pred), ('랜덤포레스트', model_pred), ('규칙+모델 결합', combo_pred)]:
    y_num = y_test.map(LEVEL).values
    p_num = pred.map(LEVEL).values
    is_red = (y_test == '적색')
    summary.append({
        '방법': name,
        '정확도': round(accuracy_score(y_test, pred)*100, 1),
        '과소분류%': round((p_num < y_num).mean()*100, 1),
        '과대분류%': round((p_num > y_num).mean()*100, 1),
        '적색놓침': int((is_red & (pred != '적색')).sum()),
    })
print(pd.DataFrame(summary).to_string(index=False))

# %%
# ============================================================
# 22. 정리
# ============================================================
print('[분석 요약]')
print(f'· 공공 데이터 {len(pub):,}건, 환자 {pub["환자키"].nunique()}명, 템플릿 8종')
print(f'· 학습 {len(X_train)}명 / 검증 {len(X_test)}명 (시나리오 템플릿 단위로 분리)')
print(f'· 목표 변수 : 분류색상 (적색·황색·녹색)')
print(f'· 입력 변수 : {", ".join(FEATURES)}')
print()
print('[주의] 이 데이터는 실제 기록 35행을 바탕으로 만든 가상 데이터입니다.')
print('       위 성능은 "가상 데이터 기준 내부 검증치"이며 실제 현장 성능이 아닙니다.')

# %%
# ============================================================
# 23. (심화) 템플릿 8종을 모두 돌아가며 검증하기 — GroupKFold
# ============================================================
# 위에서는 템플릿 2종만 검증에 썼습니다(92명).
# 8종을 번갈아 검증에 넣으면 전체 835명을 모두 한 번씩 검증할 수 있습니다.

from sklearn.model_selection import GroupKFold, cross_val_predict

X_all = data[FEATURES].fillna(data[FEATURES].median())   # 결측 채우기
y_all = data[TARGET]                                 # 정답
groups_all   = data['세부상황']                             # 같은 템플릿끼리 묶어두는 기준

rf2 = RandomForestClassifier(n_estimators=300, random_state=0)

# cv=GroupKFold(4) : 템플릿을 4묶음으로 갈라 번갈아 검증
pred_all = cross_val_predict(rf2, X_all, y_all, cv=GroupKFold(4), groups=groups_all)

print('전체 검증 인원 :', len(y_all), '명')
print('정확도         :', f'{accuracy_score(y_all, pred_all)*100:.2f}%')

y_num = y_all.map(LEVEL).values
p_num = pd.Series(pred_all).map(LEVEL).values
print('과소분류       :', f'{(p_num < y_num).mean()*100:.2f}%')
print('과대분류       :', f'{(p_num > y_num).mean()*100:.2f}%')

cm2 = confusion_matrix(y_all, pred_all, labels=ORDER)
print()
print(pd.DataFrame(cm2, index=['실제_'+g for g in ORDER], columns=['예측_'+g for g in ORDER]))
