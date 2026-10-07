# -*- coding: utf-8 -*-
"""재난현장 중증도 분류·처치 판단 전체 파이프라인.

  1단 [규칙] START 분류      — 설명 가능한 안전 하한
  2단 [모델] 경계 보정        — 규칙이 못 가르는 녹색/황색 구간
  3단 [결합] 안전우선 채택    — 두 판정 중 더 위중한 쪽
  4단 [규칙] MARCH 처치 순서  — 표준 준수 판정

실행: python soms/pipeline.py   (결과는 soms/metrics.json 에 저장)
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd, numpy as np
from load import load
from preprocess import run as preprocess_run
from start_engine import start_triage, ORDER
from march_engine import PRIORITY, next_action, needs, violates_order
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                             f1_score, mean_absolute_error, confusion_matrix)

MIL = {'전투', '훈련사고'}
VITALS = ['의식수준(AVPU)', '호흡수(회/분)', '맥박수(회/분)', '수축기혈압(mmHg)']
DERIVED = ['쇼크지수', 'AVPU_점수']          # preprocess.py 파생 변수
INV = {v: k for k, v in ORDER.items()}
def _enc(est, X=None):
    """범주형만 정수 인코딩하고 수치형은 원값을 유지한다.
    (문자열로 통째 인코딩하면 '100' < '11' < '2' 가 되어 활력징후의 크기 관계가 깨진다)"""
    if X is None:
        return make_pipeline(OrdinalEncoder(handle_unknown='use_encoded_value',
                                            unknown_value=-1), est)
    cat = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    pre = ColumnTransformer(
        [('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), cat)],
        remainder='passthrough')
    return make_pipeline(pre, est)
_rf  = lambda: RandomForestClassifier(n_estimators=400, random_state=0, n_jobs=-1)

def _cv(X, y, groups, est=None, n=4):
    """시나리오 템플릿 단위 교차검증 — 같은 대본이 학습·검증에 함께 들어가지 않게 한다."""
    X = X.copy()
    for c in X.columns:
        if X[c].dtype == bool:
            X[c] = X[c].astype(int)
    return cross_val_predict(_enc(est or _rf(), X), X, y,
                             cv=GroupKFold(n_splits=n), groups=groups)

def _triage_rates(truth, pred, n):
    off  = pd.Series(pred).map(ORDER).values - truth.map(ORDER).values
    crit = (truth == '적색').values
    miss = (crit & (np.array(pred) != '적색')).sum()
    return dict(과소분류=int((off < 0).sum()), 과대분류=int((off > 0).sum()),
                적색놓침=int(miss), 적색수=int(crit.sum()),
                정확도=round(float((np.array(pred) == truth.values).mean()) * 100, 1),
                과소분류율=round(float((off < 0).mean()) * 100, 1),
                과대분류율=round(float((off > 0).mean()) * 100, 1),
                적색놓침률=round(float(miss / crit.sum()) * 100, 1), 대상=n)

def run(public_only=True, verbose=True):
    df = preprocess_run(save=False)
    if public_only:
        df = df[~df['상황구분'].isin(MIL)].copy()
    M = {'데이터': dict(이벤트=len(df), 환자=int(df['환자키'].nunique()),
                      시나리오=int(df['시나리오ID'].nunique()),
                      템플릿=int(df['세부상황'].nunique()))}

    # ── 1~3단: 중증도 분류 (현장 분류 결정 시점 = 환자당 1건) ──
    tri = df[df['유형'] == '환자분류'].copy()
    tri['START'] = tri.apply(lambda r: start_triage(r['의식수준(AVPU)'], r['호흡수(회/분)'],
                                                    r['맥박수(회/분)'], r['수축기혈압(mmHg)']), axis=1)
    s = tri[tri['분류색상'] != '흑색'].copy()          # 흑색은 표본이 극소 → 3분류로 평가
    s['ML']  = _cv(s[VITALS + DERIVED], s['분류색상'], s['세부상황'])
    s['결합'] = [INV[max(ORDER[a], ORDER[b])] for a, b in zip(s['START'], s['ML'])]
    M['중증도분류'] = {k: _triage_rates(s['분류색상'], s[k], len(s))
                   for k in ('START', 'ML', '결합')}
    M['혼동행렬'] = {k: confusion_matrix(s['분류색상'], s[k],
                                     labels=['녹색', '황색', '적색']).tolist()
                  for k in ('START', 'ML', '결합')}
    M['중증도분류']['베이스라인'] = round(float(s['분류색상'].value_counts(normalize=True).iloc[0]) * 100, 1)

    # ── 이송 우선순위 ──
    d = df[df['후송우선순위'] != '미결정']
    F = VITALS + DERIVED + ['부상기전', '부상부위', '상황']
    p  = _cv(d[F], d['후송우선순위'], d['세부상황'])
    p0 = _cv(d[F], d['후송우선순위'], d['세부상황'], DummyClassifier(strategy='most_frequent'))
    M['이송우선순위'] = dict(정확도=round(accuracy_score(d['후송우선순위'], p)*100, 1),
                       베이스라인=round(accuracy_score(d['후송우선순위'], p0)*100, 1),
                       macroF1=round(f1_score(d['후송우선순위'], p, average='macro')*100, 1),
                       대상=len(d))

    # ── 4단: MARCH 규칙 엔진 검증 ──
    cuf = df[df['구분'] == 'CUF']
    hit = tot = bad = chk = 0
    for _, g in df.sort_values('식별시간').groupby('환자키'):
        seq = [c for c in g['MARCH코드'] if c in PRIORITY]
        if len(seq) >= 2:
            chk += 1
            seen, firsts = set(), []
            for c in seq:
                if c not in seen:
                    seen.add(c); firsts.append(c)
            bad += bool(violates_order(firsts))
        gg = g[g['MARCH코드'].isin(PRIORITY)]
        if gg.empty: continue
        req, done = set(gg['MARCH코드']), set()
        for _, r in gg.iterrows():
            tot += 1; hit += (next_action(done, r['구분'], req) == r['MARCH코드'])
            done.add(r['MARCH코드'])
    M['규칙엔진'] = dict(HotZone준수=round(float((cuf['MARCH코드']=='M').mean())*100, 1),
                     HotZone대상=len(cuf),
                     순서준수=round((1 - bad/chk)*100, 1), 순서대상=chk, 순서이탈=bad,
                     엔진재현율=round(hit/tot*100, 1), 재현대상=tot)

    # ── 필요 처치단계 다중라벨 (모델 vs 수기 규칙) ──
    pat = df.groupby('환자키').first().reset_index()
    pat['세부상황'] = df.groupby('환자키')['세부상황'].first().values
    FX = ['부상기전','부상부위','의식수준(AVPU)','호흡수(회/분)','맥박수(회/분)',
          '수축기혈압(mmHg)','상황','위협수준','조명환경']
    manual = pat.apply(lambda r: needs(r['부상기전'], r['부상부위'], r['의식수준(AVPU)'],
                       r['호흡수(회/분)'], r['맥박수(회/분)'], r['수축기혈압(mmHg)']), axis=1)
    per, hm, hr, tt = {}, 0, 0, 0
    for c in PRIORITY:
        y = df.groupby('환자키')['MARCH코드'].apply(lambda v, c=c: c in set(v)).values.astype(int)
        if y.sum() < 20 or y.mean() > 0.99: continue
        pr = _cv(pat[FX], y, pat['세부상황'])
        mr = manual.apply(lambda v, c=c: c in v).astype(int)
        per[c] = dict(양성률=round(float(y.mean())*100,1),
                      모델재현율=round(recall_score(y,pr)*100,1),
                      수기재현율=round(recall_score(y,mr)*100,1))
        hm += recall_score(y,pr)*y.sum(); hr += recall_score(y,mr)*y.sum(); tt += y.sum()
    M['필요처치단계'] = dict(단계별=per, 모델가중재현율=round(hm/tt*100,1),
                       수기가중재현율=round(hr/tt*100,1))

    # ── 음성근거 NLP ──
    t = df[df['음성근거'] != '-']
    vec = make_pipeline(TfidfVectorizer(analyzer='char_wb', ngram_range=(2,4), min_df=2),
                        LogisticRegression(max_iter=1000))
    pv = cross_val_predict(vec, t['음성근거'], t['유형'],
                           cv=GroupKFold(4), groups=t['음성근거'])   # 같은 발화는 한쪽에만
    M['음성NLP'] = dict(정확도=round(accuracy_score(t['유형'],pv)*100,1),
                      macroF1=round(f1_score(t['유형'],pv,average='macro')*100,1),
                      베이스라인=round(float(t['유형'].value_counts(normalize=True).iloc[0])*100,1),
                      대상=len(t), 고유발화=int(t['음성근거'].nunique()))

    # ── 조치 소요시간 회귀 ──
    F2 = ['유형','MARCH코드','시나리오단계','구분','상황','위협수준','조명환경','조치자_역할']
    yr = df['조치소요(초)']
    pr  = cross_val_predict(_enc(RandomForestRegressor(n_estimators=300, random_state=0, n_jobs=-1)),
                            df[F2].astype(str), yr, cv=GroupKFold(4), groups=df['세부상황'])
    p0r = cross_val_predict(_enc(DummyRegressor()), df[F2].astype(str), yr,
                            cv=GroupKFold(4), groups=df['세부상황'])
    M['소요시간회귀'] = dict(MAE=round(mean_absolute_error(yr,pr),1),
                       베이스라인MAE=round(mean_absolute_error(yr,p0r),1))

    if verbose: report(M)
    return M

def report(M):
    d = M['데이터']
    print(f"\n{'='*66}\n재난현장 중증도 분류·처치 판단 파이프라인")
    print(f"{'='*66}")
    print(f"데이터  이벤트 {d['이벤트']:,} / 환자 {d['환자']:,} / 시나리오 {d['시나리오']} / 템플릿 {d['템플릿']}종")
    t = M['중증도분류']
    print(f"\n[1~3단] 현장 중증도 분류 — 분류 결정 시점 {t['START']['대상']}명 (적색 {t['START']['적색수']}명)")
    print(f"  {'방식':<8}{'과소분류':>14}{'과대분류':>14}{'적색 놓침':>14}{'정확도':>9}")
    for k in ('START','ML','결합'):
        v=t[k]
        print(f"  {k:<8}{v['과소분류']:>6}명({v['과소분류율']:>5.1f}%){v['과대분류']:>6}명({v['과대분류율']:>5.1f}%)"
              f"{v['적색놓침']:>6}명({v['적색놓침률']:>5.1f}%){v['정확도']:>8.1f}%")
    print(f"  ※ ACS-COT 기준: 과소분류 5% 미만 / 과대분류 25~35%   (최빈값 베이스라인 {t['베이스라인']}%)")
    r = M['규칙엔진']
    print(f"\n[4단] MARCH 규칙 엔진")
    print(f"  Hot Zone 준수 {r['HotZone준수']}% ({r['HotZone대상']:,}건)   "
          f"순서 준수 {r['순서준수']}% (환자 {r['순서대상']:,}명, 이탈 {r['순서이탈']}명)   "
          f"엔진 재현율 {r['엔진재현율']}%")
    n = M['필요처치단계']
    print(f"\n[보조] 필요 처치단계 추정  모델 {n['모델가중재현율']}%  vs  수기 규칙 {n['수기가중재현율']}%")
    e = M['이송우선순위']
    print(f"[보조] 이송 우선순위      {e['정확도']}%  (베이스라인 {e['베이스라인']}%)")
    v = M['음성NLP']
    print(f"[보조] 음성근거 NLP       {v['정확도']}%  (베이스라인 {v['베이스라인']}%, 고유 발화 {v['고유발화']}종)")
    g = M['소요시간회귀']
    print(f"[보조] 조치 소요시간 MAE  {g['MAE']}초  (베이스라인 {g['베이스라인MAE']}초)")
    print(f"\n※ 모든 수치는 가상 데이터(실기록 35행 시드) 기준 내부 검증치.\n{'='*66}\n")

if __name__ == '__main__':
    m = run(public_only=True)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'metrics.json')
    json.dump(m, open(out,'w',encoding='utf-8'), ensure_ascii=False, indent=2)
    print(f"저장: {out}")
