# -*- coding: utf-8 -*-
"""MARCH 규칙 엔진을 SOMS 타임라인 데이터로 검증한다.

실행: python3 soms/validate.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from load import load
from march_engine import PRIORITY, next_action, needs, violates_order
import pandas as pd

df = load().sort_values(['환자키', '식별시간'])
out = []

# ① CUF 단계 규칙 — TCCC상 교전 중에는 대량출혈 지혈만
cuf = df[df['구분'] == 'CUF']
m = (cuf['MARCH코드'] == 'M').sum()
out.append(('CUF 규칙 준수', f"{m/len(cuf)*100:.1f}%",
            f"{len(cuf):,}건 중 M {m:,}건, 예외 {len(cuf)-m}건(제독=화학 Hot Zone)"))

# ② MARCH 우선순위 순서 준수 (초발 기준)
first_bad = repeat_bad = checked = 0
pairs = []
for _, g in df.groupby('환자키'):
    seq = [c for c in g['MARCH코드'] if c in PRIORITY]
    if len(seq) < 2:
        continue
    checked += 1
    seen, firsts = set(), []
    for c in seq:
        if c not in seen:
            seen.add(c); firsts.append(c)
    bf = violates_order(firsts)
    if bf:
        first_bad += 1
        pairs += [f'{a}→{b}' for _, a, _, b in bf]
    elif violates_order(seq):
        repeat_bad += 1
out.append(('MARCH 순서 준수(초발)', f"{100-first_bad/checked*100:.1f}%",
            f"환자 {checked:,}명 중 역행 {first_bad}명 / 재처치만 역행 {repeat_bad}명"))

# ③ 엔진이 기록된 다음 처치를 재현하는가
hit = tot = 0
for _, g in df.groupby('환자키'):
    g = g[g['MARCH코드'].isin(PRIORITY)]
    if g.empty:
        continue
    req, done = set(g['MARCH코드']), set()
    for _, r in g.iterrows():
        tot += 1
        hit += (next_action(done, r['구분'], req) == r['MARCH코드'])
        done.add(r['MARCH코드'])
out.append(('엔진 재현율', f"{hit/tot*100:.1f}%", f"임상 처치 이벤트 {tot:,}건 기준"))

# ④ 상태만으로 필요한 처치 단계를 추정할 수 있는가
hits = acts = ests = 0
for _, g in df.groupby('환자키'):
    f = g.iloc[0]
    e = needs(f['부상기전'], f['부상부위'], f['의식수준(AVPU)'],
              f['호흡수(회/분)'], f['맥박수(회/분)'], f['수축기혈압(mmHg)'])
    a = set(g['MARCH코드']) & set(PRIORITY)
    hits += len(e & a); acts += len(a); ests += len(e)
out.append(('needs() 재현율', f"{hits/acts*100:.1f}%", f"정밀도 {hits/ests*100:.1f}%"))

print(pd.DataFrame(out, columns=['검증 항목', '결과', '비고']).to_string(index=False))
print(f"\n초발 역행 패턴: {pd.Series(pairs).value_counts().to_dict()}")
