# -*- coding: utf-8 -*-
"""군 데이터를 학습에 추가하면 공공 성능이 올라가는가 — 4폴드 비교."""
import matplotlib; matplotlib.use('Agg')
import warnings, json, io, contextlib, os
warnings.filterwarnings('ignore')
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

plt.rcParams.update({'font.family':'Apple SD Gothic Neo','axes.unicode_minus':False,
                     'figure.facecolor':'#FFFFFF','axes.facecolor':'#FFFFFF'})
TEAL, RUST, INK, INK2, MUTED, GRID = '#0E8E7E','#C4552B','#1F2A28','#4A5551','#8B948F','#E8ECEA'

nb = json.load(open('notebooks/soms_analysis.ipynb', encoding='utf-8'))
code = [''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code']
g = {}
with contextlib.redirect_stdout(io.StringIO()):
    for i, s in enumerate(code, 1):
        exec(compile(s, f'c{i}', 'exec'), g)
        if i >= 16: break
df = g['df']; LEVEL = {'녹색':0,'황색':1,'적색':2}
FEAT = ['의식점수','호흡수(회/분)','맥박수(회/분)','수축기혈압(mmHg)','쇼크지수','경과_분']

tri  = df[(df['유형']=='환자분류') & (df['분류색상']!='흑색')]
pubT = tri[tri['영역']=='공공']; milT = tri[tri['영역']=='군']
tmpl = list(pubT['세부상황'].unique())

rows = []
for k in range(0, 8, 2):
    hold = tmpl[k:k+2]
    te     = pubT[pubT['세부상황'].isin(hold)]
    tr_pub = pubT[~pubT['세부상황'].isin(hold)]
    tr_mix = pd.concat([tr_pub, milT])
    med = tr_pub[FEAT].median()
    Xte, yte = te[FEAT].fillna(med), te['분류색상']
    r = {'폴드': f'폴드 {k//2+1}', 'n': len(te)}
    for lab, tr in [('공공만', tr_pub), ('공공+군', tr_mix)]:
        m = RandomForestClassifier(n_estimators=300, random_state=0,
                                   class_weight={'녹색':1,'황색':1,'적색':10})
        m.fit(tr[FEAT].fillna(med), tr['분류색상'])
        p  = m.predict(Xte)
        yn = yte.map(LEVEL).values; pn = pd.Series(p).map(LEVEL).values
        r[f'{lab}_정확도'] = accuracy_score(yte, p)*100
        r[f'{lab}_과소']   = (pn < yn).mean()*100
    rows.append(r)

T = pd.DataFrame(rows)
avg = {'폴드':'평균', 'n': int(T['n'].sum())}
for c in T.columns:
    if c not in ('폴드','n'): avg[c] = T[c].mean()
T = pd.concat([T, pd.DataFrame([avg])], ignore_index=True)
print(T.round(1).to_string(index=False))
print(f"\n학습 규모 : 공공만 {len(pubT)}명 → 공공+군 {len(pubT)+len(milT)}명 (군 {len(milT)}명 추가)")
T.to_csv('domain_compare.csv', index=False, encoding='utf-8-sig')

# ── 차트 : 두 조건을 잇는 점 그래프 (붙어 있을수록 '차이 없음') ──
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
y = np.arange(len(T))[::-1]
for ax, (c1, c2, title, xlab) in zip(axes, [
        ('공공만_정확도','공공+군_정확도','정확도 — 높을수록 좋음','정확도 (%)'),
        ('공공만_과소','공공+군_과소','과소분류율 — 낮을수록 좋음','과소분류율 (%)')]):
    for i, yy in enumerate(y):
        a, b = T[c1][i], T[c2][i]
        ax.plot([a, b], [yy, yy], color=GRID, lw=3, zorder=1, solid_capstyle='round')
        ax.scatter([a],[yy], s=95, color=TEAL, zorder=3, label='공공만 학습' if i==0 else '')
        ax.scatter([b],[yy], s=95, color=RUST, zorder=3, label='공공+군 학습' if i==0 else '')
    for i, yy in enumerate(y):
        w = 'bold' if T['폴드'][i]=='평균' else 'normal'
        ax.text(T[[c1,c2]].iloc[i].max()+ (1.2 if 'ac' in c1 or '정확도' in c1 else 0.45), yy,
                f'{T[c1][i]:.1f} → {T[c2][i]:.1f}', va='center', fontsize=9,
                color=INK, fontweight=w)
    ax.set_yticks(y, [f"{p}\n(n={n})" for p,n in zip(T['폴드'],T['n'])], fontsize=9.5)
    ax.set_title(title, fontsize=11.5, fontweight='bold', color=INK, pad=10)
    ax.set_xlabel(xlab, fontsize=9.5, color=INK2)
    for s in ('top','right'): ax.spines[s].set_visible(False)
    ax.spines['left'].set_color(GRID); ax.spines['bottom'].set_color(GRID)
    ax.grid(axis='x', color=GRID, lw=0.8); ax.set_axisbelow(True); ax.tick_params(length=0)
axes[0].set_xlim(76, 95); axes[1].set_xlim(-0.6, 12)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', frameon=False, fontsize=10, ncol=2,
           bbox_to_anchor=(0.5, 0.005), handletextpad=0.4, columnspacing=2.5)
fig.text(0.012, 0.965, '군 데이터를 학습에 추가해도 공공 성능은 달라지지 않는다',
         fontsize=14, fontweight='bold', color=INK, va='top')
fig.text(0.012, 0.905, f'공공 템플릿 8종을 2종씩 4폴드로 검증 · 학습셋만 교체 '
         f'(공공 {len(pubT)}명 → 공공+군 {len(pubT)+len(milT)}명) · 가상 데이터 내부 검증치',
         fontsize=9, color=MUTED, va='top')
fig.tight_layout(rect=[0, 0.09, 1, 0.86])
fig.savefig('charts/4_군데이터_비교.png', dpi=200)
print('\n차트 저장 : charts/4_군데이터_비교.png')
