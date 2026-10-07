# -*- coding: utf-8 -*-
"""pipeline.py 결과(metrics.json)를 차트로 렌더링한다. 실행: python soms/charts.py"""
import os, sys, json
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(HERE, 'charts'); os.makedirs(OUT, exist_ok=True)

# 검증 통과 팔레트 (validate_palette.js, light surface — 6개 검사 전부 PASS)
TEAL, RUST, INDIGO = '#0E8E7E', '#C4552B', '#4A5FA8'
INK, INK2, MUTED, GRID, SURF = '#1F2A28', '#4A5551', '#8B948F', '#E8ECEA', '#FFFFFF'
SEQ = LinearSegmentedColormap.from_list('teal', ['#F2FAF8', '#0E5F55'])

plt.rcParams.update({
    'font.family': 'Apple SD Gothic Neo', 'axes.unicode_minus': False,
    'figure.facecolor': SURF, 'axes.facecolor': SURF,
    'axes.edgecolor': GRID, 'axes.linewidth': 0.8,
    'text.color': INK, 'axes.labelcolor': INK2,
    'xtick.color': INK2, 'ytick.color': INK2,
    'xtick.labelsize': 10, 'ytick.labelsize': 10, 'font.size': 10,
})

def _clean(ax, xgrid=True):
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    ax.spines['left'].set_color(GRID); ax.spines['bottom'].set_color(GRID)
    ax.set_axisbelow(True)
    ax.grid(axis='x' if xgrid else 'y', color=GRID, lw=0.8)
    ax.tick_params(length=0)

def title(fig, t, sub=None):
    fig.text(0.012, 0.965, t, fontsize=14, fontweight='bold', color=INK, va='top')
    if sub: fig.text(0.012, 0.905, sub, fontsize=9.5, color=MUTED, va='top')

M = json.load(open(os.path.join(HERE, 'metrics.json'), encoding='utf-8'))
T = M['중증도분류']
METHODS = [('START', 'START 규칙 단독'), ('ML', 'ML 모델 단독'), ('결합', '안전우선 결합')]

# ───────── 1. 분류 오류율 ─────────
fig, ax = plt.subplots(figsize=(8.6, 4.0))
y = np.arange(len(METHODS)); h = 0.34
under = [T[k]['과소분류율'] for k, _ in METHODS]
miss  = [T[k]['적색놓침률'] for k, _ in METHODS]
ax.barh(y + h/2, under, h, color=RUST,  label='과소분류율  (위중 환자를 가볍게 판정)')
ax.barh(y - h/2, miss,  h, color=TEAL,  label='적색 놓침률  (긴급 환자 미식별)')
for yy, v in zip(y + h/2, under):
    ax.text(v + 0.4, yy, f'{v}%', va='center', fontsize=10, fontweight='bold', color=INK)
for yy, v in zip(y - h/2, miss):
    ax.text(v + 0.4, yy, f'{v}%', va='center', fontsize=10, fontweight='bold', color=INK)
ax.axvline(5, color=INK2, ls=(0, (4, 3)), lw=1.2, zorder=3)
ax.text(5.4, -0.48, 'ACS-COT 기준  과소분류 5% 미만', fontsize=8.5, color=INK2, va='center')
ax.set_yticks(y, [lab for _, lab in METHODS], fontsize=10.5)
ax.set_ylim(len(METHODS)-0.5, -0.8); ax.set_xlim(0, 26); ax.set_xlabel('오류율 (%)', fontsize=9.5)
_clean(ax)
ax.legend(loc='lower right', frameon=False, fontsize=9.5, handlelength=1.1)
title(fig, '현장 중증도 분류 — 규칙과 모델을 합치면 둘 다보다 낫다',
      f"분류 결정 시점 {T['START']['대상']}명 (적색 {T['START']['적색수']}명) · 시나리오 템플릿 단위 교차검증 · 가상 데이터 내부 검증치")
fig.tight_layout(rect=[0, 0, 1, 0.86]); fig.savefig(f'{OUT}/1_분류오류율.png', dpi=200); plt.close(fig)

# ───────── 2. 혼동행렬 ─────────
LAB = ['녹색', '황색', '적색']
fig, axes = plt.subplots(1, 3, figsize=(9.8, 4.1))
for ax, (k, lab) in zip(axes, METHODS):
    cm = np.array(M['혼동행렬'][k])
    ax.imshow(cm, cmap=SEQ, vmin=0, vmax=cm.max())
    for i in range(3):
        for j in range(3):
            v = cm[i, j]
            ax.text(j, i, f'{v}', ha='center', va='center', fontsize=11,
                    fontweight='bold' if i == j else 'normal',
                    color='#FFFFFF' if v > cm.max()*0.55 else INK)
    ax.set_xticks(range(3), LAB, fontsize=9.5); ax.set_yticks(range(3), LAB, fontsize=9.5)
    ax.set_title(lab, fontsize=11, fontweight='bold', color=INK, pad=10)
    ax.set_xlabel('판정', fontsize=9, color=MUTED, labelpad=4)
    if ax is axes[0]: ax.set_ylabel('실제 기록', fontsize=9, color=MUTED, labelpad=4)
    for s in ax.spines.values(): s.set_color(GRID)
    ax.tick_params(length=0)
title(fig, '혼동행렬 — 오류는 모두 인접 등급 사이에서만 발생',
      '세 방식 모두 녹색↔적색을 맞바꾸는 오류 0건 · 결합은 적색 놓침을 15명까지 낮춘다')
fig.tight_layout(rect=[0.01, 0.02, 0.99, 0.80]); fig.savefig(f'{OUT}/2_혼동행렬.png', dpi=200); plt.close(fig)

# ───────── 3. 과제별 성능 ─────────
r, e, v, n, g = M['규칙엔진'], M['이송우선순위'], M['음성NLP'], M['필요처치단계'], M['소요시간회귀']
rows = [('Hot Zone 규칙 준수 판정', r['HotZone준수'], None, '규칙'),
        ('MARCH 순서 준수 판정',    r['순서준수'],    None, '규칙'),
        ('중증도 분류 (결합)',      T['결합']['정확도'], T['베이스라인'], '결합'),
        ('이송 우선순위',           e['정확도'],      e['베이스라인'], '모델'),
        ('음성근거 → 처치유형',      v['정확도'],      v['베이스라인'], '모델'),
        ('필요 처치단계 추정',       n['모델가중재현율'], n['수기가중재현율'], '모델')]
fig, ax = plt.subplots(figsize=(8.8, 4.8))
y = np.arange(len(rows))
ax.barh(y, [x[1] for x in rows], 0.5, color=[TEAL if x[3] != '규칙' else INDIGO for x in rows])
for i, (lab, val, base, kind) in enumerate(rows):
    ax.text(max(val, base or 0) + 1.5, i, f'{val}%', va='center', fontsize=10,
            fontweight='bold', color=INK)
    if base is not None:
        ax.plot([base, base], [i - 0.32, i + 0.32], color=RUST, lw=2, zorder=4)
        ax.text(base, i + 0.42, f'{base}', ha='center', va='top', fontsize=8, color=RUST)
ax.set_yticks(y, [x[0] for x in rows], fontsize=10.5); ax.invert_yaxis()
ax.set_xlim(0, 108); ax.set_xlabel('정확도 / 준수율 (%)', fontsize=9.5)
_clean(ax)
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
ax.legend(handles=[Patch(color=INDIGO, label='규칙 엔진'), Patch(color=TEAL, label='모델 · 결합'),
                   Line2D([0],[0], color=RUST, lw=2, label='베이스라인 / 수기 규칙')],
          loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=3,
          frameon=False, fontsize=9.5, handlelength=1.1)
title(fig, '과제별 성능 — 규칙과 모델의 역할 분담',
      f"공공 {M['데이터']['이벤트']:,}건 · 환자 {M['데이터']['환자']:,}명 · 템플릿 {M['데이터']['템플릿']}종 · 가상 데이터 내부 검증치   |   '필요 처치단계'는 모델이 수기 규칙에 못 미친다")
fig.tight_layout(rect=[0, 0.06, 1, 0.87]); fig.savefig(f'{OUT}/3_과제별성능.png', dpi=200); plt.close(fig)
print('저장:', OUT)
for f in sorted(os.listdir(OUT)): print('  ', f)
