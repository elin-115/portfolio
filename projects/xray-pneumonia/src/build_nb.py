# xray_improved.py (# %% 셀 구분) → xray_improved.ipynb 변환
import json, re
import sys
name = sys.argv[1] if len(sys.argv) > 1 else 'xray_improved'
src = open(name + '.py').read()
cells = []
for block in re.split(r'^# %%', src, flags=re.M)[1:]:
    if block.startswith(' [markdown]'):
        body = block[len(' [markdown]'):].strip('\n')
        txt = '\n'.join(l[2:] if l.startswith('# ') else l.lstrip('#') for l in body.split('\n'))
        cells.append({"cell_type": "markdown", "metadata": {}, "source": txt.strip()})
    else:
        lines = block.strip('\n').split('\n')
        if 'google.colab' in block:   # 코랩 셀은 코드 줄만 주석 해제
            lines = [l[2:] if l[2:].startswith(('from ', 'drive.', '!unzip', '%cd')) else l for l in lines]
        cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": '\n'.join(lines)})
nb = {"cells": cells, "metadata": {"accelerator": "GPU", "kernelspec": {"name": "python3", "display_name": "Python 3"}}, "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open(name + '.ipynb', 'w'), ensure_ascii=False, indent=1)
