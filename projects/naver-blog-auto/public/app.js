/* 네이버 블로그 자동화 대시보드 — 라이브러리 없음, 빌드 없음 */

const $ = (id) => document.getElementById(id);
const el = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text != null) n.textContent = text;
  return n;
};

const state = {
  settings: null,
  ideas: [],
  sources: [],
  post: null,        // 검토 중인 포스트
  jobs: new Map(),   // jobId -> { onDone, onError }
};

/* ------------------------------- API ------------------------------- */

async function api(path, opts = {}) {
  const res = await fetch(path, {
    headers: { 'Content-Type': 'application/json' },
    ...opts,
    body: opts.body ? JSON.stringify(opts.body) : undefined,
  });
  const json = await res.json().catch(() => ({ ok: false, error: `HTTP ${res.status}` }));
  if (!json.ok) throw Object.assign(new Error(json.error || '요청 실패'), { detail: json.detail });
  return json;
}

/** 작업을 시작하고 SSE로 완료를 기다린다. */
function runJob(jobId, { onDone, onError } = {}) {
  state.jobs.set(jobId, { onDone, onError });
}

/* ------------------------------- 상태칩 ------------------------------- */

function setChip(node, cls, label) {
  node.classList.remove('ok', 'bad', 'busy');
  if (cls) node.classList.add(cls);
  node.querySelector('.label').textContent = label;
}

async function refreshStatus() {
  const { settings, publishedToday, stockKeys } = await api('/api/status');
  state.settings = settings;
  state.stockKeys = stockKeys;
  $('chip-quota').textContent = `오늘 발행 ${publishedToday}/${settings.dailyLimit}`;
}

async function refreshNaver() {
  const chip = $('chip-naver');
  setChip(chip, 'busy', '네이버 확인 중…');
  try {
    const r = await api('/api/session');
    setChip(chip, r.loggedIn ? 'ok' : 'bad',
      r.loggedIn ? `네이버 로그인됨${r.id ? ` · ${r.id}` : ''}` : '네이버 로그인 필요');
    chip.title = r.reason || '클릭하면 로그인 창이 열립니다';
  } catch (e) {
    setChip(chip, 'bad', '네이버 확인 실패');
    chip.title = e.message;
  }
}

async function refreshAi() {
  const chip = $('chip-ai');
  setChip(chip, 'busy', 'AI 확인 중…');
  const res = await fetch('/api/ai/check').then((r) => r.json());
  if (res.ok && res.ai?.ok) {
    setChip(chip, 'ok', `AI 연결됨 · ${(res.ai.ms / 1000).toFixed(1)}s`);
    chip.title = `모델: ${res.ai.model}`;
  } else {
    setChip(chip, 'bad', 'AI 연결 안 됨');
    chip.title = res.error || '터미널에서 claude 실행 후 /login 하세요';
  }
}

/* ------------------------------- 글감 ------------------------------- */

function renderIdeas() {
  const box = $('ideas');
  box.innerHTML = '';
  if (!state.ideas.length) {
    box.append(el('div', 'empty', '아직 글감이 없습니다. 관심분야를 넣고 [글감 찾기]를 눌러보세요.'));
    return;
  }
  for (const idea of state.ideas) {
    const card = el('div', 'card');
    card.append(el('h3', null, idea.title));
    if (idea.angle) card.append(el('p', 'angle', idea.angle));
    if (idea.why) card.append(el('p', 'why', idea.why));

    if (idea.keywords?.length) {
      const tags = el('div', 'tags');
      idea.keywords.forEach((k) => tags.append(el('span', 'tag', k)));
      card.append(tags);
    }

    const srcs = state.sources.filter((s) => idea.sourceIdx?.includes(s.idx)).slice(0, 3);
    if (srcs.length) {
      const list = el('div', 'srcs');
      srcs.forEach((s) => {
        const a = el('a', null, `${s.type === 'news' ? '📰' : '✍️'} ${s.title}`);
        a.href = s.url; a.target = '_blank'; a.rel = 'noreferrer';
        list.append(a);
      });
      card.append(list);
    }

    const btn = el('button', 'primary', '이 글감으로 쓰기');
    btn.onclick = () => draft(idea.id, btn);
    card.append(btn);
    box.append(card);
  }
}

function findIdeas() {
  const btn = $('btn-ideas');
  const topics = $('topics').value.split(',').map((t) => t.trim()).filter(Boolean);
  btn.disabled = true;
  btn.textContent = '수집 중…';

  api('/api/ideas', { method: 'POST', body: { topics } })
    .then(({ jobId }) => runJob(jobId, {
      onDone: async () => {
        const run = await fetch('/api/ideas').then((r) => r.json());
        state.ideas = run.ideas || [];
        state.sources = run.sources || [];
        renderIdeas();
        btn.disabled = false; btn.textContent = '글감 찾기';
        if (topics.length) api('/api/settings', { method: 'PUT', body: { topics } }).catch(() => {});
      },
      onError: () => { btn.disabled = false; btn.textContent = '글감 찾기'; },
    }))
    .catch((e) => {
      alert(e.message);
      btn.disabled = false; btn.textContent = '글감 찾기';
    });
}

/* ------------------------------- 초안 ------------------------------- */

function draft(ideaId, btn) {
  btn.disabled = true;
  btn.textContent = '글 쓰는 중…';
  api('/api/posts', { method: 'POST', body: { ideaId, withImages: true } })
    .then(({ jobId }) => runJob(jobId, {
      onDone: (post) => {
        btn.disabled = false; btn.textContent = '이 글감으로 쓰기';
        openDraft(post);
        loadPosts();
      },
      onError: () => { btn.disabled = false; btn.textContent = '이 글감으로 쓰기'; },
    }))
    .catch((e) => {
      alert(e.message);
      btn.disabled = false; btn.textContent = '이 글감으로 쓰기';
    });
}

function editable(node, onInput) {
  node.contentEditable = 'true';
  node.spellcheck = false;
  node.addEventListener('input', () => onInput(node.textContent));
  return node;
}

function imageBlockNode(block, postId) {
  const wrap = el('div', 'd-image');
  const v = block.verdict;

  if (block.file) {
    const img = el('img');
    img.src = `/media/${postId}/${block.file.split('/').pop()}`;
    img.alt = v?.altText || block.caption || '';
    wrap.append(img);
  } else {
    wrap.append(el('div', 'noimg', v?.reason ? `이미지 없음\n${v.reason}` : '이미지 없음'));
  }

  const info = el('div', 'info');
  if (block.caption) info.append(editable(el('div', 'caption', block.caption), (t) => { block.caption = t; }));

  if (v) {
    const line = el('div', 'verdict');
    const score = el('span', `score ${v.verdict === 'pass' ? 'pass' : 'reject'}`,
      `${v.fit ?? 0}점 ${v.verdict === 'pass' ? '통과' : '탈락'}`);
    line.append(score, document.createTextNode(` — ${v.reason || ''}`));
    info.append(line);

    if (v.license) {
      const risky = /미확인/.test(v.license);
      info.append(el('div', `license${risky ? ' risk' : ''}`,
        `${v.credit || ''}${v.credit ? ' · ' : ''}${v.license}`));
    }
  }
  if (block.query) info.append(el('div', 'license', `검색어: ${block.query}`));

  wrap.append(info);
  return wrap;
}

function openDraft(post) {
  state.post = post;
  $('panel-draft').hidden = false;

  const d = post.draft;
  $('draft-meta').textContent =
    `${d.blocks.length}블록 · 약 ${d.charCount || 0}자 · 태그 ${(d.tags || []).join(', ') || '없음'} · 상태 ${post.status}`;

  const box = $('draft');
  box.innerHTML = '';
  box.append(editable(el('div', 'd-title', d.title), (t) => { d.title = t; }));

  for (const b of d.blocks) {
    switch (b.type) {
      case 'heading':
        box.append(editable(el('div', 'd-heading', b.text), (t) => { b.text = t; })); break;
      case 'paragraph':
        box.append(editable(el('div', 'd-paragraph', b.text), (t) => { b.text = t; })); break;
      case 'quote':
        box.append(editable(el('div', 'd-quote', b.text), (t) => { b.text = t; })); break;
      case 'list': {
        const ul = el('ul', 'd-list');
        b.items.forEach((item, i) => {
          ul.append(editable(el('li', null, item), (t) => { b.items[i] = t; }));
        });
        box.append(ul); break;
      }
      case 'divider': box.append(el('hr', 'd-divider')); break;
      case 'image': box.append(imageBlockNode(b, post.id)); break;
      default: if (b.text) box.append(el('div', 'd-paragraph', b.text));
    }
  }
  $('panel-draft').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

async function saveDraft() {
  if (!state.post) return;
  const btn = $('btn-save');
  btn.disabled = true;
  try {
    const { post } = await api(`/api/posts/${state.post.id}`, {
      method: 'PUT',
      body: {
        title: state.post.draft.title,
        tags: state.post.draft.tags,
        blocks: state.post.draft.blocks,
      },
    });
    state.post = post;
    btn.textContent = '저장됨 ✓';
    setTimeout(() => { btn.textContent = '수정 저장'; }, 1600);
  } catch (e) { alert(e.message); }
  btn.disabled = false;
}

function reimage() {
  if (!state.post) return;
  const btn = $('btn-reimage');
  btn.disabled = true; btn.textContent = '이미지 찾는 중…';
  api(`/api/posts/${state.post.id}/images`, { method: 'POST' })
    .then(({ jobId }) => runJob(jobId, {
      onDone: (post) => { openDraft(post); btn.disabled = false; btn.textContent = '이미지 다시 찾기'; },
      onError: () => { btn.disabled = false; btn.textContent = '이미지 다시 찾기'; },
    }))
    .catch((e) => { alert(e.message); btn.disabled = false; btn.textContent = '이미지 다시 찾기'; });
}

// 자주 쓰는 지시를 버튼으로 (클릭하면 입력창에 채워진다)
const PRESETS = [
  '각 지점 구간별 타임라인 넣어줘',
  '준비물 체크리스트 표로 정리해줘',
  '자주 묻는 질문 3개 넣어줘',
  '비용 정리 섹션 넣어줘',
  '주의사항·안전 팁 넣어줘',
];

function renderPresets() {
  const box = $('presets');
  box.innerHTML = '';
  PRESETS.forEach((text) => {
    const b = el('button', 'preset', text);
    b.onclick = () => { $('ask').value = text; $('ask').focus(); };
    box.append(b);
  });
}

function askSection() {
  if (!state.post) return;
  const instruction = $('ask').value.trim();
  if (!instruction) return $('ask').focus();

  const btn = $('btn-ask');
  btn.disabled = true;
  btn.textContent = '쓰는 중…';

  api(`/api/posts/${state.post.id}/section`, { method: 'POST', body: { instruction } })
    .then(({ jobId }) => runJob(jobId, {
      onDone: (post) => {
        btn.disabled = false; btn.textContent = '섹션 추가';
        $('ask').value = '';
        openDraft(post);
      },
      onError: (msg) => {
        btn.disabled = false; btn.textContent = '섹션 추가';
        alert(`섹션 추가 실패\n\n${msg}`);
      },
    }))
    .catch((e) => {
      alert(e.message);
      btn.disabled = false; btn.textContent = '섹션 추가';
    });
}

async function publishPost() {
  if (!state.post) return;
  if (!state.settings?.blogId) {
    alert('설정에서 내 블로그 아이디를 먼저 입력하세요.');
    return openSettings();
  }
  if (!confirm(`"${state.post.draft.title}"\n\n네이버에 실제로 발행합니다. 진행할까요?`)) return;

  await saveDraft();
  const btn = $('btn-publish');
  btn.disabled = true; btn.textContent = '발행 중…';

  api(`/api/posts/${state.post.id}/publish`, { method: 'POST', body: { headless: false } })
    .then(({ jobId }) => runJob(jobId, {
      onDone: (post) => {
        btn.disabled = false; btn.textContent = '네이버에 발행';
        state.post = post;
        refreshStatus(); loadPosts();
        if (post.publishedUrl && confirm('발행 완료! 글을 열어볼까요?')) window.open(post.publishedUrl, '_blank');
        else alert('발행이 완료되었습니다.');
      },
      onError: (msg) => {
        btn.disabled = false; btn.textContent = '네이버에 발행';
        alert(`발행 실패\n\n${msg}\n\ndata/debug/ 폴더의 스크린샷을 확인하세요.`);
        loadPosts();
      },
    }))
    .catch((e) => { alert(e.message); btn.disabled = false; btn.textContent = '네이버에 발행'; });
}

/* ------------------------------- 보관함 ------------------------------- */

async function loadPosts() {
  const { posts } = await api('/api/posts');
  const box = $('posts');
  box.innerHTML = '';
  if (!posts.length) return box.append(el('div', 'empty', '저장된 글이 없습니다.'));

  for (const p of posts.slice(0, 20)) {
    const row = el('div', 'post-row');
    row.append(el('span', `badge ${p.status}`, p.status));

    const t = el('span', 't', p.draft?.title || '(제목 없음)');
    t.style.cursor = 'pointer';
    t.onclick = async () => openDraft((await api(`/api/posts/${p.id}`)).post);
    row.append(t);

    row.append(el('span', 'when', (p.createdAt || '').slice(5, 16).replace('T', ' ')));

    if (p.publishedUrl) {
      const a = el('a', null, '보기');
      a.href = p.publishedUrl; a.target = '_blank'; a.rel = 'noreferrer';
      a.style.cssText = 'color:var(--blue);font-size:12px;text-decoration:none';
      row.append(a);
    }

    const del = el('button', 'ghost small', '삭제');
    del.onclick = async () => {
      if (!confirm('이 글을 삭제할까요?')) return;
      await api(`/api/posts/${p.id}`, { method: 'DELETE' });
      loadPosts();
    };
    row.append(del);
    box.append(row);
  }
}

/* ------------------------------- 설정 ------------------------------- */

function openSettings() {
  const s = state.settings;
  $('s-blogId').value = s.blogId || '';
  $('s-category').value = s.category || '';
  $('s-notes').value = s.notes || '';
  $('s-tone').value = s.tone;
  $('s-targetLength').value = s.targetLength;
  $('s-ideaCount').value = s.ideaCount;
  $('s-sourcesPerTopic').value = s.sourcesPerTopic;
  $('s-imagesPerPost').value = s.imagesPerPost;
  $('s-dailyLimit').value = s.dailyLimit;
  $('s-model').value = s.model;
  $('s-src-ai').checked = s.imageSources.includes('ai');
  $('s-src-stock').checked = s.imageSources.includes('stock');
  $('s-src-local').checked = s.imageSources.includes('local');
  $('s-src-naver').checked = s.imageSources.includes('naver');

  const k = state.stockKeys || {};
  $('s-keys').innerHTML = (k.unsplash || k.pexels)
    ? `스톡 API 키 감지됨: ${[k.unsplash && 'Unsplash', k.pexels && 'Pexels'].filter(Boolean).join(', ')}`
    : '스톡 API 키가 없습니다. <code>.env.example</code>을 <code>.env</code>로 복사하고 무료 키를 넣으면 저작권 안전한 사진을 씁니다.';

  $('modal-settings').hidden = false;
}

async function saveSettings() {
  const sources = [];
  if ($('s-src-local').checked) sources.push('local');
  if ($('s-src-ai').checked) sources.push('ai');
  if ($('s-src-stock').checked) sources.push('stock');
  if ($('s-src-naver').checked) sources.push('naver');

  await api('/api/settings', {
    method: 'PUT',
    body: {
      blogId: $('s-blogId').value.trim(),
      category: $('s-category').value.trim(),
      notes: $('s-notes').value.trim(),
      tone: $('s-tone').value.trim(),
      targetLength: Number($('s-targetLength').value),
      ideaCount: Number($('s-ideaCount').value),
      sourcesPerTopic: Number($('s-sourcesPerTopic').value),
      imagesPerPost: Number($('s-imagesPerPost').value),
      dailyLimit: Number($('s-dailyLimit').value),
      model: $('s-model').value,
      imageSources: sources,
    },
  });
  $('modal-settings').hidden = true;
  refreshStatus();
}

/* ------------------------------- SSE ------------------------------- */

function addLog(e) {
  const box = $('logs');
  const line = el('div', `logline lv-${e.level}`);
  line.append(el('span', 'scope', `[${e.scope}]`), el('span', 'msg', e.message));
  box.append(line);
  if (box.children.length > 400) box.firstChild.remove();
  box.scrollTop = box.scrollHeight;
}

function connectStream() {
  const es = new EventSource('/api/stream');

  es.addEventListener('hello', (ev) => {
    JSON.parse(ev.data).history.forEach(addLog);
  });
  es.addEventListener('log', (ev) => addLog(JSON.parse(ev.data)));

  es.addEventListener('job', (ev) => {
    const job = JSON.parse(ev.data);
    const handlers = state.jobs.get(job.id);

    if (job.state === 'done') {
      // 로그인 작업은 등록된 핸들러가 없어도 상태칩을 갱신한다
      if (job.name === 'login') refreshNaver();
      handlers?.onDone?.(job.result);
      state.jobs.delete(job.id);
    } else if (job.state === 'error') {
      if (job.name === 'login') { refreshNaver(); alert(`로그인 실패\n\n${job.error}`); }
      handlers?.onError?.(job.error);
      state.jobs.delete(job.id);
    }
  });

  es.onerror = () => { /* EventSource가 알아서 재연결한다 */ };
}

/* ------------------------------- 부팅 ------------------------------- */

$('chip-naver').onclick = () => {
  setChip($('chip-naver'), 'busy', '로그인 창 여는 중…');
  api('/api/login', { method: 'POST' })
    .then(({ jobId }) => runJob(jobId, { onDone: refreshNaver, onError: refreshNaver }))
    .catch((e) => { alert(e.message); refreshNaver(); });
};
$('chip-ai').onclick = refreshAi;
$('btn-ideas').onclick = findIdeas;
$('btn-save').onclick = saveDraft;
$('btn-reimage').onclick = reimage;
$('btn-ask').onclick = askSection;
$('ask').onkeydown = (e) => { if (e.key === 'Enter') askSection(); };
$('btn-publish').onclick = publishPost;
$('btn-refresh-posts').onclick = loadPosts;
$('btn-settings').onclick = openSettings;
$('btn-close-settings').onclick = () => { $('modal-settings').hidden = true; };
$('btn-save-settings').onclick = () => saveSettings().catch((e) => alert(e.message));
$('btn-clear-logs').onclick = () => { $('logs').innerHTML = ''; };
$('btn-logout').onclick = async () => {
  if (!confirm('저장된 네이버 세션을 삭제할까요? 다시 로그인해야 합니다.')) return;
  await api('/api/session', { method: 'DELETE' });
  $('modal-settings').hidden = true;
  refreshNaver();
};
$('modal-settings').onclick = (e) => {
  if (e.target === $('modal-settings')) $('modal-settings').hidden = true;
};
$('topics').onkeydown = (e) => { if (e.key === 'Enter') findIdeas(); };

(async function boot() {
  connectStream();
  renderPresets();
  await refreshStatus();
  $('topics').value = (state.settings.topics || []).join(', ');
  renderIdeas();
  loadPosts();
  refreshNaver();
  refreshAi();
})();
