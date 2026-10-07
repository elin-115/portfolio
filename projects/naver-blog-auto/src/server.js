import express from 'express';
import path from 'node:path';
import fs from 'node:fs';
import { P, ensureDirs } from './paths.js';
import { log, bus } from './log.js';
import { getSettings, saveSettings, listPosts, getPost, deletePost, savePost, publishedToday } from './store.js';
import { checkAi } from './ai.js';
import { startLogin, checkSession, clearSession } from './auth.js';
import { findIdeas, draftFromIdea, fillImages, publish, getLastRun } from './pipeline.js';
import { addSection } from './section.js';

// .env 로드 (선택적 스톡 API 키)
try { process.loadEnvFile(path.join(P.root, '.env')); } catch { /* 없으면 그만 */ }

ensureDirs();
const app = express();
app.use(express.json({ limit: '5mb' }));
// 로컬에서 직접 고쳐 쓰는 앱이라, 캐시된 옛 파일이 뜨는 편이 훨씬 성가시다.
app.use(express.static(path.join(P.root, 'public'), { etag: false, maxAge: 0, setHeaders: (res) => res.set('Cache-Control', 'no-store') }));
// 검토 화면에서 이미지 미리보기용
app.use('/media', express.static(P.images));

/* ------------------------------ 작업 러너 ------------------------------ */
/* 수집·작성·발행은 몇 분씩 걸린다. HTTP 응답을 붙잡고 있지 않고
   즉시 jobId를 돌려준 뒤 진행 상황을 SSE로 흘려보낸다. */

const jobs = new Map();
let running = null;

function startJob(name, fn) {
  if (running) throw new Error(`이미 "${running}" 작업이 실행 중입니다. 끝나면 다시 시도하세요.`);
  const id = `job-${Date.now().toString(36)}`;
  running = name;
  jobs.set(id, { id, name, state: 'running', startedAt: Date.now() });
  bus.emit('job', { id, name, state: 'running' });

  fn()
    .then((result) => {
      jobs.set(id, { id, name, state: 'done', result });
      bus.emit('job', { id, name, state: 'done', result });
    })
    .catch((err) => {
      log.error(name, err.message);
      jobs.set(id, { id, name, state: 'error', error: err.message, detail: err.detail || null });
      bus.emit('job', { id, name, state: 'error', error: err.message, detail: err.detail || null });
    })
    .finally(() => { running = null; });

  return id;
}

const ok = (res, data) => res.json({ ok: true, ...data });
const fail = (res, e, code = 400) =>
  res.status(code).json({ ok: false, error: e.message || String(e), detail: e.detail || null });

/* ------------------------------ 상태 / 설정 ------------------------------ */

app.get('/api/status', async (req, res) => {
  const s = getSettings();
  ok(res, {
    settings: s,
    publishedToday: publishedToday(),
    running,
    stockKeys: {
      unsplash: Boolean(process.env.UNSPLASH_ACCESS_KEY),
      pexels: Boolean(process.env.PEXELS_API_KEY),
    },
    hasSession: fs.existsSync(P.profile),
  });
});

app.get('/api/settings', (req, res) => ok(res, { settings: getSettings() }));
app.put('/api/settings', (req, res) => {
  try { ok(res, { settings: saveSettings(req.body || {}) }); }
  catch (e) { fail(res, e); }
});

/* ------------------------------ 네이버 세션 ------------------------------ */

app.post('/api/login', (req, res) => {
  try { ok(res, { jobId: startJob('login', () => startLogin()) }); }
  catch (e) { fail(res, e); }
});

app.get('/api/session', async (req, res) => {
  try { ok(res, await checkSession()); }
  catch (e) { fail(res, e); }
});

app.delete('/api/session', async (req, res) => {
  try { ok(res, await clearSession()); }
  catch (e) { fail(res, e); }
});

/* ------------------------------ AI 점검 ------------------------------ */

app.get('/api/ai/check', async (req, res) => {
  try { ok(res, { ai: await checkAi() }); }
  catch (e) { fail(res, e, 200); } // 실패 사유를 UI에 그대로 보여준다
});

/* ------------------------------ 파이프라인 ------------------------------ */

app.post('/api/ideas', (req, res) => {
  try {
    const topics = Array.isArray(req.body?.topics) ? req.body.topics : undefined;
    ok(res, { jobId: startJob('ideas', () => findIdeas({ topics })) });
  } catch (e) { fail(res, e); }
});

app.get('/api/ideas', (req, res) => ok(res, getLastRun()));

app.post('/api/posts', (req, res) => {
  try {
    const { ideaId, withImages = true } = req.body || {};
    if (!ideaId) throw new Error('ideaId가 필요합니다.');
    ok(res, {
      jobId: startJob('draft', async () => {
        const post = await draftFromIdea(ideaId);
        return withImages ? fillImages(post.id) : post;
      }),
    });
  } catch (e) { fail(res, e); }
});

app.post('/api/posts/:id/images', (req, res) => {
  try { ok(res, { jobId: startJob('images', () => fillImages(req.params.id)) }); }
  catch (e) { fail(res, e); }
});

// 이미 쓴 글에 AI로 섹션을 덧붙인다 (예: "각 지점 구간별 타임라인 넣어줘")
app.post('/api/posts/:id/section', (req, res) => {
  try {
    const instruction = String(req.body?.instruction || '').trim();
    if (!instruction) throw new Error('어떤 섹션을 추가할지 입력하세요.');
    ok(res, { jobId: startJob('section', () => addSection(req.params.id, instruction)) });
  } catch (e) { fail(res, e); }
});

app.post('/api/posts/:id/publish', (req, res) => {
  try {
    const headless = req.body?.headless === true; // 기본은 창을 띄운다 (문제 시 사람이 개입)
    ok(res, { jobId: startJob('publish', () => publish(req.params.id, { headless })) });
  } catch (e) { fail(res, e); }
});

/* ------------------------------ 포스트 ------------------------------ */

app.get('/api/posts', (req, res) => ok(res, { posts: listPosts() }));

app.get('/api/posts/:id', (req, res) => {
  try { ok(res, { post: getPost(req.params.id) }); }
  catch (e) { fail(res, e, 404); }
});

// 검토 화면에서 제목·본문·태그를 손보고 저장
app.put('/api/posts/:id', (req, res) => {
  try {
    const post = getPost(req.params.id);
    const { title, tags, blocks } = req.body || {};
    if (title !== undefined) post.draft.title = String(title);
    if (Array.isArray(tags)) post.draft.tags = tags.map(String);
    if (Array.isArray(blocks)) post.draft.blocks = blocks;
    ok(res, { post: savePost(post) });
  } catch (e) { fail(res, e, 404); }
});

app.delete('/api/posts/:id', (req, res) => ok(res, { deleted: deletePost(req.params.id) }));

/* ------------------------------ 작업 상태 / 로그 SSE ------------------------------ */

app.get('/api/jobs/:id', (req, res) => {
  const job = jobs.get(req.params.id);
  if (!job) return fail(res, new Error('작업을 찾을 수 없습니다.'), 404);
  ok(res, { job });
});

app.get('/api/stream', (req, res) => {
  res.writeHead(200, {
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache',
    Connection: 'keep-alive',
  });
  res.write(`event: hello\ndata: ${JSON.stringify({ history: log.history() })}\n\n`);

  const onLog = (e) => res.write(`event: log\ndata: ${JSON.stringify(e)}\n\n`);
  const onJob = (e) => res.write(`event: job\ndata: ${JSON.stringify(e)}\n\n`);
  bus.on('log', onLog);
  bus.on('job', onJob);

  const ping = setInterval(() => res.write(': ping\n\n'), 25_000);
  req.on('close', () => {
    clearInterval(ping);
    bus.off('log', onLog);
    bus.off('job', onJob);
  });
});

/* ------------------------------ 기동 ------------------------------ */

const PORT = Number(process.env.PORT) || 3000;
app.listen(PORT, '127.0.0.1', () => {
  console.log(`\n  네이버 블로그 자동화 대시보드`);
  console.log(`  → http://localhost:${PORT}\n`);
  console.log(`  처음이라면: 대시보드에서 [네이버 로그인]과 [AI 연결 확인]을 먼저 눌러보세요.\n`);
});
