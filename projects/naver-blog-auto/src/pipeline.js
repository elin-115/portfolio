import fs from 'node:fs';
import path from 'node:path';
import { P } from './paths.js';
import { collectSources } from './collect.js';
import { generateIdeas } from './ideas.js';
import { writeDraft } from './writer.js';
import { attachImages } from './images.js';
import { publishPost } from './publish.js';
import { getSettings, savePost, getPost, newPostId, publishedToday } from './store.js';
import { log } from './log.js';

/**
 * 파이프라인 오케스트레이션.
 * 각 단계는 독립적으로 호출 가능하다 (재시도/부분 실행을 위해).
 * status: idea → drafted → imaged → published | failed
 */

/**
 * 마지막 수집 결과. 글감을 고를 때와 섹션을 추가할 때 근거로 쓴다.
 * 메모리에만 두면 서버를 재시작하는 순간 글감이 사라지므로 파일에도 남긴다.
 */
const LAST_RUN_FILE = path.join(P.data, 'last-run.json');

function loadLastRun() {
  try { return JSON.parse(fs.readFileSync(LAST_RUN_FILE, 'utf8')); }
  catch { return { sources: [], ideas: [], at: null }; }
}

let lastRun = loadLastRun();

function saveLastRun(run) {
  lastRun = run;
  try { fs.writeFileSync(LAST_RUN_FILE, JSON.stringify(run, null, 2), 'utf8'); }
  catch (e) { log.warn('pipeline', `수집 결과 저장 실패: ${e.message}`); }
  return run;
}

export function getLastRun() {
  return lastRun;
}

/** 1단계: 자료 수집 + 글감 발굴 */
export async function findIdeas({ topics } = {}) {
  const s = getSettings();
  const list = (topics?.length ? topics : s.topics).filter(Boolean);
  if (!list.length) throw new Error('관심분야를 먼저 설정하세요.');

  const sources = await collectSources(list, {
    perTopic: s.sourcesPerTopic,
    withBody: Math.min(3, s.sourcesPerTopic),
  });
  const ideas = await generateIdeas(sources, {
    topics: list,
    count: s.ideaCount,
    model: s.model,
    notes: s.notes,
  });

  return saveLastRun({ sources, ideas, at: new Date().toISOString() });
}

/** 2단계: 글감 하나를 골라 초안 작성 */
export async function draftFromIdea(ideaId) {
  const s = getSettings();
  const idea = lastRun.ideas.find((i) => i.id === ideaId);
  if (!idea) throw new Error('글감을 찾을 수 없습니다. 글감 발굴을 다시 실행하세요.');

  const draft = await writeDraft(idea, lastRun.sources, {
    tone: s.tone,
    targetLength: s.targetLength,
    imagesPerPost: s.imagesPerPost,
    model: s.model,
    notes: s.notes,
  });

  const post = savePost({
    id: newPostId(),
    status: 'drafted',
    topic: idea.keywords?.[0] || s.topics[0] || '',
    idea,
    sources: (idea.sourceIdx?.length
      ? lastRun.sources.filter((x) => idea.sourceIdx.includes(x.idx))
      : lastRun.sources.slice(0, 5)
    ).map(({ idx, type, title, url, press, author, date }) =>
      ({ idx, type, title, url, press, author, date })),
    draft,
    publishedUrl: null,
    publishedAt: null,
    createdAt: new Date().toISOString(),
  });

  log.info('pipeline', `초안 저장 완료: ${post.id}`);
  return post;
}

/** 3단계: 이미지 수집 + AI 검수 */
export async function fillImages(postId) {
  const s = getSettings();
  const post = getPost(postId);
  await attachImages(post, { sources: s.imageSources, model: s.model });
  post.status = 'imaged';
  return savePost(post);
}

/** 4단계: 발행 (사용자가 검토 후 클릭) */
export async function publish(postId, { headless = false } = {}) {
  const s = getSettings();
  if (!s.blogId) throw new Error('설정에서 내 블로그 아이디를 먼저 입력하세요.');

  const done = publishedToday();
  if (done >= s.dailyLimit) {
    throw new Error(
      `오늘 이미 ${done}건을 발행했습니다 (한도 ${s.dailyLimit}건). ` +
      '계정 보호를 위해 막았습니다. 설정에서 한도를 바꿀 수 있습니다.',
    );
  }

  const post = getPost(postId);
  try {
    const result = await publishPost(post, {
      blogId: s.blogId,
      category: s.category,
      headless,
    });
    post.status = 'published';
    post.publishedUrl = result.url || null;
    post.publishedAt = new Date().toISOString();
    savePost(post);
    log.info('pipeline', `발행 완료: ${post.publishedUrl || '(URL 확인 실패)'}`);
    return post;
  } catch (e) {
    post.status = 'failed';
    post.lastError = e.message;
    savePost(post);
    throw e;
  }
}

/** 초안+이미지까지 한 번에 (발행 직전에서 멈춘다) */
export async function prepare(ideaId) {
  const post = await draftFromIdea(ideaId);
  return fillImages(post.id);
}
