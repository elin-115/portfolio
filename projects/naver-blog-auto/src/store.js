import fs from 'node:fs';
import path from 'node:path';
import { P, ensureDirs } from './paths.js';

ensureDirs();

export const DEFAULT_SETTINGS = {
  topics: ['홈트레이닝'],          // 관심분야
  blogId: '',                      // 내 블로그 아이디 (발행 대상)
  tone: '친근한 존댓말',
  notes: '',                       // 내가 아는 사실 / 글쓰기 지시 (수집 자료보다 우선)
  targetLength: 1800,              // 목표 글자수
  ideaCount: 6,                    // 한 번에 뽑을 글감 수
  sourcesPerTopic: 8,              // 관심분야당 수집할 원문 수
  imagesPerPost: 3,
  imageSources: ['local', 'ai'],
  category: '',                    // 발행 카테고리명 (비우면 기본값)
  dailyLimit: 3,                   // 하루 최대 발행 수 (계정 보호)
  model: 'sonnet',
};

function readJson(file, fallback) {
  try { return JSON.parse(fs.readFileSync(file, 'utf8')); }
  catch { return fallback; }
}

function writeJson(file, value) {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, JSON.stringify(value, null, 2), 'utf8');
}

export function getSettings() {
  return { ...DEFAULT_SETTINGS, ...readJson(P.settings, {}) };
}

export function saveSettings(patch) {
  const next = { ...getSettings(), ...patch };
  writeJson(P.settings, next);
  return next;
}

export function newPostId() {
  const d = new Date();
  const stamp = [
    d.getFullYear(),
    String(d.getMonth() + 1).padStart(2, '0'),
    String(d.getDate()).padStart(2, '0'),
    String(d.getHours()).padStart(2, '0'),
    String(d.getMinutes()).padStart(2, '0'),
    String(d.getSeconds()).padStart(2, '0'),
  ].join('');
  return `${stamp}-${Math.random().toString(36).slice(2, 6)}`;
}

const postFile = (id) => path.join(P.posts, `${id}.json`);

export function savePost(post) {
  post.updatedAt = new Date().toISOString();
  writeJson(postFile(post.id), post);
  return post;
}

export function getPost(id) {
  const p = readJson(postFile(id), null);
  if (!p) throw new Error(`포스트를 찾을 수 없습니다: ${id}`);
  return p;
}

export function listPosts() {
  if (!fs.existsSync(P.posts)) return [];
  return fs.readdirSync(P.posts)
    .filter((f) => f.endsWith('.json'))
    .map((f) => readJson(path.join(P.posts, f), null))
    .filter(Boolean)
    .sort((a, b) => (b.createdAt || '').localeCompare(a.createdAt || ''));
}

export function deletePost(id) {
  try { fs.unlinkSync(postFile(id)); return true; } catch { return false; }
}

/** 오늘 발행한 글 수 (dailyLimit 체크용) */
export function publishedToday() {
  const today = new Date().toISOString().slice(0, 10);
  return listPosts().filter(
    (p) => p.status === 'published' && (p.publishedAt || '').slice(0, 10) === today,
  ).length;
}
