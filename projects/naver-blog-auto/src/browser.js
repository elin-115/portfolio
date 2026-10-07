import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import fs from 'node:fs';
import path from 'node:path';
import { chromium } from 'playwright';
import { P, ensureDirs } from './paths.js';
import { log } from './log.js';

const execFileAsync = promisify(execFile);
ensureDirs();

let installChecked = false;

/** 크로미움 바이너리가 없으면 자동 설치한다. */
export async function ensureBrowserInstalled() {
  if (installChecked) return;
  try {
    const exe = chromium.executablePath();
    if (fs.existsSync(exe)) { installChecked = true; return; }
  } catch { /* 아래에서 설치 */ }

  log.step('browser', 'Playwright 크로미움이 없어 설치합니다. 최초 1회, 몇 분 걸립니다…');
  await execFileAsync('npx', ['playwright', 'install', 'chromium'], {
    cwd: P.root,
    maxBuffer: 1024 * 1024 * 20,
    timeout: 15 * 60 * 1000,
  });
  log.info('browser', '크로미움 설치 완료.');
  installChecked = true;
}

const LAUNCH_ARGS = [
  '--disable-blink-features=AutomationControlled',
  '--no-default-browser-check',
  '--no-first-run',
  '--lang=ko-KR',
];

/**
 * 로그인 상태를 계속 들고 있어야 하므로 persistent context(=실제 크롬 프로필)를 쓴다.
 * 프로필 폴더가 곧 "입장권"이고, data/session/ 아래에만 저장된다.
 */
export async function openContext({ headless = true } = {}) {
  await ensureBrowserInstalled();
  const ctx = await chromium.launchPersistentContext(P.profile, {
    headless,
    args: LAUNCH_ARGS,
    viewport: { width: 1440, height: 950 },
    locale: 'ko-KR',
    timezoneId: 'Asia/Seoul',
    userAgent:
      'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) ' +
      'Chrome/131.0.0.0 Safari/537.36',
  });
  ctx.setDefaultTimeout(20_000);
  // webdriver 흔적 제거 (탐지 회피가 아니라, 정상 렌더링을 위한 최소한의 정리)
  await ctx.addInitScript(() => {
    Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
  });
  await restoreCookiesIfMissing(ctx);
  return ctx;
}

/**
 * 크롬 프로필에 로그인 쿠키가 없으면 저장해둔 storageState에서 되살린다.
 *
 * 크롬은 쿠키를 메모리에 두고 가끔 디스크에 쓴다. 그래서 프로세스가 비정상 종료되면
 * 분명 로그인했는데도 프로필에는 쿠키가 없는 상태가 된다.
 * 그때 사용자에게 다시 로그인시키지 않기 위한 복구 경로다.
 */
async function restoreCookiesIfMissing(ctx) {
  if (!fs.existsSync(P.storageState)) return false;
  try {
    const current = await ctx.cookies('https://www.naver.com').catch(() => []);
    if (current.some((c) => c.name === 'NID_AUT')) return false; // 이미 있다

    const saved = JSON.parse(fs.readFileSync(P.storageState, 'utf8'));
    const naverCookies = (saved.cookies || []).filter((c) => (c.domain || '').includes('naver'));
    if (!naverCookies.length) return false;

    await ctx.addCookies(naverCookies);
    log.info('browser', '프로필 쿠키가 비어 있어 저장된 세션에서 복구했습니다.');
    return true;
  } catch (e) {
    log.warn('browser', `세션 복구 실패: ${e.message}`);
    return false;
  }
}

/** 열려있는 첫 페이지를 재사용하거나 새로 만든다. */
export async function firstPage(ctx) {
  const pages = ctx.pages();
  return pages.length ? pages[0] : await ctx.newPage();
}

export const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/** 사람처럼 보이도록 약간의 흔들림을 준 대기. */
export const humanPause = (min = 400, max = 1200) =>
  sleep(min + Math.floor(Math.random() * (max - min)));

/** 사람처럼 한 글자씩 입력. */
export async function humanType(locator, text, { min = 12, max = 45 } = {}) {
  for (const ch of text) {
    await locator.type(ch, { delay: min + Math.random() * (max - min) });
  }
}

/**
 * 포커스가 있는 곳에 키보드로 직접 입력한다.
 *
 * 에디터처럼 Enter를 칠 때마다 새 문단 요소가 생기는 곳에서는 Locator에 대고
 * 타이핑하면 안 된다. 처음 잡은 요소(=첫 문단)에 모든 글이 쌓여버린다.
 * 사람이 하듯 커서를 한 번 놓고, 그다음부터는 키보드만 쓴다.
 */
export async function humanKeys(page, text, { min = 12, max = 45 } = {}) {
  for (const ch of text) {
    await page.keyboard.type(ch, { delay: min + Math.random() * (max - min) });
  }
}

/** 실패 시 원인 파악용 스크린샷 + HTML 덤프. */
export async function dumpDebug(page, label) {
  try {
    const stamp = new Date().toISOString().replace(/[:.]/g, '-');
    const base = path.join(P.debug, `${stamp}_${label}`);
    await page.screenshot({ path: `${base}.png`, fullPage: true });
    fs.writeFileSync(`${base}.html`, await page.content(), 'utf8');
    log.warn('browser', `실패 시점을 저장했습니다: data/debug/${path.basename(base)}.png`);
    return `${base}.png`;
  } catch (e) {
    log.warn('browser', `디버그 덤프 실패: ${e.message}`);
    return null;
  }
}
