import fs from 'node:fs';
import { openContext, firstPage, sleep } from './browser.js';
import { S } from '../config/selectors.js';
import { P } from './paths.js';
import { log } from './log.js';

/**
 * 네이버 로그인.
 * 아이디/비밀번호는 이 코드가 절대 다루지 않는다.
 * 실제 브라우저 창을 띄워 사용자가 직접 입력하고, 끝나면 세션만 로컬에 저장한다.
 */

let loginInFlight = null;

/** 로그인 창이 열려 있는 동안인지. (같은 프로필로 브라우저를 두 번 띄우면 안 된다) */
export function isLoginInFlight() {
  return loginInFlight !== null;
}

// 네이버 로그인 세션 쿠키. 이 두 개가 생기면 로그인된 것이다.
const AUTH_COOKIES = ['NID_AUT', 'NID_SES'];

async function hasAuthCookies(ctx) {
  const cookies = await ctx.cookies('https://www.naver.com').catch(() => []);
  const names = new Set(cookies.map((c) => c.name));
  return AUTH_COOKIES.every((n) => names.has(n));
}

/**
 * 대시보드 "네이버 로그인" 버튼이 부르는 함수. 창을 띄우고 완료될 때까지 기다린다.
 *
 * 완료 판정은 **쿠키로만** 한다. 페이지를 열거나 이동시키지 않는다.
 * 사용자가 비밀번호나 2차 인증을 입력하는 중에 폴링이 화면을 건드리면 안 되고,
 * 로그인 후 기기 등록 화면처럼 nid.naver.com에 머무는 경우도 놓치지 않기 위해서다.
 */
export function startLogin({ waitMs = 5 * 60 * 1000 } = {}) {
  if (loginInFlight) return loginInFlight;

  loginInFlight = (async () => {
    let ctx;
    try {
      log.step('auth', '로그인 창을 엽니다. 뜬 브라우저에서 직접 로그인해 주세요.');
      ctx = await openContext({ headless: false });
      const page = await firstPage(ctx);
      await page.goto(S.login.url, { waitUntil: 'domcontentloaded' });

      const deadline = Date.now() + waitMs;
      let lastNotice = 0;

      while (Date.now() < deadline) {
        await sleep(2000);
        if (ctx.pages().length === 0) throw new Error('사용자가 로그인 창을 닫았습니다.');

        if (await hasAuthCookies(ctx)) {
          await sleep(1500); // 쿠키가 자리잡을 시간을 조금 준다
          await ctx.storageState({ path: P.storageState });
          log.info('auth', '로그인 성공. 세션을 로컬에 저장했습니다.');
          return { ok: true, savedTo: 'data/session/' };
        }

        const left = Math.round((deadline - Date.now()) / 1000);
        if (left % 30 < 2 && left !== lastNotice) {
          lastNotice = left;
          log.info('auth', `로그인 대기 중… (${left}초 남음)`);
        }
      }
      throw new Error('제한 시간 안에 로그인이 완료되지 않았습니다.');
    } finally {
      if (ctx) await ctx.close().catch(() => {});
      loginInFlight = null;
    }
  })();

  return loginInFlight;
}

/** 저장된 세션이 아직 살아있는지 조용히 확인 (헤드리스). */
export async function checkSession() {
  // 로그인 창이 열려 있으면 같은 프로필을 두 번 열게 되어 세션이 깨진 것처럼 보인다.
  if (loginInFlight) return { loggedIn: false, pending: true, reason: '로그인 진행 중입니다.' };
  if (!fs.existsSync(P.profile)) return { loggedIn: false, reason: '저장된 세션이 없습니다.' };

  let ctx;
  try {
    ctx = await openContext({ headless: true });
    if (!(await hasAuthCookies(ctx))) {   // openContext가 이미 복구를 시도한 뒤다
      return { loggedIn: false, reason: '세션이 만료되었습니다. 다시 로그인하세요.' };
    }
    // 쿠키가 실제로 통하는지 한 번 확인한다.
    // 주의: naver.com 마크업에는 로그인 상태와 무관하게 숨겨진 로그인 링크가 하나 있다.
    // 그래서 개수가 아니라 "눈에 보이는" 로그인 링크가 있는지로 판정해야 한다.
    const page = await firstPage(ctx);
    await page.goto('https://www.naver.com', { waitUntil: 'domcontentloaded', timeout: 20_000 });
    const visibleLogin = await page.evaluate(() =>
      [...document.querySelectorAll('a[href*="nidlogin.login"]')]
        .filter((a) => a.offsetParent !== null).length,
    ).catch(() => 0);

    return visibleLogin === 0
      ? { loggedIn: true }
      : { loggedIn: false, reason: '세션이 만료되었습니다. 다시 로그인하세요.' };
  } catch (e) {
    return { loggedIn: false, reason: e.message };
  } finally {
    if (ctx) await ctx.close().catch(() => {});
  }
}

/** 세션 삭제 (로그아웃). */
export async function clearSession() {
  if (loginInFlight) throw new Error('로그인 창이 열려 있습니다. 먼저 닫아주세요.');
  fs.rmSync(P.profile, { recursive: true, force: true });
  fs.rmSync(P.storageState, { force: true });
  log.info('auth', '저장된 네이버 세션을 삭제했습니다.');
  return { ok: true };
}
