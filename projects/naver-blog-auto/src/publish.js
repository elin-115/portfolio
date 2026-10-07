import path from 'node:path';
import fs from 'node:fs';
import { openContext, firstPage, humanPause, humanKeys, sleep, dumpDebug } from './browser.js';
import { S, firstMatch } from '../config/selectors.js';
import { P } from './paths.js';
import { log } from './log.js';

/**
 * 스마트에디터 ONE 자동 작성 + 발행.
 *
 * 이 파일이 이 프로젝트에서 가장 깨지기 쉽다. 네이버가 클래스명(해시 접미사)을
 * 수시로 바꾸기 때문이다. 그래서:
 *  - 셀렉터는 전부 config/selectors.js 의 후보 배열에서 온다
 *  - 서식 적용이 실패해도 글 자체는 올라가도록 평문 폴백을 둔다
 *  - 실패하면 data/debug/ 에 스크린샷과 HTML을 남긴다
 */

/** 에디터가 들어있는 iframe으로 내려간다. iframe이 없으면 페이지 자체를 쓴다. */
async function getEditorFrame(page) {
  const handle = await page.locator(S.editor.frame).first().elementHandle({ timeout: 15_000 })
    .catch(() => null);
  if (!handle) return page;
  const frame = await handle.contentFrame();
  return frame || page;
}

/** 진입 시 뜨는 팝업들(이전 글 불러오기, 도움말 등)을 닫는다. */
async function dismissPopups(scope) {
  for (const sel of S.editor.dismiss) {
    try {
      const loc = scope.locator(sel).first();
      if (await loc.isVisible({ timeout: 1200 })) {
        await loc.click({ timeout: 2000 });
        log.info('publish', `팝업을 닫았습니다: ${sel}`);
        await humanPause(300, 700);
      }
    } catch { /* 없으면 그만 */ }
  }
}

/** 툴바 버튼을 눌러 서식을 적용한다. 못 찾으면 false를 돌려주고 호출부가 폴백한다. */
async function applyFormat(scope, kind) {
  const candidates = S.editor.toolbar[kind];
  if (!candidates) return false;
  const btn = await firstMatch(scope, candidates, { timeout: 1500 });
  if (!btn) return false;
  try {
    await btn.click({ timeout: 2500 });
    await humanPause(200, 500);
    return true;
  } catch { return false; }
}

/**
 * 인용구·구분선·이미지 뒤에 이어 쓸 새 문단을 만들고 커서를 옮긴다.
 *
 * 스마트에디터에서 인용구와 구분선은 "글자 서식"이 아니라 독립 컴포넌트다.
 * 그래서 버튼을 한 번 더 눌러도 해제되지 않고 같은 컴포넌트가 하나 더 생긴다.
 * (그렇게 해서 글 전체가 인용구 안으로 빨려 들어간 적이 있다)
 * 사람이 하듯 마지막 컴포넌트 아래 빈 영역을 클릭해 새 문단을 만드는 것이 확실하다.
 */
async function newParagraph(page, scope) {
  try {
    const content = await scope.locator('.se-content').first().boundingBox();
    const last = await scope.locator('.se-content .se-component').last().boundingBox();
    if (!content || !last) return false;
    const y = Math.min(last.y + last.height + 30, content.y + content.height - 8);
    await page.mouse.click(content.x + content.width / 2, y);
    await humanPause(300, 700);
    return true;
  } catch (e) {
    log.warn('publish', `새 문단 생성 실패: ${e.message.split('\n')[0]}`);
    return false;
  }
}

/** 커서가 놓인 자리에 한 줄 쓰고 줄바꿈. */
async function typeLine(page, text) {
  await humanKeys(page, text);
  await page.keyboard.press('Enter');
  await humanPause(150, 400);
}

/** 이미지 한 장을 본문에 넣는다. */
async function insertImage(page, scope, absPath) {
  const btn = await firstMatch(scope, S.editor.toolbar.image, { timeout: 3000 });
  if (!btn) {
    log.warn('publish', '이미지 툴바 버튼을 찾지 못했습니다. 이 이미지는 건너뜁니다.');
    return false;
  }
  try {
    const [chooser] = await Promise.all([
      page.waitForEvent('filechooser', { timeout: 15_000 }),
      btn.click({ timeout: 5000 }),
    ]);
    await chooser.setFiles(absPath);
    log.info('publish', `이미지 업로드: ${path.basename(absPath)}`);
    // 업로드·렌더링 대기
    await sleep(3500);
    await page.keyboard.press('Escape').catch(() => {});
    await humanPause(400, 900);
    return true;
  } catch (e) {
    log.warn('publish', `이미지 삽입 실패: ${e.message.split('\n')[0]}`);
    return false;
  }
}

/** 블록 배열을 에디터에 실제로 입력한다. */
async function writeBody(page, scope, post) {
  const body = await firstMatch(scope, S.editor.body, { timeout: 15_000 });
  if (!body) throw new Error('본문 입력 영역을 찾지 못했습니다. config/selectors.js의 editor.body를 확인하세요.');

  // 커서를 본문에 한 번 놓고, 이후에는 키보드만 쓴다.
  // (Enter마다 새 문단 요소가 생기므로 Locator에 대고 타이핑하면 첫 문단에 다 쌓인다)
  await body.click();
  await humanPause();

  let formatWarned = false;
  const warnOnce = () => {
    if (!formatWarned) {
      formatWarned = true;
      log.warn('publish', '툴바 서식을 적용하지 못해 평문으로 대체합니다. (글은 정상 발행됩니다)');
    }
  };

  for (const block of post.draft.blocks) {
    switch (block.type) {
      case 'heading': {
        const ok = await applyFormat(scope, 'bold');
        if (!ok) warnOnce();
        await typeLine(page, block.text);
        if (ok) await applyFormat(scope, 'bold'); // 굵게 해제
        break;
      }
      case 'paragraph':
        await typeLine(page, block.text);
        await page.keyboard.press('Enter');       // 문단 사이 여백
        break;

      case 'quote': {
        // 인용구는 컴포넌트라 "적용 → 입력 → 새 문단으로 탈출" 순서로 다뤄야 한다
        await newParagraph(page, scope);
        const ok = await applyFormat(scope, 'quote');
        if (!ok) { warnOnce(); await typeLine(page, `“${block.text}”`); break; }
        await humanKeys(page, block.text);
        await humanPause(300, 700);
        await newParagraph(page, scope);
        break;
      }
      case 'list':
        for (const item of block.items) await typeLine(page, `· ${item}`);
        await page.keyboard.press('Enter');
        break;

      case 'divider': {
        // 구분선 버튼은 빈 문단에 커서가 있을 때 뜨는 삽입 메뉴 안에 있다
        await newParagraph(page, scope);
        const ok = await applyFormat(scope, 'divider');
        if (!ok) { warnOnce(); await typeLine(page, '— — —'); break; }
        await newParagraph(page, scope);
        break;
      }
      case 'image': {
        if (!block.file) break; // AI 검수에서 탈락한 자리는 비워둔다
        const abs = path.resolve(P.root, block.file);
        if (!fs.existsSync(abs)) {
          log.warn('publish', `이미지 파일이 없습니다: ${block.file}`);
          break;
        }
        await newParagraph(page, scope);
        await insertImage(page, scope, abs);
        // 이미지 삽입 뒤 커서가 캡션 칸이나 이미지 안에 머물 수 있다
        await newParagraph(page, scope);
        if (block.caption) await typeLine(page, block.caption);
        break;
      }
      default:
        if (block.text) await typeLine(page, block.text);
    }
  }
}

/** 발행 레이어: 카테고리 → 태그 → 발행. */
async function doPublish(page, scope, post, { category }) {
  const openBtn = await firstMatch(page, S.editor.publishOpen, { timeout: 8000 })
    || await firstMatch(scope, S.editor.publishOpen, { timeout: 5000 });
  if (!openBtn) throw new Error('발행 버튼을 찾지 못했습니다.');

  await openBtn.click();
  await humanPause(900, 1600);

  // 발행 레이어는 iframe 밖(부모 문서)에 뜨기도 하고 안에 뜨기도 한다.
  // 어느 쪽인지 먼저 정하고, 그 레이어에서만 카테고리·태그를 건드린다.
  let layer = null;
  for (const cand of [page, scope]) {
    const hit = await firstMatch(cand, [...S.editor.tagInput, ...S.editor.categorySelect], { timeout: 2500 });
    if (hit) { layer = cand; break; }
  }
  if (!layer) {
    log.warn('publish', '발행 레이어를 찾지 못했습니다. 카테고리·태그 없이 진행합니다.');
  } else {
    if (category) {
      const sel = await firstMatch(layer, S.editor.categorySelect, { timeout: 2500 });
      if (sel) {
        await sel.click().catch(() => {});
        await humanPause(400, 800);
        const opt = layer.locator(S.editor.categoryOption(category)).first();
        if (await opt.count().catch(() => 0)) {
          await opt.click().catch(() => {});
          log.info('publish', `카테고리 선택: ${category}`);
        } else {
          log.warn('publish', `카테고리 "${category}"를 찾지 못해 기본값으로 둡니다.`);
          await page.keyboard.press('Escape').catch(() => {});
        }
        await humanPause(300, 700);
      }
    }

    const tagInput = await firstMatch(layer, S.editor.tagInput, { timeout: 2500 });
    if (tagInput && post.draft.tags?.length) {
      for (const tag of post.draft.tags.slice(0, 10)) {
        await tagInput.type(tag, { delay: 40 }).catch(() => {});
        await tagInput.press('Enter').catch(() => {});
        await humanPause(150, 400);
      }
      log.info('publish', `태그 ${post.draft.tags.length}개 입력`);
    }
  }

  // 레이어 안의 최종 발행 버튼
  const confirm = await firstMatch(page, S.editor.publishConfirm, { timeout: 6000 })
    || await firstMatch(scope, S.editor.publishConfirm, { timeout: 4000 });
  if (!confirm) throw new Error('최종 발행 버튼을 찾지 못했습니다.');

  await confirm.click();
  log.step('publish', '발행 요청을 보냈습니다. 결과를 기다리는 중…');

  await page.waitForLoadState('domcontentloaded', { timeout: 60_000 }).catch(() => {});
  await sleep(4000);
  return page.url();
}

/** 캡차/추가 인증이 걸렸는지 확인한다. 자동으로 뚫지 않고 사람에게 넘긴다. */
async function detectBlock(page) {
  const url = page.url();
  if (/captcha|nidlogin|deviceConfirm/i.test(url)) {
    return '네이버가 추가 인증(캡차 또는 로그인)을 요구합니다. ' +
           '헤드리스 대신 창을 띄운 상태로 다시 시도해 직접 인증해 주세요.';
  }
  const bodyText = await page.locator('body').innerText({ timeout: 3000 }).catch(() => '');
  if (/자동 ?입력 ?방지|보안 ?문자|비정상적인 접근/.test(bodyText)) {
    return '네이버가 자동입력 방지 문자를 요구합니다. 직접 처리가 필요합니다.';
  }
  return null;
}

/**
 * 포스트를 실제로 발행한다.
 * @param {object} post
 * @param {{blogId:string, category?:string, headless?:boolean}} opts
 */
export async function publishPost(post, { blogId, category = '', headless = false }) {
  let ctx;
  let page;
  try {
    log.step('publish', `"${post.draft.title}" 발행을 시작합니다.`);
    ctx = await openContext({ headless });
    page = await firstPage(ctx);

    await page.goto(S.editor.writeUrl(blogId), { waitUntil: 'domcontentloaded', timeout: 45_000 });
    await sleep(4000); // 에디터 iframe이 뜰 때까지
    await humanPause(1500, 2500);

    const blocked = await detectBlock(page);
    if (blocked) throw new Error(blocked);

    const scope = await getEditorFrame(page);
    await dismissPopups(scope);
    await dismissPopups(page);

    // 제목
    const title = await firstMatch(scope, S.editor.title, { timeout: 15_000 });
    if (!title) throw new Error('제목 입력란을 찾지 못했습니다. config/selectors.js의 editor.title을 확인하세요.');
    await title.click();
    await humanPause();
    await humanKeys(page, post.draft.title);
    log.info('publish', `제목 입력 완료: ${post.draft.title}`);
    await humanPause();

    // 본문
    await writeBody(page, scope, post);
    log.info('publish', '본문 입력 완료.');
    await humanPause(1000, 2000);

    // 발행
    const url = await doPublish(page, scope, post, { category });
    const finalUrl = /blog\.naver\.com/.test(url) && !/PostWriteForm/.test(url) ? url : null;
    if (!finalUrl) log.warn('publish', `발행 후 URL을 확정하지 못했습니다: ${url}`);

    return { ok: true, url: finalUrl };
  } catch (e) {
    if (page) await dumpDebug(page, 'publish-fail');
    log.error('publish', `발행 실패: ${e.message}`);
    throw e;
  } finally {
    if (ctx) await ctx.close().catch(() => {});
  }
}

/**
 * 셀렉터 점검 모드.
 * 에디터를 창으로 띄우고 어떤 후보가 실제로 잡히는지 출력한다.
 * 네이버가 마크업을 바꿨을 때 5분 안에 고치기 위한 도구.
 *   node src/publish.js --inspect <blogId>
 */
export async function inspectEditor(blogId) {
  const ctx = await openContext({ headless: false });
  const page = await firstPage(ctx);
  await page.goto(S.editor.writeUrl(blogId), { waitUntil: 'domcontentloaded' });
  await sleep(4000);

  const scope = await getEditorFrame(page);
  await dismissPopups(scope);
  await dismissPopups(page);

  const groups = {
    'editor.title': S.editor.title,
    'editor.body': S.editor.body,
    'toolbar.image': S.editor.toolbar.image,
    'toolbar.quote': S.editor.toolbar.quote,
    'toolbar.divider': S.editor.toolbar.divider,
    'toolbar.bold': S.editor.toolbar.bold,
    'publishOpen': S.editor.publishOpen,
  };

  console.log('\n=== 셀렉터 점검 결과 ===');
  for (const [name, cands] of Object.entries(groups)) {
    const hits = [];
    for (const sel of cands) {
      const n = await scope.locator(sel).count().catch(() => 0);
      const np = await page.locator(sel).count().catch(() => 0);
      if (n || np) hits.push(`${sel} (frame:${n}, page:${np})`);
    }
    console.log(hits.length ? `✅ ${name}\n   ${hits.join('\n   ')}` : `❌ ${name} — 후보 전부 실패`);
  }
  console.log('\n창을 열어뒀습니다. 확인 후 Ctrl+C로 종료하세요.\n');
  await sleep(10 * 60 * 1000);
  await ctx.close();
}

// CLI: node src/publish.js --inspect <blogId>
if (process.argv[1]?.endsWith('publish.js') && process.argv.includes('--inspect')) {
  const blogId = process.argv[process.argv.indexOf('--inspect') + 1];
  if (!blogId) { console.error('사용법: node src/publish.js --inspect <blogId>'); process.exit(1); }
  inspectEditor(blogId).catch((e) => { console.error(e); process.exit(1); });
}
