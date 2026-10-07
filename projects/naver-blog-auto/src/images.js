import fs from 'node:fs';
import path from 'node:path';
import { askJson } from './ai.js';
import { openContext, firstPage, humanPause } from './browser.js';
import { S } from '../config/selectors.js';
import { P } from './paths.js';
import { log } from './log.js';

/**
 * 이미지 후보 수집 → 다운로드 → AI가 실제로 열어보고 어울리는지 판정.
 * 소스 우선순위: 무료 스톡(저작권 안전) > 로컬 폴더 > 네이버 이미지(옵트인)
 */

const UA = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36';

/* ---------------------------------- 후보 수집 ---------------------------------- */

async function fromUnsplash(query, n) {
  const key = process.env.UNSPLASH_ACCESS_KEY;
  if (!key) return [];
  try {
    const res = await fetch(
      `https://api.unsplash.com/search/photos?query=${encodeURIComponent(query)}&per_page=${n}&orientation=landscape`,
      { headers: { Authorization: `Client-ID ${key}` } },
    );
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const json = await res.json();
    return (json.results || []).map((r) => ({
      source: 'unsplash',
      url: r.urls?.regular,
      credit: `Unsplash / ${r.user?.name || 'unknown'}`,
      pageUrl: r.links?.html,
      license: '상업적 사용 가능 (Unsplash License)',
    })).filter((c) => c.url);
  } catch (e) {
    log.warn('images', `Unsplash 실패: ${e.message}`);
    return [];
  }
}

async function fromPexels(query, n) {
  const key = process.env.PEXELS_API_KEY;
  if (!key) return [];
  try {
    const res = await fetch(
      `https://api.pexels.com/v1/search?query=${encodeURIComponent(query)}&per_page=${n}&orientation=landscape`,
      { headers: { Authorization: key } },
    );
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const json = await res.json();
    return (json.photos || []).map((p) => ({
      source: 'pexels',
      url: p.src?.large,
      credit: `Pexels / ${p.photographer || 'unknown'}`,
      pageUrl: p.url,
      license: '상업적 사용 가능 (Pexels License)',
    })).filter((c) => c.url);
  } catch (e) {
    log.warn('images', `Pexels 실패: ${e.message}`);
    return [];
  }
}

/**
 * AI 이미지 생성 (Pollinations — 무료, API 키 불필요).
 *
 * 중요: 실제 사진처럼 만들지 않는다. **일러스트 스타일로 고정**한다.
 * 등산 후기에 AI가 만든 사실적인 산 사진을 넣으면 읽는 사람은 글쓴이가 찍은
 * 사진으로 받아들인다. 그건 독자를 속이는 일이다.
 * 그래서 설명용 삽화로만 쓰이도록 스타일을 강제하고, 캡션에도 AI 생성임을 남긴다.
 */
function fromAiGenerated(query, n) {
  const STYLE = 'flat vector editorial illustration, clean simple shapes, muted natural palette, '
    + 'no text, no watermark, not photorealistic';
  return Array.from({ length: n }, (_, i) => ({
    source: 'ai',
    url: `https://image.pollinations.ai/prompt/${encodeURIComponent(`${query}, ${STYLE}`)}`
       + `?width=1280&height=720&nologo=true&model=flux&seed=${Date.now() % 100000 + i * 7}`,
    credit: 'AI 생성 이미지',
    license: 'AI 생성 — 실제 촬영 사진이 아닙니다',
  }));
}

function fromLocal(n) {
  try {
    return fs.readdirSync(P.localImages)
      .filter((f) => /\.(jpe?g|png|webp)$/i.test(f))
      .slice(0, n)
      .map((f) => ({
        source: 'local',
        localPath: path.join(P.localImages, f),
        credit: '내 로컬 이미지',
        license: '직접 보유',
      }));
  } catch { return []; }
}

/** 네이버 이미지 검색 (저작권 확인이 필요한 소스 — 설정에서 켜야 동작). */
async function fromNaver(query, n) {
  let ctx;
  try {
    ctx = await openContext({ headless: true });
    const page = await firstPage(ctx);
    await page.goto(S.imageSearch.url(query), { waitUntil: 'domcontentloaded' });
    await humanPause(1000, 1800);

    for (const sel of S.imageSearch.thumb) {
      const urls = await page.locator(sel).evaluateAll(
        (els) => els.map((e) => e.getAttribute('src') || e.getAttribute('data-lazy-src')).filter(Boolean),
      ).catch(() => []);
      const usable = urls.filter((u) => u.startsWith('http')).slice(0, n);
      if (usable.length) {
        return usable.map((url) => ({
          source: 'naver',
          url,
          credit: '네이버 이미지 검색',
          license: '⚠️ 저작권 미확인 — 사용 전 출처 확인 필요',
        }));
      }
    }
    return [];
  } catch (e) {
    log.warn('images', `네이버 이미지 검색 실패: ${e.message}`);
    return [];
  } finally {
    if (ctx) await ctx.close().catch(() => {});
  }
}

/* ---------------------------------- 다운로드 ---------------------------------- */

async function download(candidate, destDir, name) {
  fs.mkdirSync(destDir, { recursive: true });

  if (candidate.localPath) {
    const dest = path.join(destDir, `${name}${path.extname(candidate.localPath)}`);
    fs.copyFileSync(candidate.localPath, dest);
    return dest;
  }

  // Pollinations는 요청을 받고 나서 그림을 그리기 때문에 응답이 느리다
  const res = await fetch(candidate.url, {
    signal: AbortSignal.timeout(candidate.source === 'ai' ? 120_000 : 30_000),
    headers: {
      'User-Agent': UA,
      // 네이버 CDN은 referer 없으면 막는다
      ...(candidate.source === 'naver' ? { Referer: 'https://search.naver.com/' } : {}),
    },
  });
  if (!res.ok) throw new Error(`다운로드 실패 HTTP ${res.status}`);

  const type = res.headers.get('content-type') || '';
  const ext = type.includes('png') ? '.png' : type.includes('webp') ? '.webp' : '.jpg';
  const dest = path.join(destDir, `${name}${ext}`);
  fs.writeFileSync(dest, Buffer.from(await res.arrayBuffer()));

  const { size } = fs.statSync(dest);
  if (size < 8000) { // 썸네일/플레이스홀더는 버린다
    fs.unlinkSync(dest);
    throw new Error(`이미지가 너무 작습니다 (${size}B)`);
  }
  return dest;
}

/* ---------------------------------- AI 검수 ---------------------------------- */

const JUDGE_SYSTEM = `당신은 블로그 편집자다. 사진이 글의 해당 문단에 정말 어울리는지 냉정하게 본다.
검색어와 대충 맞기만 한 사진, 워터마크가 박힌 사진, 글의 톤과 어긋나는 사진은 떨어뜨린다.`;

/**
 * 이미지 파일을 AI가 실제로 열어서 판정한다.
 * `--allowedTools Read` 로 Read 툴을 허용해야 claude가 파일을 볼 수 있다.
 */
async function judge(filePath, { context, caption, source, model = 'sonnet' }) {
  const rel = path.relative(P.root, filePath);
  const aiNote = source === 'ai' ? `
이 이미지는 AI가 생성한 삽화다. 실제 사진이 아니므로 사진 같은 사실성은 기대하지 않는다.
대신 이런 점을 엄격히 본다:
- 형태가 뭉개지거나 알아볼 수 없게 그려지지 않았는가 (AI 생성물의 흔한 결함)
- 손가락·글자·물건 개수가 이상하게 그려지지 않았는가
- 삽화로서 문단 내용을 실제로 설명해주는가
알아보기 어렵거나 어색하면 주저 없이 탈락시켜라.
` : '';
  const prompt = `이미지 파일을 열어서 봐라: ${rel}
${aiNote}

이 사진은 아래 블로그 문단 옆에 들어갈 예정이다.

문단: "${context.slice(0, 500)}"
넣으려는 캡션: "${caption}"

사진을 직접 보고 판단하라:
- 이 문단의 내용과 실제로 맞아떨어지는가?
- 워터마크, 깨진 이미지, 알아보기 힘든 사진은 아닌가?
- 블로그 본문에 넣었을 때 어색하지 않은가?

아래 JSON 하나만 출력하라:
{
  "fit": 0~100 사이 점수,
  "verdict": "pass" 또는 "reject",
  "reason": "그렇게 판단한 이유 한 문장 (한국어)",
  "altText": "이 사진을 설명하는 대체 텍스트 (한국어)"
}

fit이 60 미만이면 verdict는 반드시 "reject"다.`;

  return askJson({
    prompt,
    system: JUDGE_SYSTEM,
    allowTools: ['Read'],
    model,
    timeoutMs: 240_000,
    cwd: P.root,
  });
}

/* ---------------------------------- 메인 ---------------------------------- */

/**
 * 초안의 image 블록들을 채운다.
 * @param {object} post   저장된 포스트 (draft.blocks 포함)
 * @param {object} opts
 */
export async function attachImages(post, {
  sources = ['stock', 'local', 'naver'],
  candidatesPerBlock = 4,
  model = 'sonnet',
} = {}) {
  const imageBlocks = post.draft.blocks.filter((b) => b.type === 'image');
  if (!imageBlocks.length) {
    log.info('images', '이미지 블록이 없습니다.');
    return post;
  }

  const destDir = path.join(P.images, post.id);
  log.step('images', `이미지 ${imageBlocks.length}자리를 채우는 중…`);

  for (const [i, block] of imageBlocks.entries()) {
    const query = block.query || post.draft.title;
    log.info('images', `[${i + 1}/${imageBlocks.length}] "${query}" 후보를 모으는 중…`);

    /** @type {any[]} */
    let candidates = [];
    // 직접 찍은 사진이 가장 정직하다. 그다음이 AI 삽화.
    if (sources.includes('local')) {
      candidates.push(...fromLocal(candidatesPerBlock));
    }
    if (sources.includes('ai') && candidates.length < candidatesPerBlock) {
      candidates.push(...fromAiGenerated(query, Math.min(2, candidatesPerBlock - candidates.length)));
    }
    if (sources.includes('stock')) {
      candidates.push(...await fromUnsplash(query, candidatesPerBlock));
      if (candidates.length < candidatesPerBlock) {
        candidates.push(...await fromPexels(query, candidatesPerBlock - candidates.length));
      }
    }
    if (sources.includes('naver') && candidates.length < candidatesPerBlock) {
      candidates.push(...await fromNaver(query, candidatesPerBlock - candidates.length));
    }

    if (!candidates.length) {
      log.warn('images', `후보를 못 찾았습니다: "${query}". 이 자리는 비웁니다.`);
      block.verdict = { verdict: 'reject', reason: '이미지 후보를 찾지 못했습니다.', fit: 0 };
      continue;
    }

    // 앞 문단을 판정 문맥으로 쓴다
    const blockIdx = post.draft.blocks.indexOf(block);
    const context = post.draft.blocks
      .slice(0, blockIdx).reverse()
      .find((b) => b.type === 'paragraph')?.text || post.draft.summary || post.draft.title;

    const tried = [];
    let chosen = null;

    for (const [n, cand] of candidates.entries()) {
      let file;
      try {
        file = await download(cand, destDir, `${block.id}-${n}`);
      } catch (e) {
        log.warn('images', `후보 ${n + 1} 다운로드 실패: ${e.message}`);
        continue;
      }

      let v;
      try {
        v = await judge(file, { context, caption: block.caption, source: cand.source, model });
      } catch (e) {
        log.warn('images', `후보 ${n + 1} AI 판정 실패: ${e.message}`);
        continue;
      }

      const record = {
        file: path.relative(P.root, file),
        source: cand.source,
        credit: cand.credit,
        license: cand.license,
        pageUrl: cand.pageUrl || null,
        fit: Number(v.fit) || 0,
        verdict: v.verdict === 'pass' && Number(v.fit) >= 60 ? 'pass' : 'reject',
        reason: String(v.reason || ''),
        altText: String(v.altText || ''),
      };
      tried.push(record);
      log.info('images', `  후보 ${n + 1} (${cand.source}): ${record.fit}점 ${record.verdict === 'pass' ? '통과' : '탈락'} — ${record.reason}`);

      if (record.verdict === 'pass') { chosen = record; break; }
    }

    // 통과가 없으면 그중 가장 점수 높은 것도 쓰지 않는다 (사용자 검토 화면에 사유를 남긴다)
    block.candidates = tried;
    if (chosen) {
      block.file = chosen.file;
      block.verdict = chosen;
      block.caption = block.caption || chosen.altText;
      // AI가 만든 그림을 직접 찍은 사진처럼 보이게 두지 않는다
      if (chosen.source === 'ai' && !/AI/.test(block.caption)) {
        block.caption = `${block.caption} (AI 생성 이미지)`;
      }
    } else {
      block.file = null;
      block.verdict = tried.sort((a, b) => b.fit - a.fit)[0]
        || { verdict: 'reject', reason: '모든 후보가 탈락했습니다.', fit: 0 };
      log.warn('images', `"${query}" — 통과한 이미지가 없습니다.`);
    }
  }

  const filled = imageBlocks.filter((b) => b.file).length;
  log.info('images', `이미지 ${filled}/${imageBlocks.length}자리를 채웠습니다.`);
  return post;
}
