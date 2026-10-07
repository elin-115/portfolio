import { openContext, firstPage, humanPause, dumpDebug } from './browser.js';
import { S } from '../config/selectors.js';
import { log } from './log.js';

/**
 * 네이버 뉴스 / 블로그 검색 결과 수집.
 * 마크업이 자주 바뀌므로 셀렉터 후보를 순서대로 시도하고,
 * 일부가 깨져도 나머지는 건지도록 항목 단위로 감싼다.
 */

const clean = (s) => (s || '').replace(/\s+/g, ' ').trim();

/**
 * 페이지 안에서 실행되는 추출기.
 * 클래스명 대신 "링크의 href 패턴 + 그 링크를 감싼 문서 구조"로 뽑기 때문에
 * 네이버가 디자인 시스템을 바꿔도 잘 버틴다.
 */
function extractInPage({ mode, excludeSrc, postSrc, authorHint, minTitleLen, limit }) {
  // 스크린리더용 안내 문구가 링크 텍스트에 섞여 들어온다. 걷어낸다.
  const NOISE = /새\s*창\s*열림|Keep에\s*바로가기|본문\s*듣기|언론사\s*선정|네이버뉴스/g;
  const clean = (s) => (s || '').replace(NOISE, ' ').replace(/\s+/g, ' ').trim();
  const DATE_RE = /(\d+\s*(?:초|분|시간|일|주|개월|년)\s*전|\d{4}[.\-]\s*\d{1,2}[.\-]\s*\d{1,2}\.?)/;

  /**
   * 링크를 감싼 항목 박스를 찾는다.
   * 언론사명·작성일은 제목/요약보다 바깥에 있는 경우가 많아,
   * 날짜가 보일 때까지 한 단계씩 더 올라가되 너무 커지면 멈춘다.
   */
  const containerOf = (a, titleLen) => {
    let fallback = a.parentElement || a;
    let n = a.parentElement;
    for (let i = 0; i < 7 && n; i++, n = n.parentElement) {
      const text = clean(n.innerText);
      if (text.length > titleLen + 40 && fallback === (a.parentElement || a)) fallback = n;
      if (DATE_RE.test(text)) return n;            // 날짜까지 품은 박스를 찾았다
      if (text.length > titleLen + 1200) break;    // 여기서 더 가면 다른 항목까지 삼킨다
    }
    return fallback;
  };

  const anchors = [...document.querySelectorAll('a[href]')];

  if (mode === 'blog') {
    const postRe = new RegExp(postSrc);
    const groups = new Map();

    for (const a of anchors) {
      const m = a.href.match(postRe);
      if (!m) continue;
      const key = `${m[1]}/${m[2]}`;
      const text = clean(a.innerText);
      if (!text) continue;
      if (!groups.has(key)) groups.set(key, { url: a.href.split('?')[0], blogId: m[1], texts: [], nodes: [] });
      const g = groups.get(key);
      g.texts.push(text);
      g.nodes.push(a);
    }

    const out = [];
    for (const g of groups.values()) {
      // 같은 글에 제목 링크와 요약 링크가 함께 걸린다. 짧은 쪽이 제목, 긴 쪽이 요약.
      const sorted = [...new Set(g.texts)].sort((a, b) => a.length - b.length);
      const title = sorted.find((t) => t.length >= minTitleLen) || sorted[0];
      if (!title) continue;
      const summary = sorted[sorted.length - 1] === title ? '' : sorted[sorted.length - 1];

      const box = containerOf(g.nodes[0], title.length);
      const boxText = clean(box.innerText);
      // 프로필 링크는 글번호가 없는 blog.naver.com/<아이디> 형태다.
      // (제목 링크가 잡히지 않도록 글번호가 붙은 것은 제외한다)
      const profile = [...box.querySelectorAll(`a[href*="blog.naver.com/${g.blogId}"]`)]
        .find((el) => !/\/\d{6,}/.test(el.href));
      const author = clean(box.querySelector(authorHint)?.innerText)
        || clean(profile?.innerText)
        || g.blogId;

      out.push({
        type: 'blog',
        title,
        url: g.url,
        author: author.slice(0, 40),
        summary: summary.slice(0, 400),
        date: (boxText.match(DATE_RE) || [''])[0],
        body: '',
      });
      if (out.length >= limit) break;
    }
    return out;
  }

  // 뉴스: 같은 기사에 제목 링크와 요약 링크가 함께 걸린다.
  // 블로그와 마찬가지로 href로 묶고, 짧은 쪽을 제목 / 긴 쪽을 요약으로 본다.
  const excludeRe = new RegExp(excludeSrc);
  const groups = new Map();

  for (const a of anchors) {
    if (!/^https?:/.test(a.href) || excludeRe.test(a.href)) continue;
    const text = clean(a.innerText);
    if (!text) continue;
    const key = a.href.split('?')[0];
    if (!groups.has(key)) groups.set(key, { url: a.href, texts: [], nodes: [] });
    groups.get(key).texts.push(text);
    groups.get(key).nodes.push(a);
  }

  const out = [];
  for (const g of groups.values()) {
    const sorted = [...new Set(g.texts)].sort((a, b) => a.length - b.length);
    const title = sorted.find((t) => t.length >= minTitleLen);
    if (!title) continue;
    const summary = sorted[sorted.length - 1] === title ? '' : sorted[sorted.length - 1];

    const box = containerOf(g.nodes[0], title.length);
    let rest = clean(box.innerText).split(title).join(' ');
    if (summary) rest = rest.split(summary).join(' ');
    const date = (rest.match(DATE_RE) || [''])[0];
    rest = clean(rest.replace(DATE_RE, ' '));

    out.push({
      type: 'news',
      title,
      url: g.url,
      press: rest.slice(0, 25),
      summary: summary.slice(0, 400),
      date,
    });
    if (out.length >= limit) break;
  }
  return out;
}

async function collectNews(page, topic, limit) {
  // 최신순만 보면 키워드가 스쳐 지나간 엉뚱한 기사가 섞이고,
  // 관련도순만 보면 오래된 기사가 올라온다. 둘을 합쳐서 중복을 걷어낸다.
  const items = [];
  // 최신순에 절반, 나머지는 관련도순으로 채운다.
  // (최신순이 먼저 다 채워버리면 관련도순이 반영되지 않는다)
  const budget = { 1: Math.ceil(limit / 2), 0: limit };
  for (const sort of [1, 0]) {
    if (items.length >= budget[sort]) continue;
    await page.goto(S.newsSearch.url(topic, sort), { waitUntil: 'domcontentloaded' });
    await humanPause(1200, 2200);

    const found = await page.evaluate(extractInPage, {
      mode: 'news',
      excludeSrc: S.newsSearch.excludeHref.source,
      minTitleLen: S.newsSearch.minTitleLen,
      limit,
    }).catch(() => []);

    for (const it of found) {
      if (items.length >= budget[sort]) break;
      if (!items.some((o) => o.url === it.url || o.title === it.title)) items.push(it);
    }
  }

  if (!items.length) {
    await dumpDebug(page, `news-${topic}`);
    log.warn('collect', `뉴스 결과를 파싱하지 못했습니다: "${topic}"`);
  } else {
    log.info('collect', `뉴스 ${items.length}건 수집: "${topic}"`);
  }
  return items.map((i) => ({ ...i, topic }));
}

async function collectBlogs(page, topic, limit) {
  await page.goto(S.blogSearch.url(topic), { waitUntil: 'domcontentloaded' });
  await humanPause(1200, 2200);

  const items = await page.evaluate(extractInPage, {
    mode: 'blog',
    postSrc: S.blogSearch.postHref.source,
    authorHint: S.blogSearch.authorHint,
    minTitleLen: S.blogSearch.minTitleLen,
    limit,
  }).catch(() => []);

  if (!items.length) {
    await dumpDebug(page, `blog-${topic}`);
    log.warn('collect', `블로그 결과를 파싱하지 못했습니다: "${topic}"`);
  } else {
    log.info('collect', `블로그 ${items.length}건 수집: "${topic}"`);
  }
  return items.map((i) => ({ ...i, topic }));
}

/** 블로그 글 본문까지 읽어온다. 네이버 블로그는 본문이 #mainFrame iframe 안에 있다. */
async function fetchBlogBody(page, source, maxChars = 2500) {
  try {
    await page.goto(source.url, { waitUntil: 'domcontentloaded', timeout: 20_000 });
    await humanPause(600, 1200);

    // iframe 안이면 프레임으로 내려간다
    let scope = page;
    const frameEl = await page.locator(S.blogPost.frame).first().elementHandle().catch(() => null);
    if (frameEl) {
      const frame = await frameEl.contentFrame();
      if (frame) scope = frame;
    }

    for (const sel of S.blogPost.body) {
      const text = await scope.locator(sel).first().innerText({ timeout: 5000 }).catch(() => null);
      if (clean(text).length > 150) return clean(text).slice(0, maxChars);
    }
  } catch (e) {
    log.warn('collect', `본문 읽기 실패 (${source.title.slice(0, 20)}…): ${e.message.split('\n')[0]}`);
  }
  return '';
}

/**
 * 관심분야 목록에 대해 뉴스 + 인기 블로그 글을 모은다.
 * @param {string[]} topics
 * @param {object} opts
 * @param {number} opts.perTopic  관심분야당 수집 수 (뉴스/블로그 각각 절반씩)
 * @param {number} opts.withBody  본문까지 읽을 블로그 글 수
 */
export async function collectSources(topics, { perTopic = 8, withBody = 3 } = {}) {
  const half = Math.max(2, Math.ceil(perTopic / 2));
  let ctx;
  const all = [];

  try {
    ctx = await openContext({ headless: true });
    const page = await firstPage(ctx);

    for (const topic of topics) {
      log.step('collect', `"${topic}" 자료를 모으는 중…`);
      const news = await collectNews(page, topic, half);
      await humanPause();
      const blogs = await collectBlogs(page, topic, half);

      // 상위 블로그 글은 본문까지 읽어 글쓰기 근거로 쓴다
      for (const b of blogs.slice(0, withBody)) {
        b.body = await fetchBlogBody(page, b);
        await humanPause();
      }
      all.push(...news, ...blogs);
    }
  } finally {
    if (ctx) await ctx.close().catch(() => {});
  }

  log.info('collect', `총 ${all.length}건 수집 완료.`);
  return all.map((s, i) => ({ idx: i, ...s }));
}
