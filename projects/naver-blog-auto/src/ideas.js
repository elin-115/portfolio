import { askJson } from './ai.js';
import { log } from './log.js';

/** 수집한 원문을 프롬프트에 넣을 만큼만 압축한다. */
function condense(sources, maxPerSource = 400) {
  return sources.map((s) => {
    const parts = [
      `[${s.idx}] (${s.type === 'news' ? '뉴스' : '블로그'}) ${s.title}`,
      s.press || s.author ? `  출처: ${s.press || s.author}` : '',
      s.date ? `  일자: ${s.date}` : '',
      s.summary ? `  요약: ${s.summary.slice(0, 200)}` : '',
      s.body ? `  본문: ${s.body.slice(0, maxPerSource)}` : '',
    ].filter(Boolean);
    return parts.join('\n');
  }).join('\n\n');
}

const SYSTEM = `당신은 네이버 블로그를 10년 운영한 콘텐츠 기획자다.
검색 유입이 잘 되면서도 사람이 읽고 싶어지는 글감을 고른다.
낚시성 주제, 근거 없는 추측, 이미 식은 주제는 고르지 않는다.`;

/**
 * 수집 자료에서 글감 후보를 뽑는다.
 * @param {object[]} sources  collectSources() 결과
 * @param {object} opts
 */
export async function generateIdeas(sources, { topics = [], count = 6, model = 'sonnet', notes = '' } = {}) {
  if (!sources.length) throw new Error('수집된 자료가 없어 글감을 뽑을 수 없습니다.');

  log.step('ideas', `자료 ${sources.length}건에서 글감 ${count}개를 고르는 중…`);

  // 블로그 주인이 아는 사실은 검색으로 긁어온 자료보다 항상 우선한다.
  // 글감 단계에서 사실이 틀리면 그 뒤로 쓰는 글이 통째로 틀어진다.
  const notesBlock = notes.trim() ? `
## ⚠️ 블로그 주인이 알려준 사실 — 최우선
아래는 이 주제를 실제로 아는 사람이 알려준 내용이다.
**수집 자료와 충돌하면 무조건 이쪽이 옳다.** 이와 어긋나는 글감은 아예 만들지 마라.

${notes.trim()}
` : '';

  const prompt = `아래는 "${topics.join(', ')}" 분야에서 지금 검색 상위에 노출되는 뉴스와 블로그 글이다.
${notesBlock}
${condense(sources)}

---
이 자료를 근거로, 지금 쓰면 좋을 네이버 블로그 글감 ${count}개를 골라라.

원칙:
- 자료에 실제로 근거가 있는 주제만 고른다. 없는 사실을 지어내지 않는다.
- 서로 겹치지 않게, 각기 다른 각도로 고른다.
- 이미 남들이 똑같이 쓴 각도라면 한 겹 비틀어서 차별화한다.
- 제목은 검색어가 자연스럽게 들어가되 낚시성 문구는 쓰지 않는다.

각 글감에 대해 아래 JSON 배열 형식으로만 답하라:
[
  {
    "title": "블로그 글 제목 (25~40자)",
    "angle": "어떤 각도로 풀어낼지 한 문장",
    "keywords": ["검색 유입을 노릴 키워드", "3~5개"],
    "why": "왜 지금 이 주제인지, 위 자료의 어떤 점을 근거로 삼았는지",
    "sourceIdx": [참고한 자료의 [번호]들]
  }
]`;

  const ideas = await askJson({ prompt, system: SYSTEM, model, timeoutMs: 300_000 });
  if (!Array.isArray(ideas)) throw new Error('AI가 글감 배열을 반환하지 않았습니다.');

  const normalized = ideas
    .filter((i) => i && i.title)
    .map((i, n) => ({
      id: `idea-${n}`,
      title: String(i.title).trim(),
      angle: String(i.angle || '').trim(),
      keywords: Array.isArray(i.keywords) ? i.keywords.map(String) : [],
      why: String(i.why || '').trim(),
      sourceIdx: Array.isArray(i.sourceIdx) ? i.sourceIdx.map(Number).filter((x) => !Number.isNaN(x)) : [],
    }));

  log.info('ideas', `글감 ${normalized.length}개를 뽑았습니다.`);
  return normalized;
}
