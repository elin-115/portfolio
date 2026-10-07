import { askJson } from './ai.js';
import { log } from './log.js';

/**
 * 글감 + 근거 자료로 블로그 본문을 만든다.
 * 출력은 "블록 배열" — 스마트에디터 서식에 1:1로 매핑되고,
 * 이미지를 문단 사이 정확한 위치에 꽂을 수 있다.
 */

const SYSTEM = `당신은 네이버 블로그 글을 쓰는 사람이다.
AI가 쓴 티가 나는 글투를 쓰지 않는다:
- "~에 대해 알아보겠습니다", "결론적으로", "다양한", "중요한 역할을 합니다" 같은 상투구를 피한다
- 불릿만 잔뜩 나열하고 끝내지 않는다. 문장으로 설명한다
- 과장된 감탄사와 이모지 남발을 하지 않는다
직접 겪어본 사람이 차분히 설명하는 톤으로 쓴다.`;

const BLOCK_SPEC = `블록 종류:
- {"type":"heading","text":"소제목"}          — 본문을 나누는 소제목
- {"type":"paragraph","text":"문단 내용"}      — 3~4줄 분량의 문단
- {"type":"quote","text":"강조할 한 문장"}     — 인용구 서식으로 들어감
- {"type":"list","items":["항목1","항목2"]}    — 목록
- {"type":"divider"}                          — 구분선
- {"type":"image","query":"이미지 검색어(영어)","caption":"사진 설명"}  — 이미지 자리`;

function sourceContext(sources, idxs) {
  const picked = idxs?.length ? sources.filter((s) => idxs.includes(s.idx)) : sources.slice(0, 6);
  return picked.map((s) => [
    `[${s.idx}] ${s.title}`,
    s.press || s.author ? `출처: ${s.press || s.author}` : '',
    s.summary ? `요약: ${s.summary.slice(0, 300)}` : '',
    s.body ? `본문: ${s.body.slice(0, 1200)}` : '',
  ].filter(Boolean).join('\n')).join('\n\n');
}

/**
 * @param {object} idea      generateIdeas() 항목
 * @param {object[]} sources 수집 자료 전체
 * @param {object} opts
 */
export async function writeDraft(idea, sources, {
  tone = '친근한 존댓말',
  targetLength = 1800,
  imagesPerPost = 3,
  model = 'sonnet',
  notes = '',
} = {}) {
  log.step('writer', `"${idea.title}" 초안을 쓰는 중…`);

  // 블로그 주인이 직접 아는 사실은 검색으로 긁어온 자료보다 항상 우선한다.
  // (예: 무박 종주를 1박 종주 후기와 섞어버리는 사고를 막는다)
  const notesBlock = notes.trim() ? `
## ⚠️ 블로그 주인이 알려준 사실 — 최우선
아래는 이 주제를 실제로 아는 사람이 직접 알려준 내용이다.
**수집 자료와 충돌하면 무조건 이쪽이 옳다.** 자료 쪽을 버려라.

${notes.trim()}
` : '';

  const prompt = `아래 자료를 근거로 네이버 블로그 글을 새로 써라.
${notesBlock}
## 쓸 글
제목 방향: ${idea.title}
풀어낼 각도: ${idea.angle}
노릴 키워드: ${idea.keywords.join(', ')}

## 근거 자료
${sourceContext(sources, idea.sourceIdx)}

---
## 반드시 지킬 것

1. **원문을 베끼지 마라.** 자료에서 사실과 흐름만 가져오고, 문장은 전부 새로 쓴다.
   자료의 문장이 3어절 이상 그대로 들어가면 안 된다. (네이버 유사문서 필터에 걸린다)
2. 분량은 공백 포함 ${targetLength}자 내외.
3. 톤: ${tone}
4. 도입부는 독자가 겪는 상황이나 질문으로 연다. 자기소개나 인사로 시작하지 않는다.
5. 소제목(heading)을 3~5개 넣어 스캔하기 쉽게 만든다.
6. 이미지 자리(image 블록)를 ${imagesPerPost}개 넣는다. 도입부 직후와 소제목 사이에 자연스럽게 배치한다.
   image의 query는 스톡 사진 검색용이므로 **영어**로, 구체적인 장면을 묘사해서 쓴다.
   (예: "person stretching at home living room morning" — "health" 같은 추상어는 쓰지 마라)
7. 자료에 없는 통계·가격·날짜를 지어내지 마라. 불확실하면 쓰지 않는다.
8. 마무리는 요약 반복이 아니라, 독자가 바로 해볼 수 있는 한 가지를 남긴다.

${BLOCK_SPEC}

아래 JSON 하나만 출력하라:
{
  "title": "실제 발행할 제목",
  "tags": ["네이버 태그", "5~8개", "# 없이"],
  "summary": "이 글이 무엇에 대한 글인지 한 문장",
  "blocks": [ ... ]
}`;

  const draft = await askJson({ prompt, system: SYSTEM, model, timeoutMs: 600_000 });

  if (!draft?.blocks?.length) throw new Error('AI가 본문 블록을 만들지 못했습니다.');

  const blocks = draft.blocks
    .filter((b) => b && b.type)
    .map((b, i) => {
      const base = { id: `b${i}`, type: b.type };
      if (b.type === 'list') return { ...base, items: (b.items || []).map(String) };
      if (b.type === 'divider') return base;
      if (b.type === 'image') {
        return {
          ...base,
          query: String(b.query || '').trim(),
          caption: String(b.caption || '').trim(),
          file: null,       // images.js가 채운다
          verdict: null,    // AI 검수 결과
        };
      }
      return { ...base, text: String(b.text || '').trim() };
    })
    .filter((b) => b.type === 'divider' || b.type === 'image' || b.text || b.items?.length);

  const charCount = blocks
    .filter((b) => b.text || b.items)
    .reduce((n, b) => n + (b.text?.length || (b.items || []).join('').length), 0);

  log.info('writer', `초안 완성: ${blocks.length}블록, 약 ${charCount}자`);

  return {
    title: String(draft.title || idea.title).trim(),
    tags: Array.isArray(draft.tags) ? draft.tags.map((t) => String(t).replace(/^#/, '').trim()) : [],
    summary: String(draft.summary || '').trim(),
    blocks,
    charCount,
  };
}
