import { askJson } from './ai.js';
import { getPost, savePost, getSettings } from './store.js';
import { getLastRun } from './pipeline.js';
import { log } from './log.js';

/**
 * 이미 쓴 초안에 섹션을 추가한다.
 *
 * 핵심은 "수집한 원문을 근거로만 쓰게 하는 것"이다.
 * 등산 소요시간이나 가격처럼 틀리면 실제로 곤란해지는 숫자를 AI가 지어내면 안 되기 때문에,
 * 근거에 없는 값은 채우지 말고 비우도록 프롬프트에 못을 박는다.
 */

const SYSTEM = `당신은 블로그 글을 다듬는 편집자다.
주어진 근거 자료에 실제로 나오는 사실만 쓴다.
시간·거리·가격·날짜처럼 확인 가능한 숫자는 근거에 있는 것만 사용하고,
없으면 그 부분을 비워둔다. 그럴듯하게 지어내는 것이 가장 나쁜 결과다.`;

/** 초안 본문을 프롬프트용 텍스트로 편다. */
function flatten(blocks) {
  return blocks.map((b, i) => {
    if (b.type === 'heading') return `[${i}] ## ${b.text}`;
    if (b.type === 'list') return `[${i}] - ${(b.items || []).join(' / ')}`;
    if (b.type === 'image') return `[${i}] (이미지 자리)`;
    if (b.type === 'divider') return `[${i}] ---`;
    return `[${i}] ${b.text || ''}`;
  }).join('\n');
}

/**
 * @param {string} postId
 * @param {string} instruction  사용자가 쓴 지시 (예: "각 지점 구간별 타임라인 넣어줘")
 */
export async function addSection(postId, instruction) {
  if (!instruction?.trim()) throw new Error('어떤 섹션을 추가할지 지시를 입력하세요.');

  const s = getSettings();
  const post = getPost(postId);
  const { sources } = getLastRun();

  const withBody = (sources || []).filter((x) => (x.body || '').length > 300);
  const evidence = withBody.length
    ? withBody.map((x) => `[${x.idx}] ${x.title}\n${x.body}`).join('\n\n---\n\n')
    : (post.sources || []).map((x) => `[${x.idx}] ${x.title}`).join('\n');

  if (!withBody.length) {
    log.warn('section', '근거 원문이 없습니다. 글감 발굴을 다시 실행하면 더 정확해집니다.');
  }

  const notesBlock = (s.notes || '').trim() ? `
## ⚠️ 블로그 주인이 알려준 사실 — 최우선
**수집 자료와 충돌하면 무조건 이쪽이 옳다.**

${s.notes.trim()}
` : '';

  log.step('section', `"${instruction}" — 근거 ${withBody.length}건으로 섹션을 만드는 중…`);

  const result = await askJson({
    model: s.model,
    timeoutMs: 600_000,
    system: SYSTEM,
    prompt: `아래는 이미 작성된 블로그 글과, 그 글의 근거가 된 원문 자료다.
${notesBlock}
## 현재 글
제목: ${post.draft.title}

${flatten(post.draft.blocks)}

## 근거 자료
${evidence}

---
## 요청
${instruction}

## 반드시 지킬 것
1. **근거 자료에 실제로 나오는 정보만 쓴다.** 시간·거리·가격·날짜는 근거에 있는 값만 사용한다.
2. 자료마다 값이 다르면 범위로 적는다. (예: "약 1시간 20분~1시간 50분")
3. 근거에서 확인되지 않는 항목은 **그냥 빼라.** 추정치로 채우지 마라.
4. 문장은 새로 쓴다. 원문 표현을 그대로 옮기지 않는다.
5. 기존 글의 톤(${s.tone})을 유지하고, 이미 쓴 내용을 반복하지 않는다.
6. 이 섹션이 들어갈 위치를 기존 글의 블록 번호로 지정한다. (그 번호 **앞에** 삽입된다)

아래 JSON 하나만 출력하라:
{
  "insertBefore": 삽입할 위치의 블록 번호 (맨 뒤면 ${post.draft.blocks.length}),
  "usedSources": [근거로 삼은 자료 번호들],
  "blocks": [
    {"type":"heading","text":"..."},
    {"type":"paragraph","text":"..."},
    {"type":"list","items":["...","..."]},
    {"type":"quote","text":"..."}
  ]
}`,
  });

  const fresh = (result.blocks || [])
    .filter((b) => b && b.type)
    .map((b, i) => {
      const base = { id: `s${Date.now().toString(36)}-${i}`, type: b.type };
      if (b.type === 'list') return { ...base, items: (b.items || []).map(String) };
      if (b.type === 'divider') return base;
      return { ...base, text: String(b.text || '').trim() };
    })
    .filter((b) => b.type === 'divider' || b.items?.length || b.text);

  if (!fresh.length) throw new Error('AI가 섹션을 만들지 못했습니다.');

  const blocks = post.draft.blocks;
  let at = Number(result.insertBefore);
  if (!Number.isInteger(at) || at < 0 || at > blocks.length) at = blocks.length;

  post.draft.blocks = [...blocks.slice(0, at), ...fresh, ...blocks.slice(at)];
  post.draft.charCount = post.draft.blocks
    .reduce((n, b) => n + (b.text?.length || (b.items || []).join('').length), 0);

  savePost(post);
  log.info('section', `섹션 ${fresh.length}블록을 ${at}번 위치에 추가했습니다. (총 ${post.draft.charCount}자)`);
  return post;
}
