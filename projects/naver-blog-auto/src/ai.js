import { spawn } from 'node:child_process';
import { log } from './log.js';

/**
 * Claude Code CLI를 `-p`(비대화형)로 호출하는 래퍼.
 * API 키 종량과금이 아니라 CLI에 로그인된 구독 계정을 그대로 사용한다.
 */

// 부모가 Claude Code 세션이면 호스트 전용 인증 환경변수가 상속되어
// 자식 `claude`가 "Not logged in"으로 죽는다. 그래서 걷어내고 실행한다.
function cleanEnv() {
  const env = { ...process.env };
  const strip = [
    'ANTHROPIC_BASE_URL', 'CLAUDECODE', 'CLAUDE_CODE_ENTRYPOINT',
    'CLAUDE_CODE_SESSION_ID', 'CLAUDE_CODE_HOST_SESSION_ID',
    'CLAUDE_CODE_CHILD_SESSION', 'CLAUDE_CODE_SDK_HAS_OAUTH_REFRESH',
    'CLAUDE_CODE_SDK_HAS_HOST_AUTH_REFRESH', 'CLAUDE_CODE_MESSAGING_SOCKET',
    'CLAUDE_CODE_MESSAGING_TOKEN', 'CLAUDE_CODE_OAUTH_SCOPES',
    'CLAUDE_AGENT_SDK_VERSION', 'CLAUDE_CODE_EXECPATH', 'CLAUDE_PID',
  ];
  for (const k of strip) delete env[k];
  return env;
}

export class AiError extends Error {
  constructor(message, detail) {
    super(message);
    this.name = 'AiError';
    this.detail = detail;
  }
}

/**
 * @param {object} opts
 * @param {string} opts.prompt        stdin으로 넘길 본 프롬프트
 * @param {string} [opts.system]      역할 지시 (--append-system-prompt)
 * @param {string[]} [opts.allowTools] 허용 툴. 기본은 빈 배열 = 툴 없이 순수 생성
 * @param {string} [opts.model]       'sonnet' | 'opus' | 'haiku' 등
 * @param {number} [opts.timeoutMs]
 * @param {string} [opts.cwd]
 */
export function runClaude({
  prompt,
  system,
  allowTools = [],
  model = 'sonnet',
  timeoutMs = 300_000,
  cwd = process.cwd(),
} = {}) {
  const args = ['-p', '--output-format', 'json', '--model', model];
  if (system) args.push('--append-system-prompt', system);
  // 허용 툴을 명시하면 비대화형에서도 권한 프롬프트 없이 통과한다.
  args.push('--allowedTools', allowTools.join(','));

  return new Promise((resolve, reject) => {
    let child;
    try {
      child = spawn('claude', args, { cwd, env: cleanEnv(), stdio: ['pipe', 'pipe', 'pipe'] });
    } catch (err) {
      return reject(new AiError('claude CLI를 실행할 수 없습니다. 설치 여부를 확인하세요.', err.message));
    }

    let out = '';
    let err = '';
    let settled = false;

    const timer = setTimeout(() => {
      if (settled) return;
      settled = true;
      child.kill('SIGKILL');
      reject(new AiError(`AI 응답이 ${Math.round(timeoutMs / 1000)}초를 넘겨 중단했습니다.`, err.slice(-500)));
    }, timeoutMs);

    child.stdout.on('data', (d) => { out += d; });
    child.stderr.on('data', (d) => { err += d; });

    child.on('error', (e) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      reject(new AiError('claude CLI 실행 실패 (PATH에 claude가 있는지 확인하세요).', e.message));
    });

    child.on('close', (code) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);

      let payload = null;
      try { payload = JSON.parse(out); } catch { /* 아래에서 처리 */ }

      if (!payload) {
        return reject(new AiError(
          `claude CLI가 예상치 못한 출력을 냈습니다 (exit ${code}).`,
          (err || out).slice(-800),
        ));
      }
      const text = typeof payload.result === 'string' ? payload.result : '';
      if (payload.is_error) {
        if (/not logged in/i.test(text)) {
          return reject(new AiError(
            'Claude CLI에 로그인되어 있지 않습니다. 터미널에서 `claude` 실행 후 /login 하세요.',
            text,
          ));
        }
        return reject(new AiError(`AI 호출 실패: ${text || 'unknown'}`, err.slice(-500)));
      }
      resolve({ text, raw: payload });
    });

    child.stdin.end(prompt, 'utf8');
  });
}

/** 응답에서 첫 번째 JSON 객체/배열을 뽑아낸다 (코드펜스·설명문 섞여 있어도 동작). */
export function extractJson(text) {
  if (!text) throw new AiError('AI가 빈 응답을 반환했습니다.');
  const fenced = text.match(/```(?:json)?\s*([\s\S]*?)```/i);
  const body = fenced ? fenced[1] : text;

  const start = body.search(/[[{]/);
  if (start === -1) throw new AiError('AI 응답에서 JSON을 찾지 못했습니다.', text.slice(0, 500));

  const open = body[start];
  const close = open === '{' ? '}' : ']';
  let depth = 0;
  let inStr = false;
  let esc = false;
  for (let i = start; i < body.length; i++) {
    const ch = body[i];
    if (inStr) {
      if (esc) esc = false;
      else if (ch === '\\') esc = true;
      else if (ch === '"') inStr = false;
      continue;
    }
    if (ch === '"') inStr = true;
    else if (ch === open) depth++;
    else if (ch === close) {
      depth--;
      if (depth === 0) {
        const slice = body.slice(start, i + 1);
        try { return JSON.parse(slice); }
        catch (e) { throw new AiError('AI가 만든 JSON 파싱에 실패했습니다.', slice.slice(0, 500)); }
      }
    }
  }
  throw new AiError('AI 응답의 JSON이 중간에 끊겼습니다.', text.slice(-500));
}

/** JSON 응답을 기대하는 호출. 1회 재시도한다. */
export async function askJson(opts) {
  const system = [
    opts.system || '',
    '반드시 유효한 JSON만 출력한다. 설명·인사·코드펜스 없이 JSON 하나만 낸다.',
  ].filter(Boolean).join('\n');

  let lastErr;
  for (let attempt = 1; attempt <= 2; attempt++) {
    try {
      const { text } = await runClaude({ ...opts, system });
      return extractJson(text);
    } catch (e) {
      lastErr = e;
      if (e instanceof AiError && /로그인|실행할 수 없|실행 실패/.test(e.message)) throw e;
      log.warn('ai', `JSON 응답 실패(${attempt}/2): ${e.message}`);
    }
  }
  throw lastErr;
}

/** 대시보드 "AI 연결 확인" 버튼용 헬스체크. */
export async function checkAi() {
  const started = Date.now();
  const { text, raw } = await runClaude({
    prompt: 'ok 라고만 답해. 다른 말 하지마.',
    model: 'sonnet',
    timeoutMs: 90_000,
  });
  return {
    ok: /ok/i.test(text),
    reply: text.trim().slice(0, 80),
    ms: Date.now() - started,
    model: Object.keys(raw.modelUsage || {})[0] || 'unknown',
  };
}
