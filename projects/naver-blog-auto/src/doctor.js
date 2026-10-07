import fs from 'node:fs';
import path from 'node:path';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { chromium } from 'playwright';
import { P } from './paths.js';
import { getSettings } from './store.js';
import { checkAi } from './ai.js';
import { checkSession } from './auth.js';

try { process.loadEnvFile(path.join(P.root, '.env')); } catch { /* 없으면 그만 */ }

const execFileAsync = promisify(execFile);
const results = [];

function report(name, ok, detail, fix) {
  results.push({ name, ok, detail, fix });
  const icon = ok === true ? '✅' : ok === 'warn' ? '⚠️ ' : '❌';
  console.log(`${icon} ${name}`);
  if (detail) console.log(`     ${detail}`);
  if (!ok || ok === 'warn') { if (fix) console.log(`     → ${fix}`); }
}

async function main() {
  console.log('\n네이버 블로그 자동화 — 환경 점검\n' + '─'.repeat(46));

  // 1. Node
  const major = Number(process.versions.node.split('.')[0]);
  report('Node.js', major >= 20, `v${process.versions.node}`,
    'Node 20 이상이 필요합니다. https://nodejs.org 에서 설치하세요.');

  // 2. claude CLI 존재
  let cliOk = false;
  try {
    const { stdout } = await execFileAsync('claude', ['--version'], { timeout: 15_000 });
    cliOk = true;
    report('claude CLI', true, stdout.trim());
  } catch {
    report('claude CLI', false, '실행 파일을 찾지 못했습니다.',
      'Claude Code를 설치하고 PATH에 claude가 잡히는지 확인하세요.');
  }

  // 3. claude 로그인 (실제 호출)
  if (cliOk) {
    process.stdout.write('   AI 응답을 확인하는 중…\r');
    try {
      const ai = await checkAi();
      report('claude 로그인 (구독 사용)', ai.ok,
        `응답 ${(ai.ms / 1000).toFixed(1)}초 · 모델 ${ai.model}`,
        '터미널에서 `claude`를 실행한 뒤 /login 으로 로그인하세요.');
    } catch (e) {
      report('claude 로그인 (구독 사용)', false, e.message,
        '터미널에서 `claude`를 실행한 뒤 /login 으로 로그인하세요.');
    }
  }

  // 4. 크로미움
  let exe = '';
  try { exe = chromium.executablePath(); } catch { /* 무시 */ }
  report('Playwright 크로미움', Boolean(exe && fs.existsSync(exe)),
    exe ? path.basename(exe) : '미설치',
    'npm run install-browser');

  // 5. 네이버 세션
  if (fs.existsSync(P.profile)) {
    process.stdout.write('   네이버 세션을 확인하는 중…\r');
    const s = await checkSession();
    report('네이버 로그인 세션', s.loggedIn,
      s.loggedIn ? (s.id ? `계정: ${s.id}` : '유효함') : s.reason,
      '대시보드에서 [네이버 로그인] 칩을 눌러 다시 로그인하세요.');
  } else {
    report('네이버 로그인 세션', false, '아직 로그인한 적이 없습니다.',
      '대시보드에서 [네이버 로그인] 칩을 누르세요.');
  }

  // 6. 설정
  const s = getSettings();
  report('블로그 아이디', Boolean(s.blogId), s.blogId || '미설정',
    '대시보드 [설정]에서 blog.naver.com/<아이디>의 아이디를 넣으세요. 없으면 발행이 안 됩니다.');
  report('관심분야', s.topics?.length > 0, (s.topics || []).join(', ') || '없음',
    '대시보드 상단 입력창에 관심분야를 넣으세요.');

  // 7. 스톡 API 키 (선택)
  const keys = [
    process.env.UNSPLASH_ACCESS_KEY && 'Unsplash',
    process.env.PEXELS_API_KEY && 'Pexels',
  ].filter(Boolean);
  report('스톡 이미지 API 키 (선택)', keys.length ? true : 'warn',
    keys.length ? keys.join(', ') : '없음 — 네이버 이미지/로컬 폴더만 사용합니다',
    '.env.example을 .env로 복사하고 무료 키를 넣으면 저작권 안전한 사진을 씁니다.');

  const bad = results.filter((r) => r.ok === false);
  console.log('─'.repeat(46));
  console.log(bad.length
    ? `\n${bad.length}개 항목을 먼저 해결하세요: ${bad.map((r) => r.name).join(', ')}\n`
    : '\n전부 정상입니다. `npm start` 로 대시보드를 여세요.\n');

  process.exit(bad.length ? 1 : 0);
}

main().catch((e) => { console.error(e); process.exit(1); });
