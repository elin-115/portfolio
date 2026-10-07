import { EventEmitter } from 'node:events';

// 대시보드로 실시간 로그를 흘려보내는 단일 버스. SSE가 여기에 구독한다.
export const bus = new EventEmitter();
bus.setMaxListeners(50);

const history = [];
const MAX_HISTORY = 300;

function emit(level, scope, message, extra) {
  const entry = {
    ts: new Date().toISOString(),
    level,
    scope,
    message: String(message),
    ...(extra ? { extra } : {}),
  };
  history.push(entry);
  if (history.length > MAX_HISTORY) history.shift();
  bus.emit('log', entry);
  const tag = `[${scope}]`.padEnd(10);
  const line = `${tag} ${entry.message}`;
  if (level === 'error') console.error(line);
  else console.log(line);
  return entry;
}

export const log = {
  info: (scope, msg, extra) => emit('info', scope, msg, extra),
  warn: (scope, msg, extra) => emit('warn', scope, msg, extra),
  error: (scope, msg, extra) => emit('error', scope, msg, extra),
  step: (scope, msg, extra) => emit('step', scope, msg, extra),
  history: () => history.slice(),
};
