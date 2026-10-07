import path from 'node:path';
import fs from 'node:fs';
import { fileURLToPath } from 'node:url';

export const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

export const P = {
  root: ROOT,
  config: path.join(ROOT, 'config'),
  data: path.join(ROOT, 'data'),
  session: path.join(ROOT, 'data', 'session'),
  profile: path.join(ROOT, 'data', 'session', 'chrome-profile'),
  storageState: path.join(ROOT, 'data', 'session', 'naver-storage-state.json'),
  posts: path.join(ROOT, 'data', 'posts'),
  images: path.join(ROOT, 'data', 'images'),
  localImages: path.join(ROOT, 'data', 'local-images'),
  debug: path.join(ROOT, 'data', 'debug'),
  settings: path.join(ROOT, 'data', 'settings.json'),
};

export function ensureDirs() {
  for (const key of ['session', 'posts', 'images', 'localImages', 'debug']) {
    fs.mkdirSync(P[key], { recursive: true });
  }
}
