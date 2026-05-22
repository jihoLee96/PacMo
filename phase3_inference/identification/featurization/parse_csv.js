const path = require('path');
const fs = require('fs');
const process = require('process');

const features = require('./features.js');
const utils = require('./utils.js');

const baseDir = path.join(__dirname, '..'); // Identification-main/
const mode = process.argv[3] || 'full';

// ✅ decoded csv root
const CSV_ROOT = path.join(baseDir, 'gt_recon_csv_out', 'users');

// ---------------------------
// CSV -> frames(bsor-style) 변환
// ---------------------------
function toNum(x) {
  if (x === undefined || x === null || x === '') return null;
  const v = Number(x);
  return Number.isFinite(v) ? v : null;
}

function csvRowToFrame(row) {
  // row values
  const hx = toNum(row.hmd_x), hy = toNum(row.hmd_y), hz = toNum(row.hmd_z);
  const hqx = toNum(row.hmd_qx), hqy = toNum(row.hmd_qy), hqz = toNum(row.hmd_qz), hqw = toNum(row.hmd_qw);

  const lx = toNum(row.left_x), ly = toNum(row.left_y), lz = toNum(row.left_z);
  const lqx = toNum(row.left_qx), lqy = toNum(row.left_qy), lqz = toNum(row.left_qz), lqw = toNum(row.left_qw);

  const rx = toNum(row.right_x), ry = toNum(row.right_y), rz = toNum(row.right_z);
  const rqx = toNum(row.right_qx), rqy = toNum(row.right_qy), rqz = toNum(row.right_qz), rqw = toNum(row.right_qw);

  const time = toNum(row.time);
  const fps = toNum(row.fps) ?? 0;

  // 필수값 누락 시 frame drop
  if (time === null) return null;
  if ([hx,hy,hz,hqx,hqy,hqz,hqw,lx,ly,lz,lqy,lqz,lqw,rx,ry,rz,rqy,rqz,rqw].some(v => v === null)) {
    // left_qx/right_qx는 빈칸이어도 될 수 있으니 제외
    // (원하면 아래에서 qx까지 필수화 가능)
  }

  return {
    time,
    fps,
    h: { p: { x: hx, y: hy, z: hz }, r: { x: hqx, y: hqy, z: hqz, w: hqw } },
    l: { p: { x: lx, y: ly, z: lz }, r: { x: lqx ?? 0, y: lqy, z: lqz, w: lqw } },
    r: { p: { x: rx, y: ry, z: rz }, r: { x: rqx ?? 0, y: rqy, z: rqz, w: rqw } },
  };
}

function loadFramesFromCsv(csvPath) {
  if (!fs.existsSync(csvPath)) return null;

  const text = fs.readFileSync(csvPath, 'utf8');
  const lines = text.split(/\r?\n/).filter(Boolean);
  if (lines.length < 2) return null;

  const header = lines[0].split(',');
  const frames = [];

  for (let i = 1; i < lines.length; i++) {
    const cols = lines[i].split(',');
    const row = {};
    for (let j = 0; j < header.length; j++) row[header[j]] = cols[j];

    const frame = csvRowToFrame(row);
    if (frame) frames.push(frame);
  }

  // time 정렬 보장
  frames.sort((a, b) => a.time - b.time);

  return frames.length > 0 ? frames : null;
}

// ---------------------------
// 샘플 추출 (bsor.getSamples 대체)
// ---------------------------
function getSamplesFromCsv(user, replays) {
  const samples = [];

  for (const replay of replays) {
    // sessions.json에는 *.bsor로 들어있음 → *.csv로 치환
    if (!replay.includes('bsor')) continue;

    const csvName = replay.replace(/\.bsor(temp)?$/i, '.csv');
    const csvPath = path.join(CSV_ROOT, user, csvName);

    const frames = loadFramesFromCsv(csvPath);
    if (!frames) {
      console.warn('⚠️ CSV frames 없음:', csvPath);
      continue;
    }

    // 원래 코드 로직: replay endTime 이용해서 1초 window start 후보 랜덤
    const endTime = frames[frames.length - 1].time;
    let windowStarts = [];
    for (let t = 1; t <= Math.floor(endTime - 2); t++) windowStarts.push(t);
    utils.shuffle(windowStarts);

    // NOTE features는 이제 없으니까 (bsor noteCutInfo 없음)
    // 대신 window start마다 1개 sample 생성
    while (windowStarts.length > 0) {
      const winStart = windowStarts.pop();
      const framesBefore = utils.fastTimeSlice(frames, winStart, winStart + 1);

      if (!framesBefore || framesBefore.length === 0) continue;

      let sample = [user];
      // mode full이면 원래는 note features가 붙는데, CSV에 note 없음 → 생략
      sample = sample.concat(features.makeMotionFeatures(framesBefore));

      if (!sample.includes(null) && !sample.includes(undefined) && !sample.includes(NaN)) {
        samples.push(sample);
      }
    }
  }

  return samples;
}

// ---------------------------
// MAIN (parse.js 흐름 거의 그대로)
// ---------------------------
const id = parseInt(process.argv[2]); // node number
const sessions = JSON.parse(fs.readFileSync(path.join(baseDir, 'data', 'sessions.json'), 'utf8')); // same sessions.json

const EXCLUDE_USERS = new Set([
  "0099f7dd-8e9e-4726-8959-2868e2e9460c",
  "0054909e-36b3-4595-b4f5-f357778b12e9",
  "00c3c170-4d87-4cec-9939-99d3b4d10b26",
  "00e95b89-2b88-4c29-974d-cf9e1934ec5b",
  "00779c2d-b2df-45b1-995d-c453270464b5",
  "01607512-5d45-40e8-bfb0-718ab719ff83",
  "011c3958-fb8f-463d-8087-4adee42f1ac8",
  "014f9ee8-8c6a-4359-8242-7e54641d63e1",
  "0192c163-54d8-43a8-89e6-89a96e0a970b",
  "01d96786-8615-4f6f-a896-03e3b156f756",
  "01d20c1d-6476-40d8-81c6-ea84d9558eca",
  "01ea714e-154d-4d23-930c-b7800b5ec54f",
  "02140320-625c-4d93-8006-9171302256b7",
  "025d789e-09e7-4cdf-89e9-ff95f627e18c",
]);

const users = Object.keys(sessions).filter(u => !EXCLUDE_USERS.has(u));

function handleUser(user, set, count) {
  const samples = getSamplesFromCsv(user, sessions[user][set]);
  utils.shuffle(samples);
  const selected = samples.slice(0, count);

  if (samples.length === 0) {
    console.warn('⚠️ 샘플 없음:', user, set);
    return;
  } else if (samples.length < count) {
    console.warn('⚠️ 샘플 부족:', user, set, samples.length, '<', count);
  }

  const dirPath = path.join(baseDir, 'data', set);
  fs.mkdirSync(dirPath, { recursive: true });

  fs.writeFileSync(
    path.join(dirPath, `${user}.csv`),
    selected.map(r => r.join(',')).join('\n') + '\n'
  );
}

const { performance } = require('perf_hooks');
const t0 = performance.now();

for (let i = 0; i < Math.min(500, users.length); i++) {
  if (i % 32 == id) {
    const user = users[i];

    handleUser(user, 'train', 150);
    handleUser(user, 'validate', 5);
    handleUser(user, 'cluster', 50);
    handleUser(user, 'test', 50);
    console.log(i / users.length);
  }
}

const t1 = performance.now();
const time = t1 - t0;
console.log('Featurization finished in time: ', time);
fs.writeFileSync(path.join(baseDir, 'stats', 'featurization', `${id}.txt`), time.toString());
