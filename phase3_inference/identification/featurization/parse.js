const path = require('path');
const fs = require('fs');
const process = require('process');
const bsor = require('./open-replay-decoder.js');
const features = require('./features.js');
const utils = require('./utils.js');

const baseDir = path.join(__dirname, '..'); // Identification-main/
const mode = process.argv[3] || 'full';

function getSamples(user, replays) { // get samples from a list of replay files
  const samples = [];
  for (const replay of replays) { // iterate over each replay file

    // only process .bsor files
    if (replay.includes('bsor')) {
      const filePath = path.join(baseDir, 'data_bsor', 'users', user, replay);
      // check if file exists
      if (!fs.existsSync(filePath)) {
        console.warn('⚠️ 파일 없음:', filePath);
        continue;
      }

      // read and decode the .bsor file
      const file = fs.readFileSync(filePath); // readFileSync returns a Buffer of the entire file

      try {
        // return value of decode is {info, frames, notes, walls}
        const data = bsor.decode(file.buffer); // in open-replay-decoder.js

        // 이 부분 출력 해보면 data 구조 알 수 있음
        // console.log(JSON.stringify(data, null, 2));
        if (data && data.frames?.length && data.notes?.length && data.info) { // valid data
          
          // process each note in the replay
          // only use notes with valid cuts and within time bounds
          // extract features before and after the note cut event
          const endTime = data.frames[data.frames.length - 1].time;

          // non-overlapping 1초 윈도우 start 후보 만들기 (정수초 단위)
          let windowStarts = [];
          for (let t = 1; t <= Math.floor(endTime - 2); t++) {
            windowStarts.push(t);
          }
          utils.shuffle(windowStarts);

          for (const note of data.notes) {
            if (note.noteCutInfo && note.noteCutInfo.speedOK && note.noteCutInfo.directionOK &&
                note.noteCutInfo.saberTypeOK && note.eventTime > 1 && note.eventTime < (endTime - 1)) {
              let sample = [user];
              
              if (mode === 'full') sample = sample.concat(features.makeNoteFeatures(note));
              // features from 1 second before and after the note cut event
                  
              // 겹치지 않는 1초 윈도우 start 하나를 가져옴
              if (windowStarts.length === 0) continue;   // 더 이상 윈도우 없으면 샘플 스킵
              const winStart = windowStarts.pop();

              
              const framesBefore = utils.fastTimeSlice(data.frames, winStart, winStart + 1);

              // const framesBefore = utils.fastTimeSlice(data.frames, note.eventTime - 1, note.eventTime);
              sample = sample.concat(features.makeMotionFeatures(framesBefore));
              // const framesAfter = utils.fastTimeSlice(data.frames, note.eventTime, note.eventTime + 1);
              // sample = sample.concat(features.makeMotionFeatures(framesBefore));

              if (!sample.includes(null) && !sample.includes(undefined) && !sample.includes(NaN)) {
                samples.push(sample);
              }
            }
          }
        }
      } catch (err) { console.error(err); }
    }
  }
  return samples;
}

const id = parseInt(process.argv[2]); // node number
// SESSION.JSON 필요함!!!
const sessions = JSON.parse(fs.readFileSync(path.join(baseDir, 'data', 'sessions.json'), 'utf8')); // read session data

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


// const users = Object.keys(sessions); // get user list


// handle a single user
function handleUser(user, set, count) { // set: train, validate, cluster, test
  const samples = getSamples(user, sessions[user][set]); // get user's samples in the set
  utils.shuffle(samples);
  const selected = samples.slice(0, count);

  if (samples.length === 0) {
    console.warn('⚠️ 샘플 없음:', user, set);
    return; // no samples
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

for (let i = 0; i < Math.min(500, users.length); i++) { // limit to 500 users
  if (i % 32 == id) { // distribute work across 32 nodes
    const user = users[i];
    
    handleUser(user, 'train', 150); // max 150 samples
    handleUser(user, 'validate', 5); // max 5 samples
    handleUser(user, 'cluster', 50); // max 50 samples
    handleUser(user, 'test', 50); // max 50 samples
    console.log(i / users.length); // progress
  }
}

const t1 = performance.now();
const time = t1 - t0;
console.log('Featurization finished in time: ', time);
fs.writeFileSync(path.join(baseDir, 'stats', 'featurization', `${id}.txt`), time.toString());
