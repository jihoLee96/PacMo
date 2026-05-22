// IEEE754 float64 → float32 압축 시뮬레이션
function compressToFloat32(value) {
  // 1. Float64를 Float32로 변환 (손실 발생)
  const f32 = new Float32Array(1);
  f32[0] = value;

  // 2. Float32를 다시 Float64로 복원
  const back = f32[0];

  return back;
}

// ===== Compression modes =====
// 'float32'  : JS Number(64-bit) → Float32로 캐스팅(일반적인 네트워킹: Unity/Mirror에서 float 전송)
// 'quant16'  : Mirror의 PackedVector 유사: 지정 범위를 16bit 정수로 양자화했다가 복원
// 'half'     : (옵션) 16-bit half float 근사. 필요 없으면 안 써도 됨.
const COMPRESSION = process.env.BSOR_COMPRESSION || 'quant16';

// quant16에 사용할 범위 (Mirror의 PackedVector처럼 "알고 있는 범위"를 지정)
const Q16_MIN = -100.0;   // 필요에 맞게 조정
const Q16_MAX =  100.0;   // 필요에 맞게 조정

function toFloat32(v) {
  const f32 = new Float32Array(1);
  f32[0] = v;
  return f32[0]; // 64→32→64 과정으로 실제 IEEE754 정밀도 손실 재현
}

// Mirror PackedVector 유사: [min,max]→0..65535 양자화 후 복원
function quantize16(v, min, max) {
  const clamped = Math.min(max, Math.max(min, v));
  const t = (clamped - min) / (max - min);      // 0..1
  return Math.round(t * 65535);                 // 0..65535
}
function dequantize16(q, min, max) {
  const t = q / 65535;                          // 0..1
  return t * (max - min) + min;                 // 복원
}

function compressValue(v) {
  switch (COMPRESSION) {
    case 'float32':
      return toFloat32(v);
    case 'quant16': {
      const q = quantize16(v, Q16_MIN, Q16_MAX);
      return dequantize16(q, Q16_MIN, Q16_MAX);
    }
    case 'half': {
      // 간단 근사: float32→(10비트 유효숫자 수준으로) 잘라서 half 비슷하게
      // 정확한 IEEE754 half 변환이 필요하면 별도 라이브러리/루틴 쓰는 걸 추천.
      // 여기서는 네트워크 전송시 '대략 half 정도의 손실'만 재현.
      const f32 = toFloat32(v);
      // mantissa를 대략 10비트 수준으로 줄이기 (2^10 ~= 1024)
      const scale = Math.pow(2, 10);
      return Math.round(f32 * scale) / scale;
    }
    default:
      return v;
  }
}


function makeMotionFeature(frames, object, aspect, coord) {
  let values = frames.map(frame => frame[object][aspect][coord]);

  // 반올림하여 정밀도 제한 - float 소수점 6자리로 자르기
  // values = values.map(v => parseFloat(v.toFixed(6)));
  // values = values.map(v => compressToFloat32(v));
  values = values.map(compressValue);

  values.sort();
  const min = values[0];
  const max = values[values.length - 1];

  const half = Math.floor(values.length / 2);
  const med = (values.length % 2) ? values[half] : ((values[half - 1] + values[half]) / 2.0);

  let sum = 0;
  for (let i = 0; i < values.length; i++) {
    sum += values[i];
  }
  const mean = sum / values.length;

  let squares = 0;
  for (let i = 0; i < values.length; i++) {
    squares += ((values[i] - mean) ** 2);
  }
  const std = Math.sqrt(squares / values.length);

  return [min, max, mean, med, std];
}

function makeMotionFeatures(frames) {
  var features = [];
  for (let object of ['h', 'l', 'r']) {
    for (let aspect of ['p', 'r']) {
      for (let coord of ['x', 'y', 'z']) {
        features = features.concat(makeMotionFeature(frames, object, aspect, coord));
      }
    }
    features = features.concat(makeMotionFeature(frames, object, 'r', 'w'));
  }

  return features;
}

function coords(feature) {
  return [feature.x, feature.y, feature.z];
}

function makeNoteFeatures(note) { // extract note features
  let note_id = note.noteID
  let x = note_id
  let cutDirection = (x % 10)
  x = (x - cutDirection) / 10
  let colorType = (x % 10)
  x = (x - colorType) / 10
  let noteLineLayer = (x % 10)
  x = (x - noteLineLayer) / 10
  let lineIndex = (x % 10)
  x = (x - lineIndex) / 10
  let scoringType = (x % 10)
  var feature = ['N' + note_id, cutDirection, colorType, noteLineLayer, lineIndex, scoringType];
  let cut = note.noteCutInfo
  feature = feature.concat([cut.saberType, cut.saberSpeed, cut.timeDeviation, cut.cutDirDeviation, cut.cutDistanceToCenter, cut.cutAngle, cut.beforeCutRating, cut.afterCutRating])
  feature = feature.concat(coords(cut.saberDir))
  feature = feature.concat(coords(cut.cutPoint))
  feature = feature.concat(coords(cut.cutNormal))
  return feature;
}

module.exports.makeMotionFeatures = makeMotionFeatures;
module.exports.makeNoteFeatures = makeNoteFeatures;
