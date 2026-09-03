import * as tf from '@tensorflow/tfjs';

// Test basic variable creation and dispose
const init = tf.ones([3]);
const v = tf.variable(init, true, 'test');
console.log('after variable, data:', await v.data());
init.dispose();
console.log('after init.dispose, data:', await v.data());

// Test float32ToBase64 / decode using module code
const { CharTokenizer, LocalTransformerLM, MODEL_PRESETS } = await import('./src/ai-core.js');

const preset = MODEL_PRESETS.find(p => p.id === 'spark-80k');
const tok = new CharTokenizer('Hello world. This is a test.', 256);
console.log('tokenizer vocab:', tok.vocabSize, 'itos[0]:', tok.itos[0]);

const model = new LocalTransformerLM(preset, tok);
console.log('model params:', model.parameterCount());

// Train one step
const batch = { xs: tf.tensor2d([[1,2,3],[4,5,6]], [2,3], 'int32'), ys: tf.tensor2d([[2,3,4],[5,6,7]], [2,3], 'int32'), sequenceLength: 3 };
const result = await model.trainStep(batch.xs, batch.ys, { learningRate: 0.001 });
console.log('train result:', result);

// Export
const payload = await model.exportJSON();
console.log('payload format:', payload.format);
console.log('weights count:', payload.weights.length);
console.log('first weight name:', payload.weights[0]?.name);
console.log('first weight data length:', payload.weights[0]?.data?.length);
console.log('first weight data starts with:', payload.weights[0]?.data?.slice(0, 20));

// Check for null data
const nulls = payload.weights.filter(w => w.data === null || w.data === undefined);
console.log('weights with null/undefined data:', nulls.length);

// Check JSON stringify for null values inside data
const str = JSON.stringify(payload);
const nullCount = (str.match(/:null/g) || []).length;
console.log('null occurrences in JSON string:', nullCount);

// Train many steps to try to corrupt weights
for (let i = 0; i < 50; i++) {
  const result = await model.trainStep(batch.xs, batch.ys, { learningRate: 0.01 });
  if (i % 10 === 0) console.log('step', i, 'loss:', result.loss, 'skipped:', result.skipped, 'badGrad:', result.hadBadGradients);
}

// Check for NaN after training
let nanCount = 0;
for (const [name, variable] of Object.entries(model.vars)) {
  const raw = await variable.data();
  for (let i = 0; i < raw.length; i++) {
    if (!Number.isFinite(raw[i])) nanCount++;
  }
}
console.log('NaN values after training:', nanCount);

// Export after training
const payload2 = await model.exportJSON();
const nulls2 = payload2.weights.filter(w => w.data === null || w.data === undefined);
console.log('weights with null/undefined after many steps:', nulls2.length);
const str2 = JSON.stringify(payload2);
const nullCount2 = (str2.match(/:null/g) || []).length;
console.log('null occurrences in JSON after many steps:', nullCount2);

// Try generate after many steps
const gen2 = await model.generate('Hello', { maxNewTokens: 10, temperature: 0.8 });
console.log('generated after many steps:', JSON.stringify(gen2));

// Corrupt one weight manually
const corruptVar = model.vars['lmHead/w'];
const badData = new Float32Array(corruptVar.shape.reduce((a,b)=>a*b,1));
badData[0] = NaN; badData[1] = Infinity; badData[2] = -Infinity;
const badTensor = tf.tensor(badData, corruptVar.shape, 'float32');
corruptVar.assign(badTensor); badTensor.dispose();
console.log('corrupted lmHead/w NaN count:', (await corruptVar.data()).filter(x => !Number.isFinite(x)).length);

const payload3 = await model.exportJSON();
console.log('after corruption null data count:', payload3.weights.filter(w => w.data === null || w.data === undefined).length);
console.log('after corruption null strings:', (JSON.stringify(payload3).match(/:null/g) || []).length);
const lw = payload3.weights.find(w => w.name === 'lmHead/w');
console.log('lmHead/w data len:', lw?.data?.length, 'start:', lw?.data?.slice(0,20));

batch.xs.dispose();
batch.ys.dispose();
model.dispose();

// Test large array float32ToBase64
const big = new Float32Array(260000);
for (let i = 0; i < big.length; i++) big[i] = Math.sin(i) * 0.5;
console.log('big array length:', big.length, 'bytes:', big.byteLength);
function float32ToBase64(values) {
  const bytes = new Uint8Array(values.buffer, values.byteOffset, values.byteLength);
  let binary = '';
  const chunkSize = 0x8000;
  for (let offset = 0; offset < bytes.length; offset += chunkSize) {
    const chunk = bytes.subarray(offset, offset + chunkSize);
    binary += String.fromCharCode(...chunk);
  }
  return globalThis.btoa ? globalThis.btoa(binary) : Buffer.from(binary, 'binary').toString('base64');
}
function base64ToFloat32(base64) {
  const binary = globalThis.atob ? globalThis.atob(base64) : Buffer.from(base64, 'base64').toString('binary');
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) bytes[index] = binary.charCodeAt(index);
  const aligned = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
  return new Float32Array(aligned);
}
const enc = float32ToBase64(big);
console.log('encoded length:', enc.length);
const dec = base64ToFloat32(enc);
console.log('decoded length:', dec.length, 'match:', big[0] === dec[0] && big[big.length-1] === dec[big.length-1]);
