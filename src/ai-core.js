import * as tf from '@tensorflow/tfjs';

export const ESTIMATE_VOCAB_SIZE = 256;
export const EXPORT_FORMAT = 'v0idgpt-local-transformer-v2';
const MAX_ABS_WEIGHT = 30;
const REPAIR_YIELD_INTERVAL = 8;

export const MODEL_PRESETS = [
  {
    id: '1param-joke',
    name: '1 Parameter',
    tagline: 'A single trainable number. Good luck.',
    dModel: 1,
    nLayers: 1,
    nHeads: 1,
    ffDim: 1,
    context: 2,
    defaultBatch: 1,
    defaultLr: 0.1,
    risk: 'safe',
  },
  {
    id: 'spark-80k',
    name: 'Spark 80K',
    tagline: 'Tiny local transformer for instant experiments.',
    dModel: 48,
    nLayers: 2,
    nHeads: 4,
    ffDim: 192,
    context: 64,
    defaultBatch: 16,
    defaultLr: 0.0015,
    risk: 'safe',
  },
  {
    id: 'pulse-250k',
    name: 'Pulse 250K',
    tagline: 'Still quick, noticeably better pattern capacity.',
    dModel: 88,
    nLayers: 2,
    nHeads: 4,
    ffDim: 352,
    context: 96,
    defaultBatch: 12,
    defaultLr: 0.0012,
    risk: 'safe',
  },
  {
    id: 'ember-500k',
    name: 'Ember 500K',
    tagline: 'Recommended browser starter model.',
    dModel: 128,
    nLayers: 2,
    nHeads: 4,
    ffDim: 512,
    context: 128,
    defaultBatch: 10,
    defaultLr: 0.001,
    risk: 'safe',
  },
  {
    id: 'nova-1m',
    name: 'Nova 1M',
    tagline: 'Small GPT-style decoder with real attention depth.',
    dModel: 160,
    nLayers: 3,
    nHeads: 5,
    ffDim: 640,
    context: 160,
    defaultBatch: 8,
    defaultLr: 0.0008,
    risk: 'safe',
  },
  {
    id: 'vector-5m',
    name: 'Vector 5M',
    tagline: 'Heavier local training; use WebGL/GPU.',
    dModel: 256,
    nLayers: 6,
    nHeads: 8,
    ffDim: 1024,
    context: 256,
    defaultBatch: 4,
    defaultLr: 0.0006,
    risk: 'caution',
  },
  {
    id: 'apex-20m',
    name: 'Apex 20M',
    tagline: 'Big for a browser tab. Smaller batches only.',
    dModel: 512,
    nLayers: 6,
    nHeads: 8,
    ffDim: 2048,
    context: 384,
    defaultBatch: 1,
    defaultLr: 0.00035,
    risk: 'warning',
  },
  {
    id: 'titan-75m',
    name: 'Titan 75M',
    tagline: 'Serious hardware needed. Can lock up normal laptops.',
    dModel: 768,
    nLayers: 10,
    nHeads: 12,
    ffDim: 3072,
    context: 512,
    defaultBatch: 1,
    defaultLr: 0.00025,
    risk: 'danger',
  },
  {
    id: 'colossus-180m',
    name: 'Colossus 180M',
    tagline: 'Workstation/large VRAM territory.',
    dModel: 1024,
    nLayers: 14,
    nHeads: 16,
    ffDim: 4096,
    context: 768,
    defaultBatch: 1,
    defaultLr: 0.00018,
    risk: 'extreme',
  },
  {
    id: 'leviathan-500m',
    name: 'Leviathan 500M',
    tagline: 'Experimental monster config. Expect crashes without huge RAM.',
    dModel: 2048,
    nLayers: 10,
    nHeads: 16,
    ffDim: 8192,
    context: 1024,
    defaultBatch: 1,
    defaultLr: 0.00012,
    risk: 'extreme',
  },
  {
    id: 'wtf-1.76q',
    name: 'Fuckfuckfuck 1.76 Quadrillion',
    tagline: 'Not a real trainable model.',
    dModel: 65536,
    nLayers: 34,
    nHeads: 128,
    ffDim: 262144,
    context: 2048,
    defaultBatch: 1,
    defaultLr: 0.001,
    risk: 'meltdown',
  },
  {
    id: 'singularity-1b',
    name: 'Singularity 1B',
    tagline: 'The red button: billion-parameter decoder architecture.',
    dModel: 4096,
    nLayers: 5,
    nHeads: 32,
    ffDim: 16384,
    context: 2048,
    defaultBatch: 1,
    defaultLr: 0.00008,
    risk: 'meltdown',
  },
].map((preset) => ({
  ...preset,
  estimatedParams: estimateParams(preset, ESTIMATE_VOCAB_SIZE),
}));

const BASE_CHARS = [
  '\n', '\t', ' ',
  ...'abcdefghijklmnopqrstuvwxyz',
  ...'ABCDEFGHIJKLMNOPQRSTUVWXYZ',
  ...'0123456789',
  ...'.,!?;:\'"`~@#$%^&*()-_=+[]{}<>/\\|',
  '“', '”', '‘', '’', '—', '–', '…', '•', '🙂', '🔥', '⚠', '️',
];

export class CharTokenizer {
  constructor(text = '', maxVocab = ESTIMATE_VOCAB_SIZE) {
    // Hybrid char/subword tokenizer: every character remains representable, then
    // frequent corpus pieces are added and encoded with longest-match greediness.
    const tokens = new Set(BASE_CHARS);
    for (const ch of Array.from(text)) tokens.add(ch);
    tokens.add('�');
    for (const piece of learnFrequentPieces(text, maxVocab, tokens)) {
      if (tokens.size >= maxVocab) break;
      tokens.add(piece);
    }
    this.unk = '�';
    this.itos = Array.from(tokens);
    this.stoi = Object.fromEntries(this.itos.map((ch, index) => [ch, index]));
    this.vocabSize = this.itos.length;
    this.maxTokenLength = Math.max(1, ...this.itos.map((token) => Array.from(token).length));
  }

  encode(text) {
    const units = Array.from(text);
    const unkId = this.stoi[this.unk];
    const ids = [];
    for (let position = 0; position < units.length;) {
      let matched = null;
      const maxLength = Math.min(this.maxTokenLength, units.length - position);
      for (let length = maxLength; length >= 1; length -= 1) {
        const piece = units.slice(position, position + length).join('');
        if (this.stoi[piece] !== undefined) {
          matched = piece;
          break;
        }
      }
      if (matched === null) {
        ids.push(unkId);
        position += 1;
      } else {
        ids.push(this.stoi[matched]);
        position += Array.from(matched).length;
      }
    }
    return ids;
  }

  decode(ids) {
    return ids.map((id) => this.itos[id] ?? '').join('').replaceAll(this.unk, '');
  }

  toJSON() {
    return { itos: this.itos, unk: this.unk, maxTokenLength: this.maxTokenLength };
  }

  static fromJSON(json) {
    const tokenizer = Object.create(CharTokenizer.prototype);
    tokenizer.itos = json.itos;
    tokenizer.unk = json.unk ?? '�';
    tokenizer.stoi = Object.fromEntries(tokenizer.itos.map((ch, index) => [ch, index]));
    tokenizer.vocabSize = tokenizer.itos.length;
    tokenizer.maxTokenLength = json.maxTokenLength ?? Math.max(1, ...tokenizer.itos.map((token) => Array.from(token).length));
    return tokenizer;
  }
}

function learnFrequentPieces(text, maxVocab, existingTokens) {
  const counts = new Map();
  const add = (piece) => {
    if (!piece || piece.length < 2 || piece.length > 36 || existingTokens.has(piece)) return;
    counts.set(piece, (counts.get(piece) ?? 0) + 1);
  };

  const patterns = [
    / ?[A-Za-z][A-Za-z0-9_'’-]{1,}/g,
    / ?\d+(?:[.,:]\d+)*/g,
    /\n\n/g,
    / ?[.!?;,]+/g,
    / ?[()\[\]{}<>]+/g,
  ];
  for (const pattern of patterns) {
    for (const match of text.matchAll(pattern)) add(match[0]);
  }

  // Add a small number of recurring byte-pair-ish fragments from longer words.
  for (const match of text.matchAll(/[A-Za-z][A-Za-z0-9_'’-]{4,}/g)) {
    const word = match[0];
    for (let size = 3; size <= Math.min(8, word.length); size += 1) {
      for (let index = 0; index <= word.length - size; index += 1) add(word.slice(index, index + size));
    }
  }

  return Array.from(counts.entries())
    .filter(([, count]) => count > 1)
    .sort((a, b) => (b[1] * Math.log2(b[0].length + 1)) - (a[1] * Math.log2(a[0].length + 1)))
    .slice(0, Math.max(0, maxVocab - existingTokens.size))
    .map(([piece]) => piece);
}

export function estimateParams(config, vocabSize = ESTIMATE_VOCAB_SIZE) {
  const d = config.dModel;
  const ff = config.ffDim;
  const layerNorms = 4 * d; // two norms per block: gamma + beta
  const attention = 4 * (d * d + d); // q, k, v, output projections + biases
  const feedForward = d * ff + ff + ff * d + d;
  const perLayer = layerNorms + attention + feedForward;
  const embeddings = vocabSize * d;
  const positions = config.context * d;
  const finalNorm = 2 * d;
  const head = d * vocabSize + vocabSize;
  return Math.round(embeddings + positions + config.nLayers * perLayer + finalNorm + head);
}

export function estimateTrainingBytes(params, context = 128, dModel = 128, layers = 2, batch = 1) {
  // Weights + Adam first moment + Adam second moment + current gradients.
  const parameterBytes = params * 4 * 4;
  // Rough forward/backward activation working set. Attention scores scale as T^2.
  const activationBytes = 4 * batch * layers * (context * dModel * 10 + context * context * 4);
  return parameterBytes + activationBytes;
}

export function bytesToHuman(bytes) {
  if (!Number.isFinite(bytes)) return 'unknown';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value >= 10 || unit === 0 ? value.toFixed(0) : value.toFixed(1)} ${units[unit]}`;
}

export function formatParams(params) {
  if (params >= 1_000_000_000) return `${(params / 1_000_000_000).toFixed(2)}B`;
  if (params >= 1_000_000) return `${(params / 1_000_000).toFixed(params >= 10_000_000 ? 0 : 1)}M`;
  if (params >= 1_000) return `${(params / 1_000).toFixed(0)}K`;
  return `${params}`;
}

function product(shape) {
  return shape.reduce((acc, value) => acc * value, 1);
}

function normalInit(shape, scale, seed) {
  return tf.randomNormal(shape, 0, scale, 'float32', seed);
}

function zeros(shape) {
  return tf.zeros(shape, 'float32');
}

function ones(shape) {
  return tf.ones(shape, 'float32');
}

function gelu(x) {
  return tf.tidy(() => {
    const cubic = x.pow(tf.scalar(3)).mul(0.044715);
    const inner = x.add(cubic).mul(Math.sqrt(2 / Math.PI));
    return x.mul(0.5).mul(tf.tanh(inner).add(1));
  });
}

export class LocalTransformerLM {
  constructor(config, tokenizer) {
    if (config.dModel % config.nHeads !== 0) {
      throw new Error(`dModel (${config.dModel}) must be divisible by nHeads (${config.nHeads}).`);
    }
    this.config = { ...config };
    this.tokenizer = tokenizer;
    this.vocabSize = tokenizer.vocabSize;
    this.scope = `v0id_${Date.now()}_${Math.random().toString(36).slice(2)}`;
    this.vars = {};
    this.trainableVars = [];
    this.logicalNameByVariableName = new Map();
    this.maskCache = new Map();
    this.optimizer = null;
    this.optimizerKey = '';
    this.initVariables();
  }

  initVariables() {
    const { dModel: d, context, nLayers, ffDim: ff } = this.config;
    // GPT-style small initialization. The previous Xavier-ish scale was too hot
    // in WebGL and could blow weights into NaN, which JSON.stringify displayed as null.
    const tokenScale = 0.02;
    const linearScale = 0.02;
    const residualScale = 0.02 / Math.sqrt(Math.max(1, 2 * nLayers));
    this.register('tokenEmbedding', normalInit([this.vocabSize, d], tokenScale, 11));
    this.register('positionEmbedding', normalInit([context, d], tokenScale, 12));

    for (let layer = 0; layer < nLayers; layer += 1) {
      const prefix = `block${layer}`;
      this.register(`${prefix}/ln1/gamma`, ones([d]));
      this.register(`${prefix}/ln1/beta`, zeros([d]));
      this.register(`${prefix}/attn/wq`, normalInit([d, d], linearScale, 101 + layer * 20));
      this.register(`${prefix}/attn/bq`, zeros([d]));
      this.register(`${prefix}/attn/wk`, normalInit([d, d], linearScale, 102 + layer * 20));
      this.register(`${prefix}/attn/bk`, zeros([d]));
      this.register(`${prefix}/attn/wv`, normalInit([d, d], linearScale, 103 + layer * 20));
      this.register(`${prefix}/attn/bv`, zeros([d]));
      this.register(`${prefix}/attn/wo`, normalInit([d, d], residualScale, 104 + layer * 20));
      this.register(`${prefix}/attn/bo`, zeros([d]));
      this.register(`${prefix}/ln2/gamma`, ones([d]));
      this.register(`${prefix}/ln2/beta`, zeros([d]));
      this.register(`${prefix}/ffn/w1`, normalInit([d, ff], linearScale, 201 + layer * 20));
      this.register(`${prefix}/ffn/b1`, zeros([ff]));
      this.register(`${prefix}/ffn/w2`, normalInit([ff, d], residualScale, 202 + layer * 20));
      this.register(`${prefix}/ffn/b2`, zeros([d]));
    }

    this.register('finalLn/gamma', ones([d]));
    this.register('finalLn/beta', zeros([d]));
    this.register('lmHead/w', normalInit([d, this.vocabSize], 0.02, 301));
    this.register('lmHead/b', zeros([this.vocabSize]));
  }

  register(logicalName, tensor) {
    const variable = tf.variable(tensor, true, `${this.scope}/${logicalName}`);
    this.vars[logicalName] = variable;
    this.trainableVars.push(variable);
    this.logicalNameByVariableName.set(variable.name, logicalName);
    // Do NOT dispose the initialization tensor immediately; in WebGL backends
    // disposing it before the variable has fully settled can corrupt variable
    // memory and lead to NaN weights / broken exports / constant K output.
    return variable;
  }

  parameterCount() {
    return Object.values(this.vars).reduce((sum, variable) => sum + product(variable.shape), 0);
  }

  getCausalMask(sequenceLength) {
    if (!this.maskCache.has(sequenceLength)) {
      const values = new Float32Array(sequenceLength * sequenceLength);
      for (let row = 0; row < sequenceLength; row += 1) {
        for (let col = row + 1; col < sequenceLength; col += 1) {
          values[row * sequenceLength + col] = -1e9;
        }
      }
      this.maskCache.set(sequenceLength, tf.keep(tf.tensor4d(values, [1, 1, sequenceLength, sequenceLength], 'float32')));
    }
    return this.maskCache.get(sequenceLength);
  }

  layerNorm(x, gamma, beta) {
    return tf.tidy(() => {
      const mean = x.mean(-1, true);
      const centered = x.sub(mean);
      const variance = centered.square().mean(-1, true);
      return centered.mul(tf.rsqrt(variance.add(1e-5))).mul(gamma).add(beta);
    });
  }

  attention(x, layer) {
    return tf.tidy(() => {
      const { dModel: d, nHeads } = this.config;
      const [batch, sequenceLength] = x.shape;
      const headSize = d / nHeads;
      const prefix = `block${layer}/attn`;
      const flat = x.reshape([-1, d]);
      const project = (wName, bName) => flat.matMul(this.vars[`${prefix}/${wName}`]).add(this.vars[`${prefix}/${bName}`])
        .reshape([batch, sequenceLength, nHeads, headSize])
        .transpose([0, 2, 1, 3]);

      const q = project('wq', 'bq');
      const k = project('wk', 'bk');
      const v = project('wv', 'bv');
      const scores = tf.matMul(q, k, false, true)
        .div(Math.sqrt(headSize))
        .add(this.getCausalMask(sequenceLength));
      const weights = tf.softmax(scores, -1);
      const mixed = tf.matMul(weights, v)
        .transpose([0, 2, 1, 3])
        .reshape([batch * sequenceLength, d]);
      return mixed.matMul(this.vars[`${prefix}/wo`]).add(this.vars[`${prefix}/bo`])
        .reshape([batch, sequenceLength, d]);
    });
  }

  feedForward(x, layer) {
    return tf.tidy(() => {
      const { dModel: d, ffDim: ff } = this.config;
      const [batch, sequenceLength] = x.shape;
      const prefix = `block${layer}/ffn`;
      const flat = x.reshape([-1, d]);
      const hidden = gelu(flat.matMul(this.vars[`${prefix}/w1`]).add(this.vars[`${prefix}/b1`]));
      return hidden.matMul(this.vars[`${prefix}/w2`]).add(this.vars[`${prefix}/b2`])
        .reshape([batch, sequenceLength, d]);
    });
  }

  forward(inputIds) {
    return tf.tidy(() => {
      const { dModel: d, context, nLayers } = this.config;
      const [batch, sequenceLength] = inputIds.shape;
      if (sequenceLength > context) {
        throw new Error(`Prompt length ${sequenceLength} exceeds context window ${context}.`);
      }
      const tokenEmbeddings = tf.gather(this.vars.tokenEmbedding, inputIds);
      const positions = tf.gather(this.vars.positionEmbedding, tf.range(0, sequenceLength, 1, 'int32')).expandDims(0);
      let x = tokenEmbeddings.add(positions);

      for (let layer = 0; layer < nLayers; layer += 1) {
        const prefix = `block${layer}`;
        const norm1 = this.layerNorm(x, this.vars[`${prefix}/ln1/gamma`], this.vars[`${prefix}/ln1/beta`]);
        x = x.add(this.attention(norm1, layer));
        const norm2 = this.layerNorm(x, this.vars[`${prefix}/ln2/gamma`], this.vars[`${prefix}/ln2/beta`]);
        x = x.add(this.feedForward(norm2, layer));
      }

      const final = this.layerNorm(x, this.vars['finalLn/gamma'], this.vars['finalLn/beta']);
      return final.reshape([batch * sequenceLength, d])
        .matMul(this.vars['lmHead/w'])
        .add(this.vars['lmHead/b'])
        .reshape([batch, sequenceLength, this.vocabSize]);
    });
  }

  lossForBatch(xs, ys) {
    return tf.tidy(() => {
      const logits = this.forward(xs);
      const [batch, sequenceLength, vocabSize] = logits.shape;
      const flatLogits = logits.reshape([batch * sequenceLength, vocabSize]);
      const flatLabels = ys.flatten();

      // Stable sparse softmax cross entropy:
      // log(sum(exp(logits))) - logits[label], with log-sum-exp stabilization.
      // This avoids the older one-hot path and is harder to push into NaN.
      const maxLogits = flatLogits.max(-1, true);
      const shifted = flatLogits.sub(maxLogits);
      const logSumExp = tf.log(tf.exp(shifted).sum(-1)).add(maxLogits.squeeze([-1]));
      // gatherND has incomplete gradient support in some tfjs backends, so use
      // oneHot only for the target lookup while keeping the stable log-sum-exp loss.
      const labelMask = tf.oneHot(flatLabels, vocabSize);
      const trueLogits = flatLogits.mul(labelMask).sum(-1);
      return logSumExp.sub(trueLogits).mean();
    });
  }

  ensureOptimizer(learningRate) {
    const key = `${learningRate}`;
    if (!this.optimizer || this.optimizerKey !== key) {
      // Larger epsilon is more stable on browser/WebGL backends.
      this.optimizer = tf.train.adam(learningRate, 0.9, 0.95, 1e-6);
      this.optimizerKey = key;
    }
  }

  async trainStep(xs, ys, options = {}) {
    const learningRate = Number(options.learningRate ?? this.config.defaultLr ?? 0.001);
    const clipNorm = Number(options.clipNorm ?? 1);
    const weightDecay = Number(options.weightDecay ?? 0.01);
    this.ensureOptimizer(learningRate);

    const { value, grads } = tf.variableGrads(() => tf.tidy(() => this.lossForBatch(xs, ys)), this.trainableVars);
    const gradNames = Object.keys(grads);
    const safeGrads = {};
    let gradNormTensor = null;
    let allFiniteGradTensor = null;
    let clippedGrads = null;

    for (const name of gradNames) {
      safeGrads[name] = tf.tidy(() => tf.where(tf.isFinite(grads[name]), grads[name], tf.zerosLike(grads[name])));
    }

    if (gradNames.length > 0) {
      gradNormTensor = tf.tidy(() => {
        const squaredSums = gradNames.map((name) => safeGrads[name].square().sum());
        return tf.sqrt(tf.addN(squaredSums).add(1e-12));
      });
      allFiniteGradTensor = tf.tidy(() => {
        const checks = gradNames.map((name) => tf.all(tf.isFinite(grads[name])));
        return tf.all(tf.stack(checks));
      });
    }

    // Read numeric health BEFORE applyGradients. The old code applied first, so a
    // single NaN loss/gradient permanently poisoned the weights and export became nulls.
    const [lossValue, gradNormValue, allFiniteGradValue] = await Promise.all([
      value.data().then((data) => data[0]),
      gradNormTensor ? gradNormTensor.data().then((data) => data[0]) : Promise.resolve(0),
      allFiniteGradTensor ? allFiniteGradTensor.data().then((data) => data[0]) : Promise.resolve(1),
    ]);

    let skipped = false;
    const lossIsFinite = Number.isFinite(lossValue);
    const gradNormIsFinite = Number.isFinite(gradNormValue);
    if (!lossIsFinite || !gradNormIsFinite || allFiniteGradValue !== 1) {
      skipped = true;
    } else if (clipNorm > 0 && gradNames.length > 0) {
      const scale = tf.tidy(() => tf.minimum(tf.scalar(1), tf.scalar(clipNorm).div(gradNormTensor.add(1e-8))));
      clippedGrads = {};
      for (const name of gradNames) {
        clippedGrads[name] = tf.tidy(() => safeGrads[name].mul(scale));
      }
      scale.dispose();
      this.optimizer.applyGradients(clippedGrads);
    } else {
      this.optimizer.applyGradients(safeGrads);
    }

    if (!skipped && weightDecay > 0) {
      const decayScale = Math.max(0, 1 - learningRate * weightDecay);
      for (const variable of this.trainableVars) {
        const logicalName = this.logicalNameByVariableName.get(variable.name) ?? '';
        const isBiasOrNorm = logicalName.includes('/b') || logicalName.toLowerCase().includes('ln') || logicalName.endsWith('/beta') || logicalName.endsWith('/gamma');
        if (!isBiasOrNorm) {
          tf.tidy(() => variable.assign(variable.mul(decayScale)));
        }
      }
    }

    value.dispose();
    if (gradNormTensor) gradNormTensor.dispose();
    if (allFiniteGradTensor) allFiniteGradTensor.dispose();
    Object.values(grads).forEach((tensor) => tensor.dispose());
    Object.values(safeGrads).forEach((tensor) => tensor.dispose());
    if (clippedGrads) Object.values(clippedGrads).forEach((tensor) => tensor.dispose());

    return {
      loss: lossValue,
      gradNorm: gradNormValue,
      perplexity: Number.isFinite(lossValue) ? Math.exp(Math.min(20, lossValue)) : Infinity,
      skipped,
      hadBadGradients: allFiniteGradValue !== 1,
    };
  }

  async generate(prompt, options = {}, onToken = () => {}) {
    const maxNewTokens = Number(options.maxNewTokens ?? 180);
    const temperature = Math.max(0.05, Number(options.temperature ?? 0.85));
    const topK = Math.max(1, Number(options.topK ?? 40));
    const topP = Math.min(1, Math.max(0.01, Number(options.topP ?? 0.92)));
    const repetitionPenalty = Math.max(1, Number(options.repetitionPenalty ?? 1.08));
    const maxRepeat = Math.max(2, Number(options.maxRepeat ?? 8));
    const streamEvery = Math.max(1, Number(options.streamEvery ?? 3));
    const yieldEvery = Math.max(1, Number(options.yieldEvery ?? 6));
    const stopSequences = options.stopSequences ?? ['\nUser:', '\nUSER:', '\nHuman:'];
    const ids = this.tokenizer.encode(prompt);
    // Repair any corruption left by unstable training so chat doesn't emit
    // constant tokens (e.g. all "K") or throw non-finite errors.
    await this.sanitizeWeights();
    let generated = '';

    for (let step = 0; step < maxNewTokens; step += 1) {
      if (options.signal?.aborted || options.shouldStop?.()) break;
      const contextIds = ids.slice(-this.config.context);
      const input = tf.tensor2d(contextIds, [1, contextIds.length], 'int32');
      const logitsTensor = tf.tidy(() => {
        const logits = this.forward(input);
        return logits.slice([0, contextIds.length - 1, 0], [1, 1, this.vocabSize]).reshape([this.vocabSize]);
      });
      input.dispose();
      const logits = Array.from(await logitsTensor.data(), (value) => (Number.isFinite(value) ? value : -Infinity));
      logitsTensor.dispose();

      if (!logits.some(Number.isFinite)) {
        throw new Error('Model produced non-finite logits. The weights are numerically corrupted; repair or reinitialize the model.');
      }

      applyRepetitionPenalty(logits, ids.slice(-128), repetitionPenalty);
      blockDegenerateRepeat(logits, ids, maxRepeat);
      const nextId = sampleFromLogits(logits, { temperature, topK, topP });
      ids.push(nextId);
      const piece = this.tokenizer.decode([nextId]);
      generated += piece;
      const shouldStream = step % streamEvery === 0 || step === maxNewTokens - 1;
      if (shouldStream) onToken(piece, generated, { step, done: false });

      if (stopSequences.some((sequence) => generated.includes(sequence))) {
        generated = trimAtStop(generated, stopSequences);
        onToken('', generated, { step, done: true });
        break;
      }
      if (step % yieldEvery === yieldEvery - 1) await tf.nextFrame();
    }

    onToken('', generated, { step: Math.min(maxNewTokens, generated.length), done: true });
    return generated;
  }

  async sanitizeWeights(options = {}) {
    const maxAbsWeight = Number(options.maxAbsWeight ?? MAX_ABS_WEIGHT);
    const stats = {
      tensorsChecked: 0,
      valuesChecked: 0,
      nonFinite: 0,
      clipped: 0,
      repairedTensors: 0,
    };

    let index = 0;
    for (const variable of Object.values(this.vars)) {
      const raw = await variable.data();
      const cleaned = new Float32Array(raw.length);
      let changed = false;
      for (let i = 0; i < raw.length; i += 1) {
        const value = raw[i];
        stats.valuesChecked += 1;
        if (!Number.isFinite(value)) {
          cleaned[i] = 0;
          stats.nonFinite += 1;
          changed = true;
        } else if (Math.abs(value) > maxAbsWeight) {
          cleaned[i] = Math.sign(value) * maxAbsWeight;
          stats.clipped += 1;
          changed = true;
        } else {
          cleaned[i] = value;
        }
      }
      stats.tensorsChecked += 1;
      if (changed) {
        const tensor = tf.tensor(cleaned, variable.shape, 'float32');
        variable.assign(tensor);
        tensor.dispose();
        stats.repairedTensors += 1;
      }
      index += 1;
      if (index % REPAIR_YIELD_INTERVAL === 0) await tf.nextFrame();
    }

    return stats;
  }

  async exportJSON(options = {}) {
    const repairStats = await this.sanitizeWeights({ maxAbsWeight: options.maxAbsWeight ?? MAX_ABS_WEIGHT });
    const weights = [];
    for (const [name, variable] of Object.entries(this.vars)) {
      const data = await variable.data();
      weights.push({
        name,
        shape: variable.shape,
        dtype: 'float32',
        encoding: 'base64-f32-le',
        data: float32ToBase64(data),
      });
    }
    return {
      format: EXPORT_FORMAT,
      compatibleFormats: ['v0idgpt-local-transformer-v1'],
      createdAt: new Date().toISOString(),
      config: this.config,
      tokenizer: this.tokenizer.toJSON(),
      exportStats: repairStats,
      weights,
    };
  }

  async importWeights(weights) {
    let repairedValues = 0;
    for (const item of weights) {
      const variable = this.vars[item.name];
      if (!variable) continue;
      const expected = product(variable.shape);
      const decoded = decodeWeightItem(item);
      if (expected !== decoded.length) {
        throw new Error(`Weight ${item.name} has ${decoded.length} values, expected ${expected}.`);
      }
      const sanitized = sanitizeNumberArray(decoded, MAX_ABS_WEIGHT);
      repairedValues += sanitized.stats.nonFinite + sanitized.stats.clipped;
      const tensor = tf.tensor(sanitized.values, item.shape, 'float32');
      variable.assign(tensor);
      tensor.dispose();
      await tf.nextFrame();
    }
    return { repairedValues };
  }

  dispose() {
    if (this.optimizer?.dispose) this.optimizer.dispose();
    Object.values(this.vars).forEach((variable) => variable.dispose());
    this.trainableVars = [];
    this.vars = {};
    this.maskCache.forEach((mask) => mask.dispose());
    this.maskCache.clear();
  }
}

export function makeBatch(encodedIds, context, batchSize) {
  if (encodedIds.length < 3) {
    throw new Error('Training text is too short. Add at least a few sentences.');
  }
  const sequenceLength = Math.max(1, Math.min(context, encodedIds.length - 1));
  const xs = new Int32Array(batchSize * sequenceLength);
  const ys = new Int32Array(batchSize * sequenceLength);
  const maxStart = Math.max(0, encodedIds.length - sequenceLength - 1);
  for (let batch = 0; batch < batchSize; batch += 1) {
    const start = maxStart === 0 ? 0 : Math.floor(Math.random() * (maxStart + 1));
    for (let offset = 0; offset < sequenceLength; offset += 1) {
      xs[batch * sequenceLength + offset] = encodedIds[start + offset];
      ys[batch * sequenceLength + offset] = encodedIds[start + offset + 1];
    }
  }
  return {
    xs: tf.tensor2d(xs, [batchSize, sequenceLength], 'int32'),
    ys: tf.tensor2d(ys, [batchSize, sequenceLength], 'int32'),
    sequenceLength,
  };
}

export function sampleFromLogits(logits, { temperature = 1, topK = 40, topP = 0.92 } = {}) {
  let ranked = logits
    .map((value, index) => ({ index, value: Number.isFinite(value) ? value / temperature : -Infinity }))
    .filter((item) => Number.isFinite(item.value))
    .sort((a, b) => b.value - a.value)
    .slice(0, Math.min(topK, logits.length));

  if (ranked.length === 0) {
    throw new Error('No finite logits available for sampling.');
  }

  const max = ranked[0].value;
  let probabilities = ranked.map((item) => ({
    ...item,
    probability: Math.exp(Math.max(-80, Math.min(80, item.value - max))),
  }));
  let total = probabilities.reduce((sum, item) => sum + item.probability, 0);
  if (!Number.isFinite(total) || total <= 0) return ranked[0].index;
  probabilities = probabilities.map((item) => ({ ...item, probability: item.probability / total }));

  if (topP < 1) {
    let cumulative = 0;
    const nucleus = [];
    for (const item of probabilities) {
      cumulative += item.probability;
      nucleus.push(item);
      if (cumulative >= topP) break;
    }
    probabilities = nucleus.length ? nucleus : [probabilities[0]];
    total = probabilities.reduce((sum, item) => sum + item.probability, 0);
    if (!Number.isFinite(total) || total <= 0) return probabilities[0].index;
    probabilities = probabilities.map((item) => ({ ...item, probability: item.probability / total }));
  }

  let roll = Math.random();
  for (const item of probabilities) {
    roll -= item.probability;
    if (roll <= 0) return item.index;
  }
  return probabilities[probabilities.length - 1]?.index ?? ranked[0].index;
}

function applyRepetitionPenalty(logits, recentIds, penalty) {
  if (penalty <= 1) return;
  const counts = new Map();
  for (const id of recentIds) counts.set(id, (counts.get(id) ?? 0) + 1);
  for (const [id, count] of counts) {
    if (id < 0 || id >= logits.length || !Number.isFinite(logits[id])) continue;
    const adjustedPenalty = penalty ** Math.min(4, count);
    logits[id] = logits[id] > 0 ? logits[id] / adjustedPenalty : logits[id] * adjustedPenalty;
  }
}

function blockDegenerateRepeat(logits, ids, maxRepeat) {
  const lastId = ids[ids.length - 1];
  if (lastId === undefined) return;
  let run = 0;
  for (let i = ids.length - 1; i >= 0 && ids[i] === lastId; i -= 1) run += 1;
  if (run >= maxRepeat && lastId >= 0 && lastId < logits.length) {
    logits[lastId] = -Infinity;
  }
}

function sanitizeNumberArray(values, maxAbsWeight = MAX_ABS_WEIGHT) {
  const cleaned = new Float32Array(values.length);
  const stats = { nonFinite: 0, clipped: 0 };
  for (let index = 0; index < values.length; index += 1) {
    const value = values[index];
    if (!Number.isFinite(value)) {
      cleaned[index] = 0;
      stats.nonFinite += 1;
    } else if (Math.abs(value) > maxAbsWeight) {
      cleaned[index] = Math.sign(value) * maxAbsWeight;
      stats.clipped += 1;
    } else {
      cleaned[index] = value;
    }
  }
  return { values: cleaned, stats };
}

function float32ToBase64(values) {
  const bytes = new Uint8Array(values.buffer, values.byteOffset, values.byteLength);
  let binary = '';
  const chunkSize = 0x8000;
  for (let offset = 0; offset < bytes.length; offset += chunkSize) {
    const chunk = bytes.subarray(offset, offset + chunkSize);
    for (let i = 0; i < chunk.length; i++) {
      binary += String.fromCharCode(chunk[i]);
    }
  }
  const btoaFn = (globalThis && globalThis.btoa) ? globalThis.btoa : (typeof btoa === 'function' ? btoa : null);
  if (!btoaFn) throw new Error('btoa is not available; cannot encode weights');
  return btoaFn(binary);
}

function base64ToFloat32(base64) {
  const atobFn = (globalThis && globalThis.atob) ? globalThis.atob : (typeof atob === 'function' ? atob : null);
  if (!atobFn) throw new Error('atob is not available; cannot decode weights');
  const binary = atobFn(base64);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }
  const aligned = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
  return new Float32Array(aligned);
}

function decodeWeightItem(item) {
  if (item.encoding === 'base64-f32-le') {
    return base64ToFloat32(item.data);
  }
  if (Array.isArray(item.data)) {
    return item.data;
  }
  throw new Error(`Unsupported weight encoding for ${item.name}.`);
}

function trimAtStop(text, stopSequences) {
  let cut = text.length;
  for (const stop of stopSequences) {
    const index = text.indexOf(stop);
    if (index !== -1) cut = Math.min(cut, index);
  }
  return text.slice(0, cut);
}
