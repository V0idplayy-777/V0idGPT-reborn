import * as tf from '@tensorflow/tfjs';
import './style.css';
import {
  CharTokenizer,
  LocalTransformerLM,
  MODEL_PRESETS,
  ESTIMATE_VOCAB_SIZE,
  bytesToHuman,
  estimateParams,
  estimateTrainingBytes,
  formatParams,
  makeBatch,
} from './ai-core.js';

const DEMO_CORPUS = `V0idGPT Reborn is a local transformer chatbot. It learns to speak by studying the text you paste. The more dialogue and patterns you give it, the better it predicts the next word.

User: What are you?
Assistant: I am a locally trained decoder-only transformer. I learn patterns from your text, not from the internet. I use embeddings, causal self-attention, feed-forward layers, layer normalization, and Adam training.

User: How do you answer?
Assistant: I predict the next token from patterns I have seen. I should be direct, helpful, and admit when I am unsure. Good results need good training text and enough steps.

User: How do I make you talk properly?
Assistant: Paste dialogue or notes into the training box. Use at least a few sentences of back-and-forth. Train for more steps. The model learns your style from the text, not from a hardcoded brain.

User: Explain local training.
Assistant: Choose a model size, paste training text, then click Start training. The model compares predicted tokens to real tokens and updates weights. Larger models learn more structure but need more memory and time.`;

const state = {
  backendReady: false,
  selectedPreset: MODEL_PRESETS[0],
  tokenizer: new CharTokenizer('User:\nAssistant:\n'),
  model: null,
  activeConfigId: null,
  isTraining: false,
  abortTraining: false,
  generationAbort: null,
  chatLog: '',
  trainedSteps: 0,
  lastLoss: null,
};

const app = document.querySelector('#app');
app.innerHTML = `
  <div class="shell">
    <header class="hero">
      <div>
        <p class="eyebrow">No API · Real decoder transformer · Runs locally in the browser</p>
        <h1>V0idGPT Reborn</h1>
        <p class="hero-copy">Train a GPT-style hybrid character/subword LLM from scratch, then chat with the weights you just trained. The chatbot generates tokens autoregressively from your input — no canned response table, no external model endpoint.</p>
      </div>
      <div class="runtime-card">
        <span class="status-dot" id="backendDot"></span>
        <strong id="backendName">Starting TensorFlow.js…</strong>
        <small id="runtimeStats">Preparing local matmul backend</small>
      </div>
    </header>

    <main class="grid-layout">
      <section class="panel models-panel">
        <div class="panel-heading">
          <div>
            <p class="eyebrow">1 · Pick architecture</p>
            <h2>10 trainable model sizes</h2>
          </div>
          <button class="ghost-button" id="recalcButton" title="Refresh memory and parameter estimates">Refresh</button>
        </div>
        <div class="model-grid" id="modelGrid"></div>
        <div class="selected-model" id="selectedDetails"></div>
        <label class="unlock-row">
          <input type="checkbox" id="unlockLarge" />
          <span>Unlock high-RAM initialisation attempts for dangerous models</span>
        </label>
        <button class="primary-button full" id="initModel">Initialize selected transformer</button>
      </section>

      <section class="panel train-panel">
        <div class="panel-heading">
          <div>
            <p class="eyebrow">2 · Train locally</p>
            <h2>Chat/train UI</h2>
          </div>
          <span class="pill" id="trainStatePill">Idle</span>
        </div>

        <label class="field-label" for="trainText">Training text</label>
        <textarea id="trainText" spellcheck="false" placeholder="Paste dialogue, notes, docs, code, or any text. The model learns next-token prediction from this data."></textarea>
        <div class="textarea-actions">
          <button class="ghost-button" id="demoCorpus">Insert tiny sanity corpus</button>
          <button class="ghost-button" id="appendChat">Append chat log</button>
          <button class="ghost-button danger-text" id="clearCorpus">Clear</button>
        </div>

        <div class="controls-grid">
          <label>Steps<input id="stepsInput" type="number" min="1" max="100000" value="120" /></label>
          <label>Batch<input id="batchInput" type="number" min="1" max="128" value="16" /></label>
          <label>Learning rate<input id="lrInput" type="number" step="0.0001" min="0.00001" value="0.0015" /></label>
          <label>Grad clip<input id="clipInput" type="number" step="0.1" min="0" value="1" /></label>
          <label>AdamW decay<input id="decayInput" type="number" step="0.001" min="0" value="0.01" /></label>
          <label>Report every<input id="reportInput" type="number" min="1" max="1000" value="5" /></label>
        </div>

        <div class="progress-wrap">
          <div class="progress-top">
            <span id="progressLabel">No training run yet.</span>
            <span id="lossLabel">loss: —</span>
          </div>
          <progress id="trainProgress" value="0" max="100"></progress>
          <small id="trainMetrics">Perplexity, tokens/sec, gradient norm, and tensor memory show here while training.</small>
        </div>

        <div class="button-row">
          <button class="primary-button" id="startTrain">Start training</button>
          <button class="secondary-button" id="stopTrain" disabled>Stop</button>
        </div>

        <div class="button-row thin">
          <button class="ghost-button" id="repairWeights" disabled>Repair NaN/Inf weights</button>
          <button class="ghost-button" id="exportModel" disabled>Export compact JSON</button>
          <label class="ghost-button file-button" for="importModel">Import JSON</label>
          <input id="importModel" type="file" accept="application/json,.json" hidden />
        </div>
      </section>

      <section class="panel chat-panel">
        <div class="panel-heading">
          <div>
            <p class="eyebrow">3 · Chat with your model</p>
            <h2>Autoregressive generation</h2>
          </div>
          <button class="ghost-button" id="clearChat">Clear chat</button>
        </div>

        <div class="chat-messages" id="chatMessages">
          <div class="system-message">Initialize and train a model, then ask it something. It will generate tokenizer pieces from the transformer logits.</div>
        </div>

        <div class="chat-input-row">
          <textarea id="chatInput" rows="3" placeholder="Message the local model…"></textarea>
          <button class="primary-button send" id="sendChat">Generate</button>
        </div>

        <div class="sampling-controls">
          <label>Max tokens<input id="maxTokensInput" type="number" min="1" max="2000" value="180" /></label>
          <label>Temperature<input id="tempInput" type="number" min="0.05" max="2" step="0.05" value="0.85" /></label>
          <label>Top-K<input id="topKInput" type="number" min="1" max="256" value="40" /></label>
          <label>Top-P<input id="topPInput" type="number" min="0.01" max="1" step="0.01" value="0.92" /></label>
          <label>Repetition penalty<input id="repInput" type="number" min="1" max="2" step="0.01" value="1.08" /></label>
        </div>
        <label class="unlock-row compact">
          <input type="checkbox" id="autoTrainChat" checked />
          <span>Append chat transcript to training box after each generation</span>
        </label>
      </section>

      <section class="panel info-panel">
        <p class="eyebrow">Quality-safe speedups included</p>
        <ul>
          <li>Vectorized TensorFlow.js matmuls with WebGL acceleration when available.</li>
          <li>True causal multi-head self-attention, learned token/position embeddings, GELU FFN, residuals, and layer norm.</li>
          <li>Adam optimizer, stable sparse cross-entropy, pre-apply NaN guards, gradient clipping, AdamW-style decay, and aggressive tensor disposal.</li>
          <li>Compact base64 weight export, NaN/Inf repair, Top-K + nucleus sampling, loop blocking, and repetition penalty.</li>
        </ul>
      </section>
    </main>
  </div>
`;

const $ = (selector) => document.querySelector(selector);
const refs = {
  backendDot: $('#backendDot'),
  backendName: $('#backendName'),
  runtimeStats: $('#runtimeStats'),
  modelGrid: $('#modelGrid'),
  selectedDetails: $('#selectedDetails'),
  unlockLarge: $('#unlockLarge'),
  initModel: $('#initModel'),
  recalcButton: $('#recalcButton'),
  trainText: $('#trainText'),
  demoCorpus: $('#demoCorpus'),
  appendChat: $('#appendChat'),
  clearCorpus: $('#clearCorpus'),
  stepsInput: $('#stepsInput'),
  batchInput: $('#batchInput'),
  lrInput: $('#lrInput'),
  clipInput: $('#clipInput'),
  decayInput: $('#decayInput'),
  reportInput: $('#reportInput'),
  trainStatePill: $('#trainStatePill'),
  trainProgress: $('#trainProgress'),
  progressLabel: $('#progressLabel'),
  lossLabel: $('#lossLabel'),
  trainMetrics: $('#trainMetrics'),
  startTrain: $('#startTrain'),
  stopTrain: $('#stopTrain'),
  repairWeights: $('#repairWeights'),
  exportModel: $('#exportModel'),
  importModel: $('#importModel'),
  chatMessages: $('#chatMessages'),
  chatInput: $('#chatInput'),
  sendChat: $('#sendChat'),
  clearChat: $('#clearChat'),
  maxTokensInput: $('#maxTokensInput'),
  tempInput: $('#tempInput'),
  topKInput: $('#topKInput'),
  topPInput: $('#topPInput'),
  repInput: $('#repInput'),
  autoTrainChat: $('#autoTrainChat'),
};

function riskBadge(risk) {
  return {
    safe: 'SAFE',
    caution: 'GPU ADVISED',
    warning: '⚠ BIG',
    danger: '🚨 DANGER',
    extreme: '☢ HUGE',
    meltdown: '💀 1B RED BUTTON',
  }[risk] ?? 'LOCAL';
}

function riskWarning(preset, params = preset.estimatedParams) {
  if (preset.risk === 'meltdown') return 'BILLION-PARAMETER MODE: likely to crash a normal browser tab. Use only on huge RAM/VRAM hardware.';
  if (preset.risk === 'extreme') return 'EXTREME: workstation-class memory required. Export your work before attempting.';
  if (preset.risk === 'danger') return 'DANGER: this can freeze or kill the tab on consumer laptops.';
  if (preset.risk === 'warning') return 'WARNING: reduce batch size and expect slow training.';
  if (params > 4_000_000) return 'Caution: use WebGL/GPU and keep batches small.';
  return 'Good quick-start size for this in-browser implementation.';
}

function renderModelGrid() {
  refs.modelGrid.innerHTML = '';
  for (const preset of MODEL_PRESETS) {
    const button = document.createElement('button');
    const bytes = estimateTrainingBytes(preset.estimatedParams, preset.context, preset.dModel, preset.nLayers, preset.defaultBatch);
    button.className = `model-card ${preset.id === state.selectedPreset.id ? 'active' : ''} risk-${preset.risk}`;
    button.innerHTML = `
      <span class="badge">${riskBadge(preset.risk)}</span>
      <strong>${preset.name}</strong>
      <small>${preset.tagline}</small>
      <span class="model-meta">${formatParams(preset.estimatedParams)} params · ctx ${preset.context} · ${bytesToHuman(bytes)} train est.</span>
    `;
    button.addEventListener('click', () => {
      state.selectedPreset = preset;
      refs.batchInput.value = preset.defaultBatch;
      refs.lrInput.value = preset.defaultLr;
      renderModelGrid();
      updateSelectedDetails();
    });
    refs.modelGrid.appendChild(button);
  }
}

function updateSelectedDetails() {
  const preset = state.selectedPreset;
  const vocabSize = state.tokenizer?.vocabSize ?? ESTIMATE_VOCAB_SIZE;
  const params = estimateParams(preset, vocabSize);
  const batch = Number(refs.batchInput.value || preset.defaultBatch || 1);
  const trainBytes = estimateTrainingBytes(params, preset.context, preset.dModel, preset.nLayers, batch);
  const active = state.model && state.activeConfigId === preset.id;
  const deviceMemory = navigator.deviceMemory ? `${navigator.deviceMemory} GB device memory reported` : 'device memory unknown';
  refs.selectedDetails.innerHTML = `
    <div class="selected-title">
      <strong>${preset.name}</strong>
      <span class="badge risk-${preset.risk}">${riskBadge(preset.risk)}</span>
    </div>
    <div class="spec-grid">
      <span><b>${formatParams(params)}</b><small>estimated params @ vocab ${vocabSize}</small></span>
      <span><b>${preset.nLayers}</b><small>decoder blocks</small></span>
      <span><b>${preset.nHeads}</b><small>attention heads</small></span>
      <span><b>${preset.dModel}</b><small>d_model</small></span>
      <span><b>${preset.ffDim}</b><small>FFN width</small></span>
      <span><b>${preset.context}</b><small>context tokens</small></span>
      <span><b>${bytesToHuman(trainBytes)}</b><small>rough training memory</small></span>
      <span><b>${deviceMemory}</b><small>browser hint</small></span>
    </div>
    <p class="warning-copy ${preset.risk !== 'safe' ? 'loud' : ''}">${riskWarning(preset, params)}</p>
    ${active ? '<p class="ready-copy">This architecture is currently initialized.</p>' : ''}
  `;
}

async function setupBackend() {
  refs.backendName.textContent = 'Selecting fastest local backend…';
  refs.backendDot.classList.add('loading');
  try {
    // Quality-safe WebGL packing: faster tensor math without replacing the model with shortcuts.
    // Do not force half-float textures; that can destabilize training on some GPUs.
    tf.env().set('WEBGL_PACK', true);
    tf.env().set('WEBGL_FORCE_F16_TEXTURES', false);
  } catch {
    // Some backends lock flags after initialization. Safe to ignore.
  }

  try {
    await tf.setBackend('webgl');
  } catch (webglError) {
    console.warn('WebGL backend unavailable, falling back to CPU:', webglError);
    await tf.setBackend('cpu');
  }
  await tf.ready();
  state.backendReady = true;
  refs.backendDot.classList.remove('loading');
  refs.backendDot.classList.add('ready');
  refreshRuntimeStats();
}

function refreshRuntimeStats() {
  const memory = tf.memory();
  const modelParams = state.model ? formatParams(state.model.parameterCount()) : 'no model';
  refs.backendName.textContent = `Backend: ${tf.getBackend()}`;
  refs.runtimeStats.textContent = `${modelParams} · tensors ${memory.numTensors} · ${bytesToHuman(memory.numBytes)} tensor memory · trained steps ${state.trainedSteps}`;
  refs.exportModel.disabled = !state.model || state.isTraining;
  refs.repairWeights.disabled = !state.model || state.isTraining;
}

function setTrainState(label, busy = false) {
  refs.trainStatePill.textContent = label;
  refs.trainStatePill.classList.toggle('busy', busy);
  refs.startTrain.disabled = busy;
  refs.stopTrain.disabled = !busy;
  refs.initModel.disabled = busy;
  refs.repairWeights.disabled = busy || !state.model;
}

function getTrainingCorpus() {
  return refs.trainText.value.trim();
}

function currentModelWouldBeUnsafe(preset, params) {
  if (params <= 5_000_000) return false;
  return !refs.unlockLarge.checked;
}

function unknownCharacters(text, tokenizer) {
  const missing = new Set();
  for (const ch of Array.from(text)) {
    if (tokenizer.stoi[ch] === undefined) missing.add(ch);
    if (missing.size >= 12) break;
  }
  return Array.from(missing);
}

async function initializeSelectedModel({ quiet = false } = {}) {
  if (!state.backendReady) await setupBackend();
  const preset = state.selectedPreset;
  if (preset && preset.id === 'wtf-1.76q') {
    window.alert('What the fuck?');
    return false;
  }
  const seedText = `${refs.trainText.value}\n${state.chatLog}\nUser:\nAssistant:\n`;
  const nextTokenizer = new CharTokenizer(seedText);
  const params = estimateParams(preset, nextTokenizer.vocabSize);
  const batch = Number(refs.batchInput.value || preset.defaultBatch || 1);
  const trainingBytes = estimateTrainingBytes(params, preset.context, preset.dModel, preset.nLayers, batch);

  if (currentModelWouldBeUnsafe(preset, params)) {
    updateStatus(`Locked ${preset.name}: ${formatParams(params)} params needs roughly ${bytesToHuman(trainingBytes)} while training. Tick the high-RAM unlock box if you really want to try it.`, 'warn');
    return false;
  }

  if (params > 20_000_000) {
    const ok = window.confirm(`${preset.name} is a real ${formatParams(params)} parameter allocation attempt. Rough training memory: ${bytesToHuman(trainingBytes)} plus browser overhead. It may freeze or crash this tab. Continue?`);
    if (!ok) return false;
  }

  if (state.model && !quiet) {
    const ok = window.confirm('Reinitializing destroys the currently loaded weights. Continue?');
    if (!ok) return false;
  }

  try {
    refs.initModel.disabled = true;
    updateStatus(`Allocating ${preset.name} locally…`, 'busy');
    await tf.nextFrame();
    state.model?.dispose();
    state.tokenizer = nextTokenizer;
    state.model = new LocalTransformerLM(preset, state.tokenizer);
    state.activeConfigId = preset.id;
    state.trainedSteps = 0;
    state.lastLoss = null;
    refs.progressLabel.textContent = `${preset.name} initialized.`;
    refs.lossLabel.textContent = 'loss: —';
    updateStatus(`${preset.name} ready: ${formatParams(state.model.parameterCount())} real trainable parameters allocated.`, 'ok');
    addSystemMessage(`${preset.name} initialized with ${formatParams(state.model.parameterCount())} trainable parameters and vocab size ${state.tokenizer.vocabSize}.`);
    renderModelGrid();
    updateSelectedDetails();
    refreshRuntimeStats();
    return true;
  } catch (error) {
    console.error(error);
    updateStatus(`Initialization failed: ${error.message}`, 'error');
    state.model = null;
    state.activeConfigId = null;
    return false;
  } finally {
    refs.initModel.disabled = state.isTraining;
  }
}

function updateStatus(message, type = 'info') {
  refs.trainMetrics.textContent = message;
  refs.trainMetrics.dataset.type = type;
}

async function startTraining() {
  if (state.isTraining) return;
  const corpus = getTrainingCorpus();
  if (corpus.length < 20) {
    updateStatus('Paste more training text first. A transformer needs data, not hardcoded answers.', 'warn');
    return;
  }

  if (!state.model) {
    const initialized = await initializeSelectedModel({ quiet: true });
    if (!initialized) return;
  }

  const missing = unknownCharacters(corpus, state.tokenizer);
  if (missing.length) {
    const ok = window.confirm(`Your corpus contains characters outside the current tokenizer: ${missing.join(' ')}. Reinitialize so the model can learn them? Current weights will reset.`);
    if (ok) {
      const initialized = await initializeSelectedModel();
      if (!initialized) return;
    }
  }

  const encoded = state.tokenizer.encode(corpus);
  const steps = Math.max(1, Number(refs.stepsInput.value || 1));
  const batchSize = Math.max(1, Number(refs.batchInput.value || state.selectedPreset.defaultBatch || 1));
  const learningRate = Math.max(0.000001, Number(refs.lrInput.value || state.selectedPreset.defaultLr || 0.001));
  const clipNorm = Math.max(0, Number(refs.clipInput.value || 0));
  const weightDecay = Math.max(0, Number(refs.decayInput.value || 0));
  const reportEvery = Math.max(1, Number(refs.reportInput.value || 5));

  state.isTraining = true;
  state.abortTraining = false;
  setTrainState('Training', true);
  refs.trainProgress.value = 0;
  updateStatus('Training started. Loss should trend down as the model learns your text.', 'busy');

  const startedAt = performance.now();
  let tokenCounter = 0;
  let stoppedForNumericSafety = false;

  try {
    for (let step = 1; step <= steps; step += 1) {
      if (state.abortTraining) break;
      const { xs, ys, sequenceLength } = makeBatch(encoded, state.model.config.context, batchSize);
      const stepStarted = performance.now();
      const result = await state.model.trainStep(xs, ys, { learningRate, clipNorm, weightDecay });
      xs.dispose();
      ys.dispose();

      if (result.skipped || !Number.isFinite(result.loss) || !Number.isFinite(result.gradNorm)) {
        state.abortTraining = true;
        stoppedForNumericSafety = true;
        refs.progressLabel.textContent = `Stopped safely at step ${step}/${steps}.`;
        refs.lossLabel.textContent = 'loss: non-finite';
        updateStatus('A non-finite loss/gradient was detected BEFORE applying it, so the weights were not poisoned. Lower the learning rate or batch size, then resume or reinitialize.', 'error');
        addSystemMessage('Training stopped safely because a NaN/Infinity was detected before it could corrupt the model.');
        break;
      }

      tokenCounter += batchSize * sequenceLength;
      state.trainedSteps += 1;
      state.lastLoss = result.loss;

      if (step === 1 || step % reportEvery === 0 || step === steps || result.hadBadGradients) {
        const elapsedSeconds = Math.max(0.001, (performance.now() - startedAt) / 1000);
        const instantSeconds = Math.max(0.001, (performance.now() - stepStarted) / 1000);
        refs.trainProgress.value = (step / steps) * 100;
        refs.progressLabel.textContent = `step ${step}/${steps} · ${Math.round(tokenCounter / elapsedSeconds)} tok/s avg · ${Math.round((batchSize * sequenceLength) / instantSeconds)} tok/s step`;
        refs.lossLabel.textContent = `loss: ${result.loss.toFixed(4)} · ppl: ${result.perplexity.toFixed(1)}`;
        const badGradNote = result.hadBadGradients ? ' · bad grad entries zeroed' : '';
        updateStatus(`grad norm ${result.gradNorm.toFixed(3)}${badGradNote} · backend ${tf.getBackend()} · tensors ${tf.memory().numTensors} · ${bytesToHuman(tf.memory().numBytes)}`, result.hadBadGradients ? 'warn' : 'busy');
        refreshRuntimeStats();
      }
      if (step % 10 === 0) await tf.nextFrame();
    }

    if (state.abortTraining) {
      if (!stoppedForNumericSafety) {
        updateStatus('Training stopped by user. Current weights are kept.', 'warn');
        refs.progressLabel.textContent = 'Training stopped.';
      }
    } else {
      updateStatus('Training complete. Chat now uses the weights you trained.', 'ok');
      refs.progressLabel.textContent = `Finished ${steps} steps.`;
      addSystemMessage(`Training finished: ${steps} steps, last loss ${state.lastLoss?.toFixed(4) ?? 'n/a'}.`);
    }
  } catch (error) {
    console.error(error);
    updateStatus(`Training failed: ${error.message}`, 'error');
    addSystemMessage(`Training error: ${error.message}`);
  } finally {
    state.isTraining = false;
    state.abortTraining = false;
    setTrainState('Idle', false);
    refreshRuntimeStats();
  }
}

function appendMessage(role, text = '') {
  const emptySystem = refs.chatMessages.querySelector('.system-message');
  if (emptySystem && refs.chatMessages.children.length === 1) emptySystem.remove();
  const message = document.createElement('div');
  message.className = `message ${role}`;
  const label = document.createElement('span');
  label.className = 'role-label';
  label.textContent = role === 'user' ? 'You' : role === 'assistant' ? 'V0idGPT' : 'System';
  const body = document.createElement('div');
  body.className = 'message-body';
  body.textContent = text;
  message.append(label, body);
  refs.chatMessages.appendChild(message);
  refs.chatMessages.scrollTop = refs.chatMessages.scrollHeight;
  return body;
}

function addSystemMessage(text) {
  appendMessage('system', text);
}

async function sendChat() {
  if (state.generationAbort) {
    state.generationAbort.abort();
    refs.sendChat.textContent = 'Stopping…';
    return;
  }

  const input = refs.chatInput.value.trim();
  if (!input) return;
  if (!state.model) {
    const initialized = await initializeSelectedModel({ quiet: true });
    if (!initialized) return;
    addSystemMessage('Model is initialized but not trained yet. Output will be mostly random until you train on text.');
  }

  appendMessage('user', input);
  refs.chatInput.value = '';
  const assistantBody = appendMessage('assistant', '');
  const prompt = `${state.chatLog}User: ${input}\nAssistant:`;
  const abort = new AbortController();
  state.generationAbort = abort;
  refs.sendChat.textContent = 'Stop';

  const startedAt = performance.now();
  try {
    const generated = await state.model.generate(prompt, {
      maxNewTokens: Number(refs.maxTokensInput.value || 180),
      temperature: Number(refs.tempInput.value || 0.85),
      topK: Number(refs.topKInput.value || 40),
      topP: Number(refs.topPInput.value || 0.92),
      repetitionPenalty: Number(refs.repInput.value || 1.08),
      streamEvery: 3,
      yieldEvery: 6,
      maxRepeat: 8,
      signal: abort.signal,
    }, (_piece, fullText) => {
      assistantBody.textContent = fullText;
      refs.chatMessages.scrollTop = refs.chatMessages.scrollHeight;
    });

    const seconds = Math.max(0.001, (performance.now() - startedAt) / 1000);
    assistantBody.textContent = generated || '(stopped)';
    state.chatLog = `${prompt}${generated}\n`;
    updateStatus(`Generated ${state.tokenizer.encode(generated).length} tokens in ${seconds.toFixed(1)}s.`, 'ok');
    if (refs.autoTrainChat.checked && generated.trim()) {
      refs.trainText.value = `${refs.trainText.value}${refs.trainText.value.trim() ? '\n\n' : ''}User: ${input}\nAssistant: ${generated}`;
    }
    refreshRuntimeStats();
  } catch (error) {
    console.error(error);
    if (String(error.message).includes('non-finite')) {
      const stats = await repairWeights({ silent: true });
      assistantBody.textContent = `Generation stopped because corrupted NaN/Inf weights were detected. I repaired ${stats.nonFinite + stats.clipped} bad values. Try generating again; if it still looks broken, reinitialize and train with a lower learning rate.`;
    } else {
      assistantBody.textContent = `Generation failed: ${error.message}`;
    }
  } finally {
    state.generationAbort = null;
    refs.sendChat.textContent = 'Generate';
  }
}

async function repairWeights({ silent = false } = {}) {
  if (!state.model) return { nonFinite: 0, clipped: 0, repairedTensors: 0 };
  updateStatus('Scanning and repairing NaN/Inf weights…', 'busy');
  const stats = await state.model.sanitizeWeights();
  const repaired = stats.nonFinite + stats.clipped;
  const message = repaired
    ? `Repaired ${stats.nonFinite} NaN/Inf values and clipped ${stats.clipped} runaway values across ${stats.repairedTensors} tensors.`
    : `Weights look clean: ${stats.valuesChecked.toLocaleString()} finite values checked.`;
  updateStatus(message, repaired ? 'warn' : 'ok');
  if (!silent) addSystemMessage(message);
  refreshRuntimeStats();
  return stats;
}

async function exportModel() {
  if (!state.model) return;
  const params = state.model.parameterCount();
  if (params > 10_000_000) {
    const ok = window.confirm(`Exporting ${formatParams(params)} parameters to JSON can be huge and slow. Continue?`);
    if (!ok) return;
  }
  updateStatus('Repair-checking and serializing compact base64 weights…', 'busy');
  const payload = await state.model.exportJSON();
  const blob = new Blob([JSON.stringify(payload)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = `${state.model.config.id ?? 'v0idgpt'}-${Date.now()}.json`;
  anchor.click();
  URL.revokeObjectURL(url);
  const repaired = payload.exportStats.nonFinite + payload.exportStats.clipped;
  updateStatus(repaired ? `Exported compact JSON after repairing ${repaired} bad values. Weight arrays are encoded, so NaN cannot stringify into null.` : 'Exported compact JSON with encoded finite weights.', repaired ? 'warn' : 'ok');
}

async function importModel(file) {
  if (!file) return;
  try {
    updateStatus('Reading model JSON…', 'busy');
    const json = JSON.parse(await file.text());
    if (!['v0idgpt-local-transformer-v1', 'v0idgpt-local-transformer-v2'].includes(json.format)) {
      throw new Error('Not a V0idGPT Reborn local transformer export.');
    }
    const tokenizer = CharTokenizer.fromJSON(json.tokenizer);
    const params = estimateParams(json.config, tokenizer.vocabSize);
    if (currentModelWouldBeUnsafe(json.config, params)) {
      updateStatus(`Import locked: ${formatParams(params)} parameters. Tick high-RAM unlock first.`, 'warn');
      return;
    }
    state.model?.dispose();
    state.tokenizer = tokenizer;
    state.model = new LocalTransformerLM(json.config, tokenizer);
    const importStats = await state.model.importWeights(json.weights);
    state.activeConfigId = json.config.id;
    const matchingPreset = MODEL_PRESETS.find((preset) => preset.id === json.config.id);
    if (matchingPreset) state.selectedPreset = matchingPreset;
    const repairNote = importStats.repairedValues ? ` Repaired ${importStats.repairedValues} invalid imported values.` : '';
    updateStatus(`Imported ${formatParams(state.model.parameterCount())} trainable parameters.${repairNote}`, importStats.repairedValues ? 'warn' : 'ok');
    addSystemMessage(`Imported model ${json.config.name ?? json.config.id}.${repairNote}`);
    renderModelGrid();
    updateSelectedDetails();
    refreshRuntimeStats();
  } catch (error) {
    console.error(error);
    updateStatus(`Import failed: ${error.message}`, 'error');
  } finally {
    refs.importModel.value = '';
  }
}

refs.initModel.addEventListener('click', () => initializeSelectedModel());
refs.recalcButton.addEventListener('click', () => {
  renderModelGrid();
  updateSelectedDetails();
  refreshRuntimeStats();
});
refs.unlockLarge.addEventListener('change', updateSelectedDetails);
refs.batchInput.addEventListener('input', updateSelectedDetails);
refs.demoCorpus.addEventListener('click', () => {
  refs.trainText.value = `${refs.trainText.value}${refs.trainText.value.trim() ? '\n\n' : ''}${DEMO_CORPUS}`;
  updateSelectedDetails();
});
refs.appendChat.addEventListener('click', () => {
  if (!state.chatLog.trim()) {
    updateStatus('No chat log to append yet.', 'warn');
    return;
  }
  refs.trainText.value = `${refs.trainText.value}${refs.trainText.value.trim() ? '\n\n' : ''}${state.chatLog.trim()}`;
});
refs.clearCorpus.addEventListener('click', () => {
  if (window.confirm('Clear the training text box?')) refs.trainText.value = '';
});
refs.trainText.addEventListener('input', updateSelectedDetails);
refs.startTrain.addEventListener('click', startTraining);
refs.stopTrain.addEventListener('click', () => {
  state.abortTraining = true;
});
refs.sendChat.addEventListener('click', sendChat);
refs.chatInput.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && (event.metaKey || event.ctrlKey)) sendChat();
});
refs.clearChat.addEventListener('click', () => {
  state.chatLog = '';
  refs.chatMessages.innerHTML = '<div class="system-message">Chat cleared. The model weights are unchanged.</div>';
});
refs.repairWeights.addEventListener('click', () => repairWeights());
refs.exportModel.addEventListener('click', exportModel);
refs.importModel.addEventListener('change', (event) => importModel(event.target.files?.[0]));

renderModelGrid();
updateSelectedDetails();
setTrainState('Idle', false);
setupBackend();
window.setInterval(refreshRuntimeStats, 2500);
