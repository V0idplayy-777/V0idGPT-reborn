# V0idGPT Reborn

A no-API, local transformer chatbot/training UI. It builds a decoder-only GPT-style model in the browser with TensorFlow.js, trains from text you paste using a hybrid character/subword tokenizer, then generates replies autoregressively from the trained weights.

## What it includes

- 10 selectable transformer architectures from ~80K parameters to ~1B parameters.
- Attention-grabbing warnings on large/high-RAM models.
- Real learned token embeddings, positional embeddings, causal multi-head self-attention, layer norm, GELU feed-forward blocks, and an LM head.
- Local training with Adam, gradient clipping, AdamW-style decay, random batched context windows, and TensorFlow.js WebGL acceleration when available.
- Chat UI with top-k/nucleus sampling, temperature, repetition penalty, and streaming token output.
- Import/export for trained weights as JSON.

> The 1B model is a real architecture option, but browser training at that scale needs huge RAM/VRAM and can crash a normal tab. Smaller presets are intended for quick in-browser use.

## Use it without running anything

The app is GitHub Pages compatible and the ready-to-serve static build is committed in `docs/`. Configure Pages to serve from the `docs/` folder, then use it entirely from the browser UI: paste training text, initialize a model, train, and chat. No user-side `npm` command or external AI API is required.

## Developer commands

Only needed if you want to modify/rebuild the site locally:

```bash
npm install
npm run dev
npm run build
```

`npm run build` writes the Pages-ready build to `docs/` with relative asset paths and `.nojekyll`.
