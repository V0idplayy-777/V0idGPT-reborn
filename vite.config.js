import { defineConfig } from 'vite';

export default defineConfig({
  // Relative asset URLs make the built app work under GitHub Pages project paths
  // like https://user.github.io/V0idGPT-reborn/ instead of only at domain root.
  base: './',
  server: {
    host: '0.0.0.0',
    allowedHosts: true,
  },
  preview: {
    host: '0.0.0.0',
    allowedHosts: true,
  },
  build: {
    outDir: 'docs',
    emptyOutDir: true,
    assetsDir: 'assets',
  },
});
