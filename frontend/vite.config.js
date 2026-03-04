import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  // GitHub Pages deploys to /<repo-name>/ subpath
  // Update this to match your repo name
  base: "/statcast-bayesian-pitch-model/",
  build: {
    outDir: "dist",
    sourcemap: false,
  },
});
