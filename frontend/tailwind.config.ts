import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./lib/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // "Deep Science Dark Mode" palette
        void: {
          DEFAULT: "#000000",
          900: "#05060a",
          800: "#0a0c12",
          700: "#101319",
          600: "#161a22",
        },
        graphite: {
          900: "#1a1f29",
          700: "#2a3140",
          500: "#4a5469",
          300: "#7c8597",
          200: "#a3acbd",
        },
        cyan: {
          glow: "#00f0ff",
          neon: "#22e2ff",
          dim: "#0aa0b8",
          deep: "#063b46",
        },
        amber: {
          warn: "#ffb347",
        },
        crimson: {
          err: "#ff4d6d",
        },
      },
      fontFamily: {
        mono: [
          "JetBrains Mono",
          "Fira Code",
          "ui-monospace",
          "SFMono-Regular",
          "Menlo",
          "monospace",
        ],
      },
      boxShadow: {
        neon: "0 0 12px rgba(34, 226, 255, 0.45)",
        "neon-soft": "0 0 24px rgba(34, 226, 255, 0.18)",
      },
      backgroundImage: {
        "grid-faint":
          "linear-gradient(to right, rgba(34,226,255,0.06) 1px, transparent 1px), linear-gradient(to bottom, rgba(34,226,255,0.06) 1px, transparent 1px)",
      },
    },
  },
  plugins: [],
};

export default config;
