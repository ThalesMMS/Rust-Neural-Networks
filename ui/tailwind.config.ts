import type { Config } from "tailwindcss";

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        canvas: "#0b0b10",
        surface: "#13131b",
        surface2: "#191924",
        border: "#26263380",
        accent: {
          DEFAULT: "#8166ff",
          soft: "#a996ff",
          dim: "#4b3aa8",
        },
      },
      fontFamily: {
        mono: ["ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
      },
    },
  },
  plugins: [],
} satisfies Config;
