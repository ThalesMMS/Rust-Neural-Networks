import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
// Tauri expects a fixed port and to ignore src-tauri file changes.
export default defineConfig({
    plugins: [react()],
    clearScreen: false,
    server: {
        port: 1420,
        strictPort: true,
        watch: {
            ignored: ["**/src-tauri/**"],
        },
    },
});
