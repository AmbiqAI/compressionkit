import { defineConfig } from "astro/config";
import { unified } from "@astrojs/markdown-remark";
import starlight from "@astrojs/starlight";
import react from "@astrojs/react";
import { heliaStarlight } from "@ambiqai/helia-ui/starlight";
import rehypeMermaid from "rehype-mermaid";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import redirects from "./src/data/redirects.json" with { type: "json" };
import { sections } from "./src/navigation.mjs";
import { fileURLToPath } from "node:url";
export default defineConfig({
  vite: {
    plugins: [
      {
        name: "compressionkit-section-membership",
        enforce: "pre",
        resolveId(source, importer) {
          // Historical page URLs do not share the section's URL prefix.
          if (
            source === "./sections" &&
            importer?.includes("/@ambiqai/helia-ui/starlight/")
          )
            return fileURLToPath(
              new URL("./src/section-matcher.ts", import.meta.url),
            );
        },
      },
    ],
  },
  site: "https://ambiqai.github.io",
  base: "/compressionkit",
  redirects: {
    ...redirects,
    "/api": "/compressionkit/reference/",
  },
  markdown: {
    processor: unified({
      remarkPlugins: [remarkMath],
      rehypePlugins: [rehypeKatex, [rehypeMermaid, { strategy: "inline-svg" }]],
    }),
  },
  integrations: [
    react(),
    starlight({
      components: { Hero: "./src/components/HomeHero.astro" },
      title: "compressionKIT",
      description: "Signal compression development kit for edge and wearable devices.",
      favicon: "/assets/compressionkit-icon.png",
      expressiveCode: {
        styleOverrides: {
          frames: { editorBackground: "var(--helia-surface-card-muted)" },
        },
      },
      customCss: [
        "./src/styles/site.css",
        "@ambiqai/helia-ui/mermaid.css",
        "katex/dist/katex.min.css",
      ],
      plugins: [
        heliaStarlight({
          accent: "kit-compression",
          sections,
          sidebar: "always",
          header: {
            title: "compressionKIT",
            titleRegularPrefix: "compression",
            hub: {
              label: "HELIA",
              href: "https://ambiqai.github.io/helia-developer-hub/",
            },
          },
          discoverability: {
            markdown: true,
            llms: true,
            jsonLd: true,
            ogImage: true,
          },
          footer: {
            logo: "ambiq",
            tagline: "Part of the Ambiq HELIA AI platform",
            links: [
              { label: "compressionKIT source on GitHub", href: "https://github.com/AmbiqAI/compressionkit" },
            ],
          },
        }),
      ],
    }),
  ],
});
