# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A personal blog built with **Hugo Extended v0.160.1** using the **hugo-book** theme, deployed to GitHub Pages at `https://maxwell60701.github.io/`. Content is written in Chinese. The blog focuses on RAG/LLM technical topics.

## Common Commands

```bash
# Local development with live reload
hugo server

# Production build (matches CI)
hugo build --gc --minify

# Create a new blog post
hugo new posts/my-post.md
```

No test suite exists for this project.

## Architecture

### Active Theme

**hugo-book** (`themes/hugo-book`, git submodule from `alex-shpak/hugo-book`). Two other submodules exist (`ananke`, `PaperMod`) but are unused — they are remnants from prior theme migrations.

### Theme Overrides

The project heavily overrides hugo-book templates in `layouts/`. Key overrides:

- **`layouts/index.html`** — Custom homepage listing posts with previews and "阅读全文" links; replaces the TOC sidebar with a post list
- **`layouts/posts/single.html`** — Custom single-post layout with prev/next navigation; restores TOC sidebar for the current page
- **`layouts/partials/docs/menu.html`** — Custom sidebar menu with brand logo and file-tree navigation
- **`layouts/partials/extend_head.html`** and **`layouts/_partials/docs/inject/head.html`** — Inject Google Fonts (Inter, JetBrains Mono, Noto Sans SC, Noto Serif SC) and Lucide Icons
- **`layouts/partials/extend_footer.html`** — Initializes Lucide icons
- **`layouts/shortcodes/code.html`** — Custom code block shortcode with language labels, copy-to-clipboard, and Catppuccin Mocha styling
- **`layouts/shortcodes/architecture.html`** — Wrapper shortcode for architecture diagrams

Some partials (`home_info.html`, `index_profile.html`, `post_meta.html`) retain PaperMod-style class names from the previous theme — they still work but are legacy artifacts.

### Dual CSS Pipeline

Custom CSS exists in **two locations** that should be kept in sync:
1. `assets/css/extended/blank.css` — Processed by Hugo Pipes (allows Sass/PostCSS)
2. `static/css/extended/blank.css` — Served as-is

Both are 894-line files with identical content. When editing styles, update both or consolidate to one path.

### Content Structure

- **Single section**: `content/posts/` (enforced by `BookSection = 'posts'` in `hugo.toml`)
- **Front matter**: TOML format (`+++` delimiters)
- **Images**: Organized by topic under `assets/img/` (e.g., `civiRag/`, `docker/`, `sft/`)

### Hugo Config Highlights (`hugo.toml`)

- `markup.goldmark.renderer.unsafe = true` — Raw HTML is allowed in Markdown
- `markup.highlight.style = "monokai"` — Syntax highlighting theme
- `markup.highlight.noClasses = false` — Uses CSS classes for highlighting (required by custom code shortcode)

## Git Workflow

- **`main`** — Production branch, auto-deployed to GitHub Pages via `.github/workflows/hugo.yaml`
- **`develop`** — Active development branch
- PRs should target `main` for deployment

## CI/CD

`.github/workflows/hugo.yaml` builds on push to `main` using:
- Dart Sass 1.99.0, Go 1.26.1, Hugo Extended 0.160.1, Node.js 24.14.1
- `hugo build --gc --minify`
- Deploys via `actions/deploy-pages@v5`

## Utility Scripts

- `generate_flowchart.py` — Generates flowchart/diagram images (e.g., `mermaid-architecture.png`) used in blog posts