# GitHub Pages (this folder)

This directory is the **static site root** for the project’s GitHub Pages landing page for end users (product-style homepage inspired by `prj-lumina/docs`).

## Enable Pages

1. In the GitHub repository, go to **Settings → Pages** (under *Code and automation*).
2. Under **Build and deployment → Source**, choose **Deploy from a branch**.
3. Select branch **`main`** and folder **`/docs`**, then save.
4. After the site builds, open:

   `https://<your-username-or-org>.github.io/<repository-name>/`

   For example, if the repo is `McK-Playground/prj-echo`, the URL is:

   `https://mck-playground.github.io/prj-echo/`

## Local preview

From the repository root:

```bash
npx --yes serve docs
```

Then open the URL printed in the terminal. Paths use `./` so they resolve correctly under a repository subpath on Pages.

## Files

- `index.html` — landing page (positions Echo as a **live meeting assistant**; hero, experience, features, privacy, install)
- `styles.css` — layout, theme tokens, terminal mock visuals, scroll-reveal styles
- `main.js` — scroll progress bar, nav scrolled state, hero parallax, theme toggle persistence, `IntersectionObserver` reveal (skipped when `prefers-reduced-motion: reduce`)
- `assets/echo-icon.svg` — app icon for the nav and favicon
- `.nojekyll` — disables Jekyll so GitHub serves files as-is
