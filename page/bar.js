/* Top bar for a project page: name back to the portfolio, GitHub link to this
   repo. Vendored copy; the portfolio site no longer serves shared page files.
   Usage: <aside class="sidebar"></aside> ... <script src="page/bar.js" data-repo="name"></script> */
(function () {
  const repo = document.currentScript.dataset.repo;
  document.documentElement.setAttribute('data-theme', 'dark');
  const el = document.querySelector('aside.sidebar');
  if (!el) return;
  el.innerHTML = `
    <a class="bar-wordmark" href="https://maltysnack.github.io/">
      <img src="https://github.com/maltysnack.png" alt="" class="bar-avatar" />
      <span class="bar-name">Milo Watkinson</span>
    </a>
    <nav class="bar-nav" aria-label="Sections"><a href="https://maltysnack.github.io/">Projects</a></nav>
    <a class="bar-icon" href="https://github.com/maltysnack/${repo}" target="_blank" rel="noopener" aria-label="Source on GitHub">
      <svg class="icon" viewBox="0 0 24 24" fill="currentColor"><path d="M12 2C6.477 2 2 6.477 2 12c0 4.418 2.865 8.166 6.839 9.489.5.092.682-.217.682-.482 0-.237-.009-.868-.014-1.703-2.782.604-3.369-1.342-3.369-1.342-.454-1.154-1.11-1.462-1.11-1.462-.907-.62.069-.608.069-.608 1.003.07 1.531 1.03 1.531 1.03.892 1.529 2.341 1.087 2.912.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.11-4.555-4.943 0-1.091.39-1.984 1.029-2.683-.103-.253-.446-1.27.098-2.647 0 0 .84-.269 2.75 1.025A9.578 9.578 0 0112 6.836a9.59 9.59 0 012.504.337c1.909-1.294 2.747-1.025 2.747-1.025.546 1.377.203 2.394.1 2.647.64.699 1.028 1.592 1.028 2.683 0 3.842-2.339 4.687-4.566 4.935.359.309.678.919.678 1.852 0 1.336-.012 2.415-.012 2.744 0 .267.18.578.688.48C19.138 20.163 22 16.418 22 12c0-5.523-4.477-10-10-10z"/></svg>
    </a>`;
})();
