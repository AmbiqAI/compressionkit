import { chromium } from '@playwright/test';
import { globSync, readFileSync } from 'node:fs';

const base = process.env.DOCS_AUDIT_URL || 'http://127.0.0.1:8778';
const routes = globSync('dist/**/*.html')
  .filter(file => !file.endsWith('404.html'))
  .filter(file => !readFileSync(file, 'utf8').includes('http-equiv="refresh"'))
  .map(file => '/compressionkit/' + file.slice(5).replace(/index.html$/, ''));
const browser = await chromium.launch({ headless: true });
const failures = [];
let checked = 0;
try {
  for (const width of [390, 1440]) {
    for (const theme of ['light', 'dark']) {
      const page = await browser.newPage({ viewport: { width, height: 900 }, colorScheme: theme });
      await page.addInitScript(theme => localStorage.setItem('starlight-theme', theme), theme);
      let resourceFailures = [];
      page.on('response', response => {
        if (response.status() >= 400) resourceFailures.push({ url: response.url(), status: response.status() });
      });
      page.on('requestfailed', request => {
        resourceFailures.push({ url: request.url(), error: request.failure()?.errorText });
      });
      for (const route of routes) {
        resourceFailures = [];
        const response = await page.goto(base + route, { waitUntil: 'load' });
        const issues = await page.evaluate(() => {
          const main = document.querySelector('main');
          const visible = element => element.getBoundingClientRect().height > 0;
          const probe = document.createElement('span');
          probe.style.color = 'var(--helia-ink-primary)';
          document.body.append(probe);
          const buttonColor = getComputedStyle(probe).color;
          probe.remove();
          return {
            overflow: document.documentElement.scrollWidth > innerWidth + 1,
            images: [...main.querySelectorAll('img')].filter(image => visible(image) && (!image.complete || !image.naturalWidth)).map(image => image.getAttribute('src')),
            headings: main.querySelectorAll('h1').length,
            emptyLinks: [...main.querySelectorAll('a')].filter(link => visible(link) && !link.textContent.trim() && !link.getAttribute('aria-label') && !link.querySelector('img[alt],svg')).map(link => link.getAttribute('href')),
            codeCards: [...main.querySelectorAll('.helia-card')].filter(card => card.querySelector('pre')).length,
            unreadableButtons: [...main.querySelectorAll('.task-links a')].filter(link => getComputedStyle(link).color !== buttonColor).length,
          };
        });
        checked++;
        if (resourceFailures.length || response.status() !== 200 || issues.overflow || issues.images.length || issues.headings !== 1 || issues.emptyLinks.length || issues.codeCards || issues.unreadableButtons) {
          failures.push({ width, theme, route, status: response.status(), resourceFailures, ...issues });
        }
        if (route === '/compressionkit/') {
          const card = page.locator('.kit-home-nav .helia-card').first();
          const destination = await card.locator('a').first().getAttribute('href');
          const description = card.locator('.helia-card-content p').first();
          await description.scrollIntoViewIfNeeded();
          const body = await description.boundingBox();
          await page.mouse.click(body.x + body.width / 2, body.y + body.height / 2);
          await page.waitForURL(url => url.pathname === destination);
          if (new URL(page.url()).pathname !== destination) failures.push({ width, theme, route, cardDestination: page.url(), expected: destination });
        }
      }
      await page.close();
    }
  }
} finally {
  await browser.close();
}
console.log(JSON.stringify({ routes: routes.length, checked, failures }, null, 2));
if (failures.length) process.exitCode = 1;
