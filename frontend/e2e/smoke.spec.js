import { test, expect } from '@playwright/test';

test('overview loads published data, preserving research-only context', async ({ page }) => {
  await page.goto('/index.html');
  await expect(page.getByRole('heading', { name: /Market sentiment, in context/i })).toBeVisible();
  await expect(page.getByText('MARKET SENTIMENT INDEX')).toBeVisible();
  await expect(page.getByText('FROZEN v2.1 RESEARCH SIGNAL')).toBeVisible();
  await expect(page.locator('body')).not.toContainText('Published data is unavailable');
});

test('explorer filters history and labels descriptive outcomes', async ({ page, isMobile }) => {
  await page.goto('/index.html');
  if (isMobile) await page.getByRole('button', {name:'Open menu'}).click();
  await page.getByRole('button', {name:'Historical Explorer'}).click();
  await expect(page.getByRole('heading', { name: 'Explore the evidence.' })).toBeVisible();
  await expect(page.getByRole('table')).toBeVisible();
  await page.getByPlaceholder('Date, action, regime...').fill('no such observation 999999');
  await expect(page.getByText('No observations match these filters.')).toBeVisible();
  await page.getByRole('button', {name:'Reset filters'}).click();
  await expect(page.getByRole('table').locator('tbody tr').first()).toBeVisible();
});

test('research panel cannot imply an approved champion', async ({ page, isMobile }) => {
  await page.goto('/index.html');
  if (isMobile) await page.getByRole('button', {name:'Open menu'}).click();
  await page.getByRole('button', {name:'Research Lab'}).click();
  await expect(page.getByRole('heading', {name:'Experimental, by design.'})).toBeVisible();
  await expect(page.getByText('Not deployed for decisions')).toBeVisible();
  await expect(page.getByText('Production action changed')).toBeVisible();
  await expect(page.getByText('Champion selected')).toBeVisible();
});

test('same-site production contract: no separate domain, relative bundles, and existing evidence URLs', async ({ page, isMobile }) => {
  const response = await page.goto('/index.html');
  expect(response?.status()).toBe(200);
  const root = page.locator('#root');
  await expect(root).toBeVisible();
  await expect(page.locator('meta[name="robots"]')).toHaveAttribute('content','noindex,nofollow');
  const scripts = await page.locator('script[src]').evaluateAll(els => els.map(el => el.getAttribute('src')));
  expect(scripts.length).toBeGreaterThan(0);
  expect(scripts.every(src => src.startsWith('./assets/'))).toBe(true);
  // The published analysis link lives in Strategy Evidence, not Overview.
  if (isMobile) await page.getByRole('button', { name: 'Open menu' }).click();
  await page.getByRole('button', { name: 'Strategy Evidence' }).click();
  await expect(page.getByRole('heading', { name: 'Signals with their caveats.' })).toBeVisible();
  await expect(page.locator('a[href="./analysis.json"]').first()).toBeVisible();
  const data = await page.request.get('/analysis.json');
  expect(data.status()).toBe(200);
  const json = await data.json();
  expect(json.latest.signal_date).toMatch(/^\d{4}-\d{2}-\d{2}$/);
});
