import { defineConfig, devices } from '@playwright/test';
export default defineConfig({
  testDir: './e2e',
  timeout: 30000,
  expect: { timeout: 15000 },
  use: { baseURL: 'http://127.0.0.1:4174', trace: 'retain-on-failure', screenshot: 'only-on-failure' },
  projects: [{ name: 'chromium', use: { ...devices['Desktop Chrome'] } }, { name: 'mobile-chromium', use: { ...devices['iPhone 13'], defaultBrowserType: 'chromium' } }],
  webServer: { command: 'python3 -m http.server 4174 --bind 127.0.0.1 --directory ../site', port: 4174, reuseExistingServer: !process.env.CI, timeout: 20000 },
  reporter: [['list']],
});
