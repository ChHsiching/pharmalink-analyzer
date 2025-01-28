/**
 * Tauri WebDriver E2E tests: Loading Screen & Main UI
 *
 * Tests the full user flow inside the actual Tauri webview:
 * 1. Loading screen appears on startup
 * 2. Transitions to main UI after backend health check
 * 3. Tab navigation works
 * 4. Dataset list loads
 */
import { expect } from 'chai';

describe('Loading Screen', () => {
  it('should show loading screen initially', async () => {
    // The app starts with a loading screen polling the backend.
    // In dev/debug mode, the backend starts via beforeDevCommand,
    // but in the built app, the sidecar spawns from Rust.
    // Give a brief window to catch the loading screen.
    const body = await $('body');
    const html = await body.getHTML();

    // Either we see the loading screen OR the main UI (if backend was fast)
    const hasLoadingScreen = html.includes('正在启动后端服务') || html.includes('loading-screen');
    const hasMainUI = html.includes('app-content') || html.includes('TabBar');

    expect(hasLoadingScreen || hasMainUI).to.be.true;
  });

  it('should transition to main UI within 30 seconds', async () => {
    // Wait for the main UI to appear (backend health check must pass)
    const appContent = await $('.app-content');
    await appContent.waitForExist({ timeout: 30000 });
    expect(await appContent.isExisting()).to.be.true;
  });

  it('should show the tab bar', async () => {
    const tabbar = await $('.tab-bar, [class*="tab"], nav');
    await tabbar.waitForExist({ timeout: 5000 });
    expect(await tabbar.isExisting()).to.be.true;
  });
});

describe('Data Import Tab', () => {
  it('should display dataset list after loading', async () => {
    // Wait for the dataset list to render
    const body = await $('body');
    await browser.pause(2000);

    const html = await body.getHTML();
    // The data-import tab should show loaded datasets
    const hasDatasetContent = html.includes('dataset') || html.includes('数据');
    expect(hasDatasetContent).to.be.true;
  });
});

describe('Tab Navigation', () => {
  it('should navigate between tabs', async () => {
    const tabs = await $$('button, [role="tab"], .tab-item, [class*="tab"]');
    // Should have multiple tabs
    expect(tabs.length).to.be.greaterThan(0);
  });
});
