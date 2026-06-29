"""
Standalone Playwright renderer — called as a subprocess by app.py so that
Playwright's Chromium launch runs in a fresh process with its own event loop,
avoiding the asyncio thread-pool SelectorEventLoop limitation on Windows.

Usage: python _playwright_render.py <html_file> <out_png> <width> <height>
"""
import sys
from playwright.sync_api import sync_playwright

html_path, out_path, w, h = sys.argv[1:]
with open(html_path, encoding="utf-8") as f:
    content = f.read()

with sync_playwright() as pw:
    browser = pw.chromium.launch()
    page = browser.new_page(viewport={"width": int(w), "height": int(h)})
    page.set_content(content)
    page.wait_for_timeout(200)
    page.screenshot(path=out_path, full_page=True)
    browser.close()
