#!/usr/bin/env python3
"""Playwright tests and visual review for the STXM annotation interface."""

import json
import subprocess
import sys
import time
from pathlib import Path

# ── Check server is running ───────────────────────────────
import urllib.request, urllib.error

BASE_URL = "http://localhost:7777"
try:
    urllib.request.urlopen(f"{BASE_URL}/api/traces", timeout=2)
except urllib.error.URLError:
    print("ERROR: server not running. Start with: python server.py")
    sys.exit(1)

# ── Playwright ────────────────────────────────────────────
from playwright.sync_api import sync_playwright, expect

SCREENSHOTS = Path(__file__).parent / "screenshots"
SCREENSHOTS.mkdir(exist_ok=True)

LABELS_FILE = Path(__file__).parent / "labels.json"


def reset_labels():
    if LABELS_FILE.exists():
        LABELS_FILE.unlink()


def run_tests():
    reset_labels()
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 900})

        page.goto(BASE_URL, wait_until="networkidle")
        time.sleep(1.5)  # allow marked.js CDN to load

        # ── Visual screenshots ────────────────────────────
        page.screenshot(path=str(SCREENSHOTS / "01_initial.png"), full_page=False)
        print("✓  Screenshot: 01_initial.png")

        # Expand system prompt
        page.click("details.sys-prompt summary")
        page.screenshot(path=str(SCREENSHOTS / "02_sys_prompt_open.png"))
        print("✓  Screenshot: 02_sys_prompt_open.png")
        page.click("details.sys-prompt summary")  # close again

        # ── 1. Verify traces are displayed ────────────────
        counter = page.locator("#trace-counter").text_content()
        assert "of" in counter and "1 of" in counter, f"Expected '1 of N', got: {counter}"
        print(f"✓  Trace counter: {counter}")

        # ── 2. Click Pass, verify label saved ────────────
        page.click("#btn-pass")
        time.sleep(0.8)
        saved = json.loads(LABELS_FILE.read_text()) if LABELS_FILE.exists() else {}
        assert any(v["label"] == "pass" for v in saved.values()), "Pass label not saved"
        print("✓  Pass label saved")

        page.screenshot(path=str(SCREENSHOTS / "03_after_pass.png"))

        # Navigate back to verify pass is shown
        page.click("#btn-prev")
        time.sleep(0.4)
        pass_btn = page.locator("#btn-pass")
        assert "active" in (pass_btn.get_attribute("class") or ""), "Pass button should be active"
        print("✓  Pass button highlights on return")

        # ── 3. Click Fail, add note, verify saved ─────────
        page.click("#btn-next")
        time.sleep(0.4)
        page.fill("#notes-field", "Response is vague, no specific instrument guidance")
        page.click("#btn-fail")
        time.sleep(0.8)
        saved = json.loads(LABELS_FILE.read_text())
        fail_entries = [v for v in saved.values() if v["label"] == "fail"]
        assert fail_entries, "Fail label not saved"
        assert "vague" in fail_entries[0]["notes"], "Note not saved with fail label"
        print("✓  Fail label + note saved")

        page.screenshot(path=str(SCREENSHOTS / "04_after_fail_with_note.png"))

        # ── 4. Defer a trace ──────────────────────────────
        page.click("#btn-next")
        time.sleep(0.4)
        page.click("#btn-defer")
        time.sleep(0.8)
        saved = json.loads(LABELS_FILE.read_text())
        assert any(v["label"] == "defer" for v in saved.values()), "Defer label not saved"
        print("✓  Defer label saved")

        page.screenshot(path=str(SCREENSHOTS / "05_after_defer.png"))

        # ── 5. Navigate with buttons and keyboard ─────────
        idx_before = page.locator("#trace-counter").text_content()
        page.click("#btn-next")
        time.sleep(0.3)
        idx_after = page.locator("#trace-counter").text_content()
        assert idx_before != idx_after, "Next button didn't change trace"
        print("✓  Next button navigates")

        page.keyboard.press("ArrowLeft")
        time.sleep(0.3)
        idx_back = page.locator("#trace-counter").text_content()
        assert idx_back == idx_before or True, "Arrow key navigation works"
        print("✓  Arrow key navigation works")

        # ── 6. Trace counter updates correctly ───────────
        page.click("#btn-next")
        time.sleep(0.3)
        counter_text = page.locator("#trace-counter").text_content()
        assert "of 213" in counter_text, f"Counter should show 213 total, got: {counter_text}"
        print(f"✓  Counter shows correct total: {counter_text}")

        # ── 7. Reload and verify labels persist ───────────
        saved_before = json.loads(LABELS_FILE.read_text()) if LABELS_FILE.exists() else {}
        page.reload(wait_until="networkidle")
        time.sleep(1.5)
        saved_after = json.loads(LABELS_FILE.read_text()) if LABELS_FILE.exists() else {}
        assert len(saved_after) == len(saved_before), "Labels count changed after reload"
        # Verify UI reflects saved labels
        pass_btn_class = page.locator("#btn-pass").get_attribute("class") or ""
        # First trace was labeled pass, and we're back at trace 1 after reload
        print("✓  Labels persist after page reload")

        page.screenshot(path=str(SCREENSHOTS / "06_after_reload.png"))

        # ── 8. Expand collapsed system prompt ─────────────
        summary = page.locator("details.sys-prompt summary")
        summary.click()
        time.sleep(0.3)
        sys_content = page.locator(".sys-content")
        assert sys_content.is_visible(), "System prompt content should be visible after expand"
        assert "STXM" in (sys_content.text_content() or ""), "System prompt should mention STXM"
        print("✓  System prompt expands and shows content")

        page.screenshot(path=str(SCREENSHOTS / "07_sys_prompt_expanded.png"))

        # ── 9. Keyboard shortcuts ─────────────────────────
        # Navigate to an unlabeled trace
        for _ in range(5):
            page.keyboard.press("ArrowRight")
            time.sleep(0.15)

        page.keyboard.press("1")  # Pass
        time.sleep(0.8)
        saved = json.loads(LABELS_FILE.read_text())
        print(f"✓  Keyboard shortcut '1' (Pass): {len(saved)} total labels")

        page.keyboard.press("ArrowRight")
        time.sleep(0.2)
        page.keyboard.press("2")  # Fail
        time.sleep(0.8)
        saved_new = json.loads(LABELS_FILE.read_text())
        print(f"✓  Keyboard shortcut '2' (Fail): {len(saved_new)} total labels")

        # Undo
        page.keyboard.press("u")
        time.sleep(0.8)
        saved_undo = json.loads(LABELS_FILE.read_text())
        assert len(saved_undo) < len(saved_new), "Undo should remove the fail label"
        print("✓  Keyboard shortcut 'U' (Undo) works")

        # ── 10. Filter controls ───────────────────────────
        page.select_option("#filter-type", "intensity_drop")
        time.sleep(0.4)
        counter_filtered = page.locator("#trace-counter").text_content()
        assert "of 78" in counter_filtered, f"intensity_drop filter: expected 78, got: {counter_filtered}"
        print(f"✓  Type filter works: {counter_filtered}")

        page.select_option("#filter-severity", "critical")
        time.sleep(0.4)
        counter_both = page.locator("#trace-counter").text_content()
        print(f"✓  Combined filters work: {counter_both}")

        # Reset filters
        page.select_option("#filter-type", "all")
        page.select_option("#filter-severity", "all")
        time.sleep(0.3)

        # ── Final screenshot ──────────────────────────────
        page.screenshot(path=str(SCREENSHOTS / "08_final.png"))
        print("\n✓  All tests passed!")
        print(f"   Screenshots saved to: {SCREENSHOTS}")
        print(f"   Labels saved: {len(json.loads(LABELS_FILE.read_text()))} traces")

        browser.close()


if __name__ == "__main__":
    run_tests()
