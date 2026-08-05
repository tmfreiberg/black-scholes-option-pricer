"""Browser smoke test for the published document.

Skipped unless `BSM_SITE_DIR` points at a built site, so a local `pytest` run is
unaffected. In CI it runs against the artifact the deploy job is about to publish, and
it gates that deploy.

What it checks is the one thing nothing else can: that a reader arriving at the page
gets a working runtime. Everything upstream verifies that the Python is correct; this
verifies that the Python reaches the browser at all — Pyodide boots, micropip resolves
the package from PyPI, and a live cell produces the expected number. A rendered page can
look perfect and be completely inert, and the failure that taught us this was silent: a
package that never installed surfaced three cells later as an undefined name.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Iterator
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

SITE_DIR = os.environ.get("BSM_SITE_DIR")

pytestmark = pytest.mark.skipif(
    SITE_DIR is None, reason="set BSM_SITE_DIR to a built site to run"
)

# Pyodide downloads roughly 10 MB and then installs the wheel, so the first cell is slow
# by the standards of a browser test and fast by the standards of the alternative.
RUNTIME_TIMEOUT_MS = 180_000


@pytest.fixture(scope="module")
def site_url() -> Iterator[str]:
    """Serve the built site on a loopback port for the duration of the module."""
    directory = Path(SITE_DIR or ".").resolve()
    handler = partial(SimpleHTTPRequestHandler, directory=str(directory))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/"
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture
def page(site_url: str) -> Iterator[object]:
    """A Chromium page with the document loaded."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        context = browser.new_page()
        context.goto(site_url, wait_until="networkidle")
        yield context
        browser.close()


class TestStaticContent:
    """These pass without the runtime, and must: the page has to be readable at once."""

    def test_title_renders(self, page: object) -> None:
        assert "Black-Scholes" in page.title()  # type: ignore[attr-defined]

    def test_figures_are_present(self, page: object) -> None:
        images = page.locator("img").all()  # type: ignore[attr-defined]
        sources = [image.get_attribute("src") or "" for image in images]
        assert any("curves" in source for source in sources)
        assert any("heatmap" in source for source in sources)

    def test_figures_actually_loaded(self, page: object) -> None:
        """A broken image tag renders as an icon and reads as a figure to a scraper."""
        broken = page.evaluate(  # type: ignore[attr-defined]
            "Array.from(document.images).filter(i => !i.complete "
            "|| i.naturalWidth === 0).length"
        )
        assert broken == 0

    def test_the_install_cell_is_visible(self, page: object) -> None:
        """The install is shown, not hidden, so that its failure is visible too."""
        content = page.content()  # type: ignore[attr-defined]
        assert "micropip.install" in content
        assert "bsm-pricer==" in content


class TestLiveRuntime:
    def test_a_cell_runs_and_prices_correctly(self, page: object) -> None:
        """The end of the chain: browser, runtime, wheel, package, right answer."""
        run_button = page.locator("button:has-text('Run')").first  # type: ignore[attr-defined]
        run_button.wait_for(state="visible", timeout=RUNTIME_TIMEOUT_MS)
        run_button.click()

        output = page.locator(".qwebr-output-code-area").first  # type: ignore[attr-defined]
        output.wait_for(timeout=RUNTIME_TIMEOUT_MS)

        text = output.inner_text()
        assert "10.45" in text, f"expected the textbook price, got: {text!r}"

    def test_the_smile_cell_runs(self, page: object) -> None:
        """The deepest path on the page: merton, through implied, through roots."""
        buttons = page.locator("button:has-text('Run')")  # type: ignore[attr-defined]
        buttons.first.wait_for(state="visible", timeout=RUNTIME_TIMEOUT_MS)
        buttons.last.click()

        output = page.locator(".qwebr-output-code-area").last  # type: ignore[attr-defined]
        output.wait_for(timeout=RUNTIME_TIMEOUT_MS)
        assert output.inner_text().strip() != ""

    def test_no_console_errors(self, site_url: str) -> None:
        """A failed micropip install surfaces here and nowhere else on the page."""
        from playwright.sync_api import sync_playwright

        errors: list[str] = []
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            context = browser.new_page()
            context.on(
                "console",
                lambda message: (
                    errors.append(message.text) if message.type == "error" else None
                ),
            )
            context.goto(site_url, wait_until="networkidle")
            context.wait_for_timeout(15_000)
            browser.close()

        fatal = [
            error
            for error in errors
            if "micropip" in error or "404" in error or "ModuleNotFoundError" in error
        ]
        assert not fatal, f"console errors: {fatal}"
