"""Browser smoke test for the published document.

Skipped unless `BSM_SITE_DIR` points at a built site, so a local `pytest` run is
unaffected. In CI it runs against the artifact the deploy job is about to publish.

What it checks is the one thing nothing else can: that a reader arriving at the page
gets a working runtime. Everything upstream verifies that the Python is correct; this
verifies that the Python reaches the browser at all. It is a regression guard rather
than a proof that any particular build works: a Quarto or quarto-live upgrade can leave
the render succeeding and the page inert, and nothing else in the suite would notice.

Selectors
---------
The class names below were read off a rendered page rather than inferred, and they are
not guessable from the source. quarto-live emits no `button` elements at all; the run
control carries `exercise-editor-btn-run-code`. Cell source lives in CodeMirror, which
builds the editor in JavaScript after load, so the code text is absent from
`page.content()` at `networkidle` and appears only once `.cm-line` elements exist. Read
text with `inner_text`, not `content`, and wait for the editors first.

If a quarto-live upgrade breaks these tests, re-read the DOM before adjusting them.
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

# Pyodide downloads roughly 10 MB, then micropip resolves the package from PyPI. Slow
# by the standards of a browser test, fast by the standards of the alternative.
RUNTIME_TIMEOUT_MS = 180_000

# Read from a rendered page. See the module docstring before changing either.
EDITOR_LINE = ".cm-line"
RUN_BUTTON = ".exercise-editor-btn-run-code"
CELL = ".exercise-cell"
OUTPUT = ".cell-output-pyodide"


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
    """A Chromium page with the document loaded and its editors instantiated."""
    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        context = browser.new_page()
        context.goto(site_url, wait_until="networkidle")
        # The editors are built after load. Nothing that reads cell text or looks for a
        # run control can proceed until at least one line exists.
        context.locator(EDITOR_LINE).first.wait_for(timeout=RUNTIME_TIMEOUT_MS)
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
        assert any("smile" in source for source in sources)

    def test_figures_actually_loaded(self, page: object) -> None:
        """A broken image tag renders as an icon and reads as a figure to a scraper."""
        broken = page.evaluate(  # type: ignore[attr-defined]
            "Array.from(document.images).filter(i => !i.complete "
            "|| i.naturalWidth === 0).length"
        )
        assert broken == 0


class TestEditors:
    def test_every_cell_became_an_editor(self, page: object) -> None:
        """Nine live cells in the document; each should instantiate."""
        assert page.locator(CELL).count() >= 9  # type: ignore[attr-defined]

    def test_the_install_is_visible(self, page: object) -> None:
        """The install is shown, not hidden, so that its failure is visible too.

        Checked through `inner_text` because the source lives in CodeMirror and never
        reaches the served HTML. Terms are asserted separately: syntax highlighting
        splits a line across spans, so `micropip.install` may not survive as one string.
        """
        body = page.inner_text("body")  # type: ignore[attr-defined]
        assert "micropip" in body
        assert "bsm-pricer" in body


class TestLiveRuntime:
    def test_a_cell_runs_and_prices_correctly(self, page: object) -> None:
        """The end of the chain: browser, runtime, package from PyPI, right answer."""
        cell = page.locator(CELL, has_text="price(contract").first  # type: ignore[attr-defined]
        run = cell.locator(RUN_BUTTON).first
        run.wait_for(state="visible", timeout=RUNTIME_TIMEOUT_MS)
        run.click()

        # Poll the document rather than one container: which element receives the result
        # is an implementation detail, the number appearing at all is not.
        page.wait_for_function(  # type: ignore[attr-defined]
            "() => document.body.innerText.includes('10.45')",
            timeout=RUNTIME_TIMEOUT_MS,
        )

    def test_the_smile_cell_runs(self, page: object) -> None:
        """The deepest path on the page: merton, through implied, through roots."""
        cell = page.locator(CELL, has_text="np.round(100 * volatility_smile").first  # type: ignore[attr-defined]
        run = cell.locator(RUN_BUTTON).first
        run.wait_for(state="visible", timeout=RUNTIME_TIMEOUT_MS)
        run.click()

        output = cell.locator(OUTPUT).first
        output.wait_for(timeout=RUNTIME_TIMEOUT_MS)
        assert output.inner_text().strip() != ""

    def test_no_fatal_console_errors(self, site_url: str) -> None:
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
            context.wait_for_timeout(45_000)
            browser.close()

        fatal = [
            error
            for error in errors
            if "micropip" in error
            or "404" in error
            or "ModuleNotFoundError" in error
            or "Failed to fetch" in error
        ]
        assert not fatal, f"console errors: {fatal}"
