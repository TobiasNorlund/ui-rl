import os
import contextlib
import asyncio
import pandas as pd
from playwright.async_api import Playwright, async_playwright


class SimpleDataEntrySession:

    def __init__(self, display: int = 1):
        self._display = display

    async def start(self):
        self._playwright = await async_playwright().start()

        # Open data sheet browser in left half of the primary screen
        self._data_sheet_browser = DataSheetBrowser(
            self._playwright,
            self._display,
        )
        await self._data_sheet_browser.start()
        # Move to left side
        os.system("xdotool key Super_L+Left")

        # Open form browser in right half of the screen
        self._form_browser = FormBrowser(
            self._playwright,
            self._display
        )
        await self._form_browser.start()
        # Move to right
        os.system("xdotool key Super_L+Right")

    async def stop(self):
        # Close Playwright browser/resources.
        with contextlib.suppress(Exception):
            if self._data_sheet_browser is not None:
                await self._data_sheet_browser.stop()
        with contextlib.suppress(Exception):
            if self._form_browser is not None:
                await self._form_browser.stop()
        with contextlib.suppress(Exception):
            if self._playwright is not None:
                await self._playwright.stop()
        
        self._playwright = None
        self._data_sheet_browser = None
        self._form_browser = None

    def get_progress(self):
        return {
            "submitted_row_indices": self._form_browser.submitted_row_indices, 
            "num_incorrect_submissions": self._form_browser.num_incorrect_submissions
        }


class DataSheetBrowser:

    def __init__(self, playwright: Playwright, display: int):
        self._playwright = playwright
        self._display = display
        self._browser = None
        self._context = None
        self._page = None

    async def start(self):
        self._browser = await self._playwright.chromium.launch(
            headless=False,
            env={
                'DISPLAY': f':{self._display}',
            }
        )
        self._context = await self._browser.new_context(no_viewport=True)
        self._page = await self._context.new_page()
        await self._page.goto("https://docs.google.com/spreadsheets/d/1wVwDmyx01J5_XSzdgkmxvOOHPLmZmsnq1YJ636XkwIA")

    async def stop(self):
        await self._browser.close()
        self._page = None
        self._context = None
        self._browser = None


class FormBrowser:
    def __init__(self, playwright: Playwright, display: int):
        self._sde_data = pd.read_csv(os.path.join(os.path.dirname(__file__), "data.csv"))
        self._playwright = playwright
        self._display = display
        self._browser = None
        self._context = None
        self._page = None

        self.submitted_row_indices = []
        self.num_incorrect_submissions = 0

    async def start(self):
        self._browser = await self._playwright.chromium.launch(
            headless=False,
            env={
                'DISPLAY': f':{self._display}',
            }
        )
        self._context = await self._browser.new_context(no_viewport=True)
        self._page = await self._context.new_page()
        await self._page.goto("https://docs.google.com/forms/d/e/1FAIpQLSef9VSfp3ISD7jr5Kgxq2UDibrT82vUEilN8vIrhCIfH5YfQQ/viewform")
        self._page.on("request", self._on_form_browser_request)

    async def stop(self):
        await self._browser.close()
        self._page = None
        self._context = None
        self._browser = None

    def _on_form_browser_request(self, request):
        if request.method == "POST" and request.is_navigation_request() and "entry.1444058590" in request.post_data_json:
            f_name = request.post_data_json["entry.1444058590"]
            l_name = request.post_data_json["entry.277017468"]
            email = request.post_data_json["entry.564455939"]
            matching = self._sde_data[(self._sde_data["First name"] == f_name) & (self._sde_data["Last name"] == l_name) & (self._sde_data["Email"] == email)]
            if len(matching) > 0:
                self.submitted_row_indices.append(int(matching.index[0]))
            else:
                self.num_incorrect_submissions += 1


if __name__ == "__main__":
    import asyncio
    async def main():
        session = SimpleDataEntrySession(display=1)
        print("Session starting...")
        await session.start()
        print("Session started")
        await asyncio.sleep(2)
        img = await session.screenshot()
        img.save("screenshot.png")
        await asyncio.sleep(30)
        await session.stop()
        print("Session stopped")

    asyncio.run(main())