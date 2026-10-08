
"""Sync Flexsin sitemap pages into the PostgreSQL-backed Ask Me store.

This module reconstructs the public content feeds from Flexsin's live
sitemap/pages because the legacy *_api.php endpoints currently return HTML
instead of JSON.

Supported source types:
- case_study
- mobile_case_study
- job
"""

from __future__ import annotations

import html
import logging
import re
import time
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse

import ingest
from vectorstore import VectorStore

log = logging.getLogger("askme.sync")

BASE_URL = "https://www.flexsin.com/"
SITEMAP_URL = urljoin(BASE_URL, "sitemap.xml")

REQUEST_TIMEOUT = 30
USER_AGENT = "AskMe-Flexsin-Sync/1.0"

PROJECT_TARGETS = {
    "businessNeedsBg",
    "midCon",
    "businessOutcome",
}

JOB_TARGETS = {
    "jobDetLeft",
}

VOID_TAGS = {
    "area",
    "base",
    "br",
    "col",
    "embed",
    "hr",
    "img",
    "input",
    "link",
    "meta",
    "param",
    "source",
    "track",
    "wbr",
}


def _normalise_space(text: str) -> str:
    """Collapse HTML whitespace into readable text."""
    text = html.unescape(text or "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _remove_repeated_heading(text: str, heading: str) -> str:
    """Remove a heading that is already present at the start of a section."""
    text = text.strip()
    heading = heading.strip()

    if not text or not heading:
        return text

    # Remove the heading once if the HTML content already contains it.
    if text.lower().startswith(heading.lower()):
        text = text[len(heading):].strip()

    return text


class PageParser(HTMLParser):
    """Extract specific content sections from Flexsin HTML pages.

    We deliberately track actual open HTML tags instead of using a numeric
    depth counter. Void tags such as <br> and <img> do not have closing tags,
    so a numeric depth counter becomes inaccurate on these pages.
    """

    def __init__(self, source_type: str):
        super().__init__(convert_charrefs=True)

        self.source_type = source_type
        self.sections: list[dict[str, str]] = []

        self.open_tags: list[str] = []

        self.active: dict | None = None

        self.title_parts: list[str] = []

        self.in_title = False

        self.skip_depth = 0

    @staticmethod
    def _classes(attrs) -> set[str]:
        values = {
            value
            for key, value in attrs
            if key == "class" and value
        }

        classes: set[str] = set()

        for value in values:
            classes.update(value.split())

        return classes

    def _target_classes(self) -> set[str]:
        if self.source_type in {"case_study", "mobile_case_study"}:
            return PROJECT_TARGETS

        if self.source_type == "job":
            return JOB_TARGETS

        return set()

    def _is_target(self, classes: set[str]) -> str | None:
        if self.source_type in {"case_study", "mobile_case_study"}:
            # The generic "midCon" class occurs many times on the page.
            # Only "midCon pfDetContent" is the Strategy & Solution section.
            if "businessNeedsBg" in classes:
                return "businessNeedsBg"

            if "businessOutcome" in classes:
                return "businessOutcome"

            if "midCon" in classes and "pfDetContent" in classes:
                return "midCon"

            return None

        if self.source_type == "job":
            if "jobDetLeft" in classes:
                return "jobDetLeft"

        return None

    @staticmethod
    def _should_skip(classes: set[str]) -> bool:
        """Ignore testimonial/form sections."""
        if "clientSpeak" in classes:
            return True

        if "reqFormCon" in classes:
            return True

        return False

    def handle_starttag(self, tag, attrs):
        tag_lower = tag.lower()

        classes = self._classes(attrs)

        # Track page title separately.
        if tag_lower == "title":
            self.in_title = True

        # Skip testimonial/form areas completely.
        if self.skip_depth:
            if tag_lower not in VOID_TAGS:
                self.skip_depth += 1
            return

        if self._should_skip(classes):
            if tag_lower not in VOID_TAGS:
                self.skip_depth = 1
            return

        # Start a target section only when we are not already collecting one.
        target_class = self._is_target(classes)

        if target_class and self.active is None:
            current_index = len(self.open_tags)

            self.active = {
                "class": target_class,
                "tag": tag_lower,
                "stack_index": current_index,
                "parts": [],
            }

        if self.active is not None:
            if tag_lower in {
                "p",
                "div",
                "section",
                "article",
                "h2",
                "h3",
                "h4",
                "li",
                "br",
            }:
                self.active["parts"].append("\n")

        # Void tags have no matching end tag and therefore must not be pushed.
        if tag_lower not in VOID_TAGS:
            self.open_tags.append(tag_lower)

    def handle_endtag(self, tag):
        tag_lower = tag.lower()

        if self.in_title and tag_lower == "title":
            self.in_title = False

        if self.skip_depth:
            if tag_lower not in VOID_TAGS:
                self.skip_depth -= 1
            return

        if self.active is not None:
            target = self.active

            if (
                tag_lower == target["tag"]
                and len(self.open_tags) - 1 == target["stack_index"]
            ):
                text = _normalise_space("".join(target["parts"]))

                if text:
                    self.sections.append(
                        {
                            "class": target["class"],
                            "text": text,
                        }
                    )

                self.active = None

        if tag_lower in self.open_tags:
            if self.open_tags[-1] == tag_lower:
                self.open_tags.pop()
            else:
                index = len(self.open_tags) - 1 - self.open_tags[::-1].index(
                    tag_lower
                )
                del self.open_tags[index:]

    def handle_data(self, data):
        if self.in_title:
            self.title_parts.append(data)

        if self.skip_depth:
            return

        if self.active is not None:
            self.active["parts"].append(data)


def extract_page(
    html_text: str,
    source_type: str,
) -> dict[str, str] | None:
    """Extract title + useful page content from a Flexsin page."""

    parser = PageParser(source_type)
    parser.feed(html_text)
    parser.close()

    title = _normalise_space(" ".join(parser.title_parts))

    if source_type in {"case_study", "mobile_case_study"}:
        required = {
            "businessNeedsBg",
            "midCon",
            "businessOutcome",
        }

        found = {
            section["class"]
            for section in parser.sections
        }

        if not required.issubset(found):
            log.warning(
                "Could not extract all required sections for %s. "
                "Required=%s Found=%s",
                source_type,
                required,
                found,
            )
            return None

        heading_map = {
            "businessNeedsBg": "Business Needs",
            "midCon": "Strategy & Solution",
            "businessOutcome": "Business Outcomes",
        }

        parts = []

        for section in parser.sections:
            section_class = section["class"]

            if section_class not in heading_map:
                continue

            heading = heading_map[section_class]

            section_text = _remove_repeated_heading(
                section["text"],
                heading,
            )

            if not section_text:
                continue

            parts.append(
                f"{heading}\n"
                f"{section_text}"
            )

        text = "\n\n".join(parts).strip()

        if not text:
            return None

        return {
            "title": title,
            "text": text,
        }

    if source_type == "job":
        if not any(
            section["class"] == "jobDetLeft"
            for section in parser.sections
        ):
            return None

        parts = [
            section["text"]
            for section in parser.sections
            if section["class"] == "jobDetLeft"
        ]

        text = "\n\n".join(parts).strip()

        if not text:
            return None

        return {
            "title": title,
            "text": text,
        }

    return None


def fetch_url(url: str) -> str:
    """Fetch a URL using urllib so sync does not require requests."""
    import urllib.request

    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": (
                "text/html,application/xhtml+xml,"
                "application/xml;q=0.9,*/*;q=0.8"
            ),
        },
    )

    with urllib.request.urlopen(
        request,
        timeout=REQUEST_TIMEOUT,
    ) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        return response.read().decode(charset, errors="replace")


def normalise_url(url: str) -> str:
    """Convert sitemap-relative URLs into canonical absolute URLs."""
    url = html.unescape((url or "").strip())

    if not url:
        return ""

    if url.startswith("//"):
        url = "https:" + url
    elif url.startswith("/"):
        url = urljoin(BASE_URL, url)
    elif not url.startswith(("http://", "https://")):
        url = urljoin(BASE_URL, url)

    parsed = urlparse(url)

    if parsed.netloc.lower() != "www.flexsin.com":
        return ""

    path = parsed.path or "/"

    if path != "/":
        path = path.rstrip("/") + "/"

    return (
        f"https://www.flexsin.com{path}"
        + (f"?{parsed.query}" if parsed.query else "")
    )


def feed_key_for(source_type: str) -> str:
    """Return the legacy feed identifier for a source type."""
    return {
        "case_study": "project_casestudy_api.php",
        "mobile_case_study": "mobile_casestudy_api.php",
        "job": "jobpost_api.php",
    }.get(source_type, source_type)


def classify_url(url: str) -> str | None:
    """Classify supported Flexsin sitemap URLs."""
    parsed = urlparse(url)
    path = parsed.path.lower()

    if path.startswith("/portfolio/"):
        return "case_study"

    if path.startswith("/mobile-portfolio/"):
        return "mobile_case_study"

    if path.startswith("/careers/"):
        return "job"

    return None


class SitemapParser(HTMLParser):
    """Small XML parser for <loc> values."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.in_loc = False
        self.parts: list[str] = []
        self.urls: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() == "loc":
            self.in_loc = True
            self.parts = []

    def handle_endtag(self, tag):
        if tag.lower() == "loc" and self.in_loc:
            value = _normalise_space("".join(self.parts))
            if value:
                self.urls.append(value)
            self.in_loc = False

    def handle_data(self, data):
        if self.in_loc:
            self.parts.append(data)


def parse_sitemap(xml_text: str) -> list[str]:
    parser = SitemapParser()
    parser.feed(xml_text)
    parser.close()
    return parser.urls


def discover_sitemaps(root_xml: str) -> list[str]:
    """Return sitemap URLs referenced by the sitemap index."""
    urls = parse_sitemap(root_xml)

    result = []

    for url in urls:
        normalised = normalise_url(url)

        if normalised:
            result.append(normalised)

    return list(dict.fromkeys(result))


def discover_content_urls() -> dict[str, list[str]]:
    """Discover project and mobile URLs from sitemap-portfolio.xml."""
    root_xml = fetch_url(SITEMAP_URL)

    sitemap_urls = discover_sitemaps(root_xml)

    candidates: dict[str, list[str]] = {
        "case_study": [],
        "mobile_case_study": [],
    }

    for sitemap_url in sitemap_urls:
        if "sitemap-portfolio.xml" not in sitemap_url:
            continue

        try:
            xml = fetch_url(sitemap_url)
        except Exception as exc:
            log.warning(
                "Could not fetch sitemap %s: %s",
                sitemap_url,
                exc,
            )
            continue

        for raw_url in parse_sitemap(xml):
            url = normalise_url(raw_url)
            source_type = classify_url(url)

            if source_type in candidates:
                candidates[source_type].append(url)

    for key in candidates:
        candidates[key] = list(dict.fromkeys(candidates[key]))

    return candidates


def discover_job_urls() -> list[str]:
    """Discover career pages from Flexsin sitemaps."""
    root_xml = fetch_url(SITEMAP_URL)

    sitemap_urls = discover_sitemaps(root_xml)

    jobs: list[str] = []

    for sitemap_url in sitemap_urls:
        try:
            xml = fetch_url(sitemap_url)
        except Exception as exc:
            log.warning(
                "Could not fetch sitemap %s: %s",
                sitemap_url,
                exc,
            )
            continue

        for raw_url in parse_sitemap(xml):
            url = normalise_url(raw_url)

            if classify_url(url) == "job":
                jobs.append(url)

    return list(dict.fromkeys(jobs))


def save_source(
    store: VectorStore,
    *,
    source_type: str,
    url: str,
    title: str,
    text: str,
) -> int:
    """Save one page as a source and replace its chunks."""
    chunks = ingest.chunk(text)

    if not chunks:
        raise ValueError("Page produced no chunks.")

    source_id = store.upsert_source(
        title=title,
        source_type=source_type,
        origin="feed",
        url=url,
        feed_key=feed_key_for(source_type),
        synced_at=time.time(),
    )

    store.replace_source_chunks(
        source_id=source_id,
        chunks=chunks,
    )

    return len(chunks)


def build_candidates() -> list[tuple[str, str]]:
    """Build all feed candidates."""
    candidates: list[tuple[str, str]] = []

    content = discover_content_urls()

    for source_type in (
        "case_study",
        "mobile_case_study",
    ):
        for url in content.get(source_type, []):
            candidates.append((url, source_type))

    for url in discover_job_urls():
        candidates.append((url, "job"))

    return candidates


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    candidates = build_candidates()

    print(f"Candidates: {len(candidates)}")

    store = VectorStore()

    successful = 0
    skipped = 0
    failed = 0

    for index, (url, source_type) in enumerate(candidates, start=1):
        print(
            f"[{index}/{len(candidates)}] "
            f"{source_type}: {url}"
        )

        try:
            page_html = fetch_url(url)

            page = extract_page(
                page_html,
                source_type,
            )

            if not page:
                skipped += 1
                print("  -> skipped: content extraction failed")
                continue

            chunk_count = save_source(
                store,
                source_type=source_type,
                url=url,
                title=page["title"],
                text=page["text"],
            )

            successful += 1

            print(
                f"  -> synced: {chunk_count} chunks"
            )

        except Exception as exc:
            failed += 1

            log.exception(
                "Failed to sync %s",
                url,
            )

            print(
                f"  -> failed: {type(exc).__name__}: {exc}"
            )

    print()
    print("=" * 60)
    print("SYNC COMPLETE")
    print("=" * 60)
    print(f"Successful: {successful}")
    print(f"Skipped:    {skipped}")
    print(f"Failed:     {failed}")


if __name__ == "__main__":
    main()

