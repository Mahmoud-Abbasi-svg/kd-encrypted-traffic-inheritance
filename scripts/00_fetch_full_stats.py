"""Extract the per-service statistics files from the full CESNET-TLS-Year22 archive.

The Zenodo archive is ~30 GB, but it contains small JSON files with flow counts per
service (per day and per week). This script reads the archive's central directory and
those JSON members through HTTP range requests, so nothing large is downloaded.

Standard library only; works on Windows and Linux.

Usage:
    python scripts/00_fetch_full_stats.py            # list + extract all JSON members
    python scripts/00_fetch_full_stats.py --list-only
"""

import argparse
import csv
import io
import os
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

ZIP_URL = "https://zenodo.org/records/10608607/files/CESNET-TLS-Year22.zip?download=1"
DEFAULT_DATA_ROOT = Path(os.environ.get("KD_DATA_ROOT", "C:/datasets"))
MAX_MEMBER_BYTES = 20 * 1024 * 1024  # skip anything that is not a small metadata file


class HTTPRangeFile(io.RawIOBase):
    """Read-only, seekable file object backed by HTTP range requests."""

    def __init__(self, url: str, read_ahead: int = 1 << 20, retries: int = 5):
        self.url = url
        self.read_ahead = read_ahead
        self.retries = retries
        self.pos = 0
        self.requests = 0
        self._cache_start = 0
        self._cache = b""
        headers = self._request(0, 0)[1]
        content_range = headers.get("Content-Range")  # case-insensitive lookup
        if content_range is None:
            raise IOError(f"No Content-Range header in response; headers: {dict(headers)}")
        self.size = int(content_range.rsplit("/", 1)[1])

    def _request(self, start: int, end: int):
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}"})
        for attempt in range(self.retries):
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    if resp.status != 206:
                        raise IOError(f"Server ignored range request (HTTP {resp.status})")
                    self.requests += 1
                    return resp.read(), resp.headers
            except (OSError, IOError) as exc:
                if attempt == self.retries - 1:
                    raise
                wait = 2**attempt
                print(f"  range {start}-{end} failed ({exc}); retrying in {wait}s", file=sys.stderr)
                time.sleep(wait)
        raise AssertionError("unreachable")

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        elif whence == io.SEEK_END:
            self.pos = self.size + offset
        return self.pos

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = self.size - self.pos
        if n == 0 or self.pos >= self.size:
            return b""
        end = min(self.pos + n, self.size)
        cache_end = self._cache_start + len(self._cache)
        if not (self._cache_start <= self.pos and end <= cache_end):
            fetch_end = min(max(end, self.pos + self.read_ahead), self.size)
            self._cache = self._request(self.pos, fetch_end - 1)[0]
            self._cache_start = self.pos
        offset = self.pos - self._cache_start
        data = self._cache[offset : offset + (end - self.pos)]
        self.pos += len(data)
        return data

    def readinto(self, buffer) -> int:
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DEFAULT_DATA_ROOT / "CESNET-TLS-Year22" / "stats_full")
    parser.add_argument("--list-only", action="store_true", help="only write the archive manifest")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    remote = HTTPRangeFile(ZIP_URL)
    print(f"Archive size: {remote.size / 1e9:.2f} GB")
    with zipfile.ZipFile(io.BufferedReader(remote, buffer_size=1 << 16)) as archive:
        members = archive.infolist()
        manifest = args.out / "archive_manifest.csv"
        with manifest.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["name", "compressed_bytes", "uncompressed_bytes"])
            for m in members:
                writer.writerow([m.filename, m.compress_size, m.file_size])
        json_members = [m for m in members if m.filename.lower().endswith(".json") and not m.is_dir()]
        print(f"Members: {len(members)} total, {len(json_members)} JSON; manifest -> {manifest}")

        basenames: dict[str, int] = {}
        for m in json_members:
            name = Path(m.filename).name
            if name.startswith("stats-") and name[6:14].isdigit():
                key = "stats-YYYYMMDD.json (daily)"
            elif name.startswith("stats"):
                key = name
            else:
                key = "<other>.json"
            basenames[key] = basenames.get(key, 0) + 1
        for key, count in sorted(basenames.items()):
            print(f"  {key}: {count}")

        if args.list_only:
            return

        extracted = skipped = 0
        for i, m in enumerate(json_members, 1):
            if m.file_size > MAX_MEMBER_BYTES:
                skipped += 1
                continue
            target = args.out / m.filename
            if target.exists() and target.stat().st_size == m.file_size:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(m))
            extracted += 1
            if i % 50 == 0:
                print(f"  {i}/{len(json_members)} JSON members processed ({remote.requests} range requests)")
        print(f"Extracted {extracted} JSON files, skipped {skipped} large ones -> {args.out}")
        print(f"Total range requests: {remote.requests}")


if __name__ == "__main__":
    main()
