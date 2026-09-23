"""Upload the deposit archives to an existing Zenodo draft, one file at a time, resumably.

The web form sends a multi-gigabyte file in a single request with no resume, which fails often on a
loaded service. This uploads through the API instead: each file is registered, streamed, and
committed on its own, and a file that is already committed on the draft is skipped, so re-running
after a failure continues rather than starting again.

The token is read from the ZENODO_TOKEN environment variable and is never written anywhere. Create
one at https://zenodo.org/account/settings/applications/tokens/new/ with the scopes `deposit:write`
and `deposit:actions`.

    $env:ZENODO_TOKEN = "..."                       # PowerShell, this session only
    python scripts/22_zenodo_upload.py --record 22916038
    python scripts/22_zenodo_upload.py --record 22916038 --list   # what the draft holds now

Publishing stays manual: this script never calls the publish endpoint, because a published record's
files cannot be changed.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

BASE = "https://zenodo.org/api"
UPLOAD_ORDER = ["README.md", "MANIFEST.csv", "checkpoints.zip", "scores-featurespace.zip",
                "scores-confirmatory.zip", "scores-exploratory.zip"]


def human(size: float) -> str:
    return f"{size / 1e9:.2f} GB" if size >= 1e9 else f"{size / 1e6:.1f} MB"


class Draft:
    """The files of one Zenodo draft, through whichever API version the record uses.

    Records made in the current interface expose `/records/<id>/draft/files`; older deposits expose a
    bucket URL instead. Both are handled, because which one a draft uses depends on when and how it
    was created, not on anything the caller controls.
    """

    def __init__(self, record: str, token: str):
        self.record, self.session = record, requests.Session()
        self.session.headers["Authorization"] = f"Bearer {token}"
        response = self.session.get(f"{BASE}/records/{record}/draft", timeout=60)
        # 404 means the draft predates the current interface. A 5xx means the newer endpoint could not
        # serialise it, which happens when an interrupted browser upload leaves a file entry with no
        # size and no checksum; the older endpoint renders the same draft and is worth trying before
        # giving up.
        if response.status_code == 404 or response.status_code >= 500:
            first = response.status_code
            response = self.session.get(f"{BASE}/deposit/depositions/{record}", timeout=60)
            if first >= 500 and response.ok:
                print(f"note: the draft endpoint answered HTTP {first}; using the older deposit API.")
        # A bad token and a draft belonging to someone else both answer 401/403, and a raw traceback
        # says neither. The three causes below are the ones that actually happen.
        if response.status_code in (401, 403):
            raise SystemExit(
                f"Zenodo refused the token for record {record} (HTTP {response.status_code}).\n"
                "  - Is ZENODO_TOKEN the token itself, not a placeholder or a quoted copy of one?\n"
                "  - Was it created with the scopes deposit:write and deposit:actions?\n"
                "  - Does the account that owns the token also own this draft?")
        if response.status_code == 404:
            raise SystemExit(f"Zenodo has no draft {record} for this account. Check the id in the "
                             "draft's URL: https://zenodo.org/uploads/<id>")
        if response.status_code >= 500:
            raise SystemExit(
                f"Zenodo failed on its side for draft {record} (HTTP {response.status_code}), on both "
                "APIs.\nAn upload interrupted in the browser can leave a file entry with no size and "
                "no checksum,\nwhich the draft cannot be rendered with. Delete any file showing "
                "'Checksum not yet calculated'\nat https://zenodo.org/uploads/" + record +
                " and run this again. Otherwise it is a Zenodo outage;\nthe upload is resumable, so "
                "waiting costs nothing.")
        response.raise_for_status()
        self.data = response.json()
        self.bucket = self.data.get("links", {}).get("bucket")

    def existing(self) -> dict[str, int]:
        """Committed files on the draft, by name, with their sizes."""
        if self.bucket:
            entries = self.session.get(f"{self.bucket}", timeout=60).json().get("contents", [])
            return {e["key"]: e["size"] for e in entries}
        listing = self.session.get(f"{BASE}/records/{self.record}/draft/files", timeout=60).json()
        return {e["key"]: e.get("size") or 0 for e in listing.get("entries", [])
                if e.get("status") == "completed"}

    def upload(self, path: Path) -> None:
        if self.bucket:
            with path.open("rb") as handle:
                response = self.session.put(f"{self.bucket}/{path.name}", data=handle, timeout=None)
            response.raise_for_status()
            return
        # the newer API registers the name, streams the bytes, then commits
        self.session.post(f"{BASE}/records/{self.record}/draft/files",
                          json=[{"key": path.name}], timeout=60)
        with path.open("rb") as handle:
            response = self.session.put(
                f"{BASE}/records/{self.record}/draft/files/{path.name}/content", data=handle,
                headers={"Content-Type": "application/octet-stream"}, timeout=None)
        response.raise_for_status()
        response = self.session.post(
            f"{BASE}/records/{self.record}/draft/files/{path.name}/commit", timeout=300)
        response.raise_for_status()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--record", required=True, help="the draft's numeric id")
    parser.add_argument("--deposit", type=Path, default=PROJECT_ROOT / "deposit")
    parser.add_argument("--list", action="store_true", help="show what the draft holds and exit")
    args = parser.parse_args()

    token = os.environ.get("ZENODO_TOKEN")
    if not token:
        raise SystemExit("Set ZENODO_TOKEN first; the script does not read the token from a file.")

    draft = Draft(args.record, token)
    on_server = draft.existing()
    print(f"draft {args.record}: {len(on_server)} file(s) already uploaded")
    for name, size in sorted(on_server.items()):
        print(f"  {name:28s} {human(size)}")
    if args.list:
        return

    files = [args.deposit / name for name in UPLOAD_ORDER if (args.deposit / name).exists()]
    missing = [name for name in UPLOAD_ORDER if not (args.deposit / name).exists()]
    if missing:
        print(f"not found in {args.deposit}, skipped: {', '.join(missing)}")

    for path in files:
        size = path.stat().st_size
        if on_server.get(path.name) == size:
            print(f"  {path.name:28s} already on the draft, skipped")
            continue
        print(f"  {path.name:28s} uploading {human(size)} ...", flush=True)
        started = time.time()
        draft.upload(path)
        elapsed = time.time() - started
        print(f"  {path.name:28s} done in {elapsed / 60:.1f} min ({human(size / max(elapsed, 1))}/s)")

    print("\nAll files uploaded. Review the draft in the browser and publish it there; this script "
          "deliberately does not publish, because a published record's files are frozen.")


if __name__ == "__main__":
    main()
