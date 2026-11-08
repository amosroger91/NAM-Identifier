#!/usr/bin/env python3
"""Collect public NAM captures without extracting untrusted archive paths.

Example config::

    {"sources": [
      {"type": "github", "repo": "owner/repo", "ref": "main", "paths": ["captures"]},
      {"type": "archive", "url": "https://example.org/pack.zip", "name": "creator-pack"},
      {"type": "file", "url": "https://example.org/model.nam", "name": "creator"},
      {"type": "file", "url": "https://example.org/model.namz", "name": "creator", "encoding": "namz"}
    ]}

Every source may declare ``license`` and ``license_url``. Missing licenses are
recorded as unknown, never inferred from public download availability. GitHub
repository license metadata is recorded when available. No third-party packages
are required for ordinary NAM files. Optional ``encoding: "namz"`` direct-file
sources require the official ``namz`` decoder on Python's import path. Output is
a flat, content-deduplicated library plus catalog.jsonl.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tarfile
import tempfile
import threading
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import uuid
import zipfile
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime


MIB = 1024 * 1024
USER_AGENT = "NAM-Identifier-library-collector/1.0"


class LimitReached(RuntimeError):
    pass


def log(message):
    print(message, file=sys.stderr, flush=True)


def safe_name(value, limit=65):
    value = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    value = re.sub(r"[^A-Za-z0-9_-]+", "-", value).strip("-_")
    return (value or "capture")[:limit]


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(MIB), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_nam(raw):
    """Validate capture structure, including all SlimmableContainer submodels."""
    try:
        data = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeError, ValueError) as exc:
        raise ValueError(f"not NAM JSON: {exc}") from exc

    def validate(model, depth=0):
        if depth > 8 or not isinstance(model, dict):
            raise ValueError("invalid NAM model object")
        if not isinstance(model.get("architecture"), str) or not isinstance(model.get("config"), dict):
            raise ValueError("NAM needs architecture and config")
        if model["architecture"] == "SlimmableContainer":
            subs = model["config"].get("submodels")
            if not isinstance(subs, list) or not subs:
                raise ValueError("SlimmableContainer needs submodels")
            for item in subs:
                if not isinstance(item, dict):
                    raise ValueError("invalid NAM submodel")
                validate(item.get("model"), depth + 1)
        elif not isinstance(model.get("weights"), list) or not model["weights"]:
            raise ValueError("NAM needs a nonempty weights array")

    validate(data)
    return data


class Collector:
    def __init__(self, args):
        self.args = args
        self.output = Path(args.output).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.mirror = Path(args.mirror_dir).resolve() if args.mirror_dir else None
        if self.mirror:
            self.mirror.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.hashes = {}
        self.mirrored = set()
        self.sources = set()
        self.stats = {"downloaded": 0, "deduplicated": 0, "skipped": 0, "errors": 0,
                      "network_bytes": 0, "stored_bytes": 0, "catalog_records": 0}
        self.errors = []
        self.started = datetime.now(timezone.utc).isoformat()
        self.catalog_path = self.output / "catalog.jsonl"
        self._seed()

    def _seed(self):
        catalog_entries = []
        if self.catalog_path.exists():
            with self.catalog_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    try:
                        entry = json.loads(line)
                        name = entry["file"]
                        if name == Path(name).name and (self.output / name).is_file():
                            catalog_entries.append(entry)
                    except (ValueError, KeyError, TypeError):
                        log("Ignoring incomplete catalog line (a previous interrupted append is recoverable)")
            # Keep a crash-truncated final line separate from the next append.
            with self.catalog_path.open("rb+") as handle:
                handle.seek(0, os.SEEK_END)
                if handle.tell():
                    handle.seek(-1, os.SEEK_END)
                    if handle.read(1) != b"\n":
                        handle.write(b"\n")
        seed_dirs = [self.output, Path("models").resolve()]
        if self.mirror:
            seed_dirs.append(self.mirror)
        visited = set()
        output_digests = {}
        for directory in seed_dirs:
            if directory in visited or not directory.is_dir():
                continue
            visited.add(directory)
            for path in sorted(directory.glob("*.nam")):
                try:
                    digest = sha256_file(path)
                    self.hashes.setdefault(digest, path)
                    if directory == self.output:
                        output_digests[path.name] = digest
                    if directory == self.mirror:
                        self.mirrored.add(digest)
                except OSError as exc:
                    log(f"Cannot seed {path.name}: {exc}")
        for entry in catalog_entries:
            if output_digests.get(entry["file"]) == entry.get("sha256"):
                self.sources.add(entry["source_url"])
        log(f"Resume: {len(self.sources)} catalog sources, {len(self.hashes)} existing unique captures")

    def error(self, source, exc):
        with self.lock:
            self.stats["errors"] += 1
            self.errors.append({"source": str(source), "error": str(exc)})
        log(f"ERROR {source}: {exc}")

    def consume_network(self, size):
        with self.lock:
            if self.stats["network_bytes"] + size > self.args.max_bytes:
                self.stop.set()
                raise LimitReached("--max-bytes network budget reached")
            self.stats["network_bytes"] += size

    def open_url(self, url):
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc or parsed.username or parsed.password:
            raise ValueError("source URL must be HTTP(S) without embedded credentials")
        for attempt in range(self.args.retries + 1):
            if self.stop.is_set():
                raise LimitReached("collection limit reached")
            try:
                request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
                return urllib.request.urlopen(request, timeout=self.args.timeout)
            except urllib.error.HTTPError as exc:
                retryable = exc.code in (429, 500, 502, 503, 504) or (
                    exc.code == 403 and exc.headers.get("X-RateLimit-Remaining") == "0")
                if not retryable or attempt == self.args.retries:
                    raise
                try:
                    delay = float(exc.headers.get("Retry-After", ""))
                except ValueError:
                    try:
                        retry_at = parsedate_to_datetime(exc.headers.get("Retry-After", ""))
                        delay = retry_at.timestamp() - time.time()
                    except (ValueError, TypeError, OverflowError):
                        delay = 2 ** (attempt + 1)
                if exc.headers.get("X-RateLimit-Remaining") == "0":
                    try:
                        delay = max(delay, float(exc.headers.get("X-RateLimit-Reset", 0)) - time.time())
                    except ValueError:
                        pass
                if delay > 60:
                    raise RuntimeError(f"server asks to wait {delay:.0f}s; rerun later (resume is automatic)") from exc
                log(f"HTTP {exc.code}; retrying in {max(1, delay):.0f}s: {url}")
                self.stop.wait(max(1, delay))
            except (urllib.error.URLError, TimeoutError, OSError):
                if attempt == self.args.retries:
                    raise
                self.stop.wait(min(2 ** (attempt + 1), 20))
        raise RuntimeError("request retries exhausted")

    def read_url(self, url, max_size):
        chunks = []
        size = 0
        with self.open_url(url) as response:
            length = response.headers.get("Content-Length")
            if length and int(length) > max_size:
                raise ValueError(f"download exceeds {max_size} byte item limit")
            while True:
                chunk = response.read(min(MIB, max_size + 1 - size))
                if not chunk:
                    break
                self.consume_network(len(chunk))
                size += len(chunk)
                if size > max_size:
                    raise ValueError(f"download exceeds {max_size} byte item limit")
                chunks.append(chunk)
        return b"".join(chunks)

    def json_url(self, url):
        return json.loads(self.read_url(url, 32 * MIB))

    def github(self, source):
        repo = source["repo"]
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo):
            raise ValueError(f"invalid GitHub repo: {repo!r}")
        api = f"https://api.github.com/repos/{repo}"
        info = self.json_url(api)
        ref = source.get("ref") or info["default_branch"]
        # Pin raw downloads to one commit even if a branch moves during collection.
        commit = self.json_url(f"{api}/commits/{urllib.parse.quote(ref, safe='')}")["sha"]
        tree = self.json_url(f"{api}/git/trees/{commit}?recursive=1")
        if tree.get("truncated"):
            raise ValueError("GitHub recursive tree was truncated; use an archive source for this repository")
        paths = source.get("paths") or []
        if not isinstance(paths, list) or not all(isinstance(p, str) for p in paths):
            raise ValueError("GitHub paths must be a list of directory prefixes")
        prefixes = [p.strip("/") for p in paths]
        license_info = info.get("license") or {}
        license_name = source.get("license") or license_info.get("spdx_id") or "unknown"
        if license_name == "NOASSERTION":
            license_name = "unknown"
        for item in sorted(tree.get("tree", []), key=lambda item: item.get("path", "")):
            name = item.get("path", "")
            if item.get("type") != "blob" or not name.lower().endswith(".nam"):
                continue
            if prefixes and not any(name == p or name.startswith(p + "/") for p in prefixes):
                continue
            url = f"https://raw.githubusercontent.com/{repo}/{commit}/{urllib.parse.quote(name, safe='/')}"
            yield {"source_url": url, "download_url": url, "path": name,
                   "creator": repo, "repo": repo, "ref": ref, "commit": commit,
                   "blob_sha": item.get("sha"), "license": license_name,
                   "license_url": source.get("license_url") or license_info.get("html_url"),
                   "size_hint": item.get("size")}

    def _mirror(self, path, digest):
        if not self.mirror or digest in self.mirrored:
            return
        destination = self.mirror / path.name
        if destination.resolve() == path.resolve():
            self.mirrored.add(digest)
            return
        if destination.exists():
            if sha256_file(destination) != digest:
                raise ValueError(f"mirror filename collision: {destination.name}")
        else:
            self.link_or_copy(path, destination)
        self.mirrored.add(digest)

    @staticmethod
    def link_or_copy(source, destination):
        temporary = destination.with_name(destination.name + "." + uuid.uuid4().hex + ".part")
        try:
            try:
                os.link(source, temporary)
            except OSError:
                shutil.copyfile(source, temporary)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)

    def save(self, candidate, raw):
        model = validate_nam(raw)
        digest = hashlib.sha256(raw).hexdigest()
        stem = PurePosixPath(candidate["path"].replace("\\", "/")).stem
        filename = f"{safe_name(candidate['creator'], 65)}--{safe_name(stem, 95)}--{digest[:12]}.nam"
        with self.lock:
            if candidate["source_url"] in self.sources:
                self.stats["skipped"] += 1
                return
            previous = self.hashes.get(digest)
            deduplicated = previous is not None
            if previous and previous.parent == self.output:
                path = previous
            else:
                if self.stats["downloaded"] >= self.args.max_files:
                    self.stop.set()
                    raise LimitReached("--max-files reached")
                if self.stats["stored_bytes"] + len(raw) > self.args.max_bytes:
                    self.stop.set()
                    raise LimitReached("--max-bytes storage budget reached")
                path = self.output / filename
                if path.exists() and sha256_file(path) != digest:
                    path = path.with_name(f"{path.stem.rsplit('--', 1)[0]}--{digest}.nam")
                    if path.exists() and sha256_file(path) != digest:
                        raise ValueError(f"existing file has conflicting content: {path.name}")
                if previous:
                    self.link_or_copy(previous, path)
                else:
                    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".part")
                    try:
                        with temporary.open("wb") as handle:
                            handle.write(raw)
                            handle.flush()
                            os.fsync(handle.fileno())
                        os.replace(temporary, path)
                    finally:
                        temporary.unlink(missing_ok=True)
                self.stats["downloaded"] += 1
                self.stats["stored_bytes"] += len(raw)
            self.hashes[digest] = path
            self._mirror(path, digest)
            record = {k: v for k, v in candidate.items() if k not in ("size_hint",)}
            record.update({"file": path.name, "sha256": digest, "size": len(raw),
                           "architecture": model["architecture"], "version": model.get("version"),
                           "sample_rate": model.get("sample_rate"), "metadata": model.get("metadata"),
                           "imported_at": datetime.now(timezone.utc).isoformat(), "deduplicated": deduplicated})
            with self.catalog_path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            self.sources.add(candidate["source_url"])
            self.stats["catalog_records"] += 1
            self.stats["deduplicated"] += int(deduplicated)
            log(f"{'DEDUP' if deduplicated else 'SAVED'} {path.name}")

    def collect_file(self, candidate):
        try:
            with self.lock:
                if candidate["source_url"] in self.sources:
                    self.stats["skipped"] += 1
                    return
            if candidate.get("size_hint", 0) > self.args.max_file_bytes:
                raise ValueError("capture exceeds --max-file-bytes")
            raw = self.read_url(candidate["download_url"], self.args.max_file_bytes)
            encoding = candidate.get("encoding") or "nam"
            candidate = {**candidate, "source_encoding": encoding,
                         "source_sha256": hashlib.sha256(raw).hexdigest(), "source_size": len(raw)}
            if encoding == "namz":
                try:
                    import namz
                except ImportError as exc:
                    raise ValueError("NAMZ source requires the official namz decoder on Python's import path") from exc
                raw = namz.unpack(raw, max_json_bytes=self.args.max_file_bytes)
                if not raw:
                    raise ValueError("NAMZ decoding failed: malformed input or decoded size exceeds --max-file-bytes")
                candidate["source_decoder"] = "namz:" + str(getattr(namz, "__version__", "unknown"))
            elif encoding != "nam":
                raise ValueError(f"unsupported source encoding {encoding!r}")
            if len(raw) > self.args.max_file_bytes:
                raise ValueError("decoded capture exceeds --max-file-bytes")
            self.save(candidate, raw)
        except LimitReached as exc:
            log(str(exc))
        except Exception as exc:
            self.error(candidate["source_url"], exc)

    def collect_files(self, candidates):
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.args.workers) as pool:
            pending = set()
            for candidate in candidates:
                if self.stop.is_set():
                    break
                if self.args.dry_run:
                    print(json.dumps(candidate, ensure_ascii=False))
                    continue
                pending.add(pool.submit(self.collect_file, candidate))
                if len(pending) >= self.args.workers:
                    _, pending = concurrent.futures.wait(pending, return_when=concurrent.futures.FIRST_COMPLETED)
            concurrent.futures.wait(pending)

    def collect_archive(self, source):
        url = source["url"]
        if self.args.dry_run:
            print(json.dumps({"archive": url, "action": "download then enumerate NAM members"}))
            return
        creator = source.get("name") or urllib.parse.urlparse(url).netloc

        def member(name, size, handle):
            if not name.lower().endswith(".nam") or self.stop.is_set():
                return
            source_url = url + "#" + urllib.parse.quote(name, safe="/")
            if source_url in self.sources:
                self.stats["skipped"] += 1
                return
            try:
                if size > self.args.max_file_bytes:
                    raise ValueError("archive member exceeds --max-file-bytes")
                raw = handle.read(self.args.max_file_bytes + 1)
                if len(raw) > self.args.max_file_bytes:
                    raise ValueError("archive member exceeds --max-file-bytes")
                self.save({"source_url": source_url, "download_url": url, "archive_url": url,
                           "path": name, "creator": creator, "license": source.get("license") or "unknown",
                           "license_url": source.get("license_url")}, raw)
            except LimitReached as exc:
                log(str(exc))
            except Exception as exc:
                self.error(source_url, exc)

        # Spool the compressed archive to disk; never extract names supplied by it.
        with tempfile.TemporaryFile(dir=self.output, suffix=".archive.part") as archive:
            with self.open_url(url) as response:
                while not self.stop.is_set():
                    chunk = response.read(MIB)
                    if not chunk:
                        break
                    self.consume_network(len(chunk))
                    archive.write(chunk)
            if self.stop.is_set():
                return
            archive.seek(0)
            if zipfile.is_zipfile(archive):
                with zipfile.ZipFile(archive) as zipped:
                    for item in zipped.infolist():
                        if not item.is_dir() and item.filename.lower().endswith(".nam"):
                            try:
                                with zipped.open(item) as handle:
                                    member(item.filename, item.file_size, handle)
                            except Exception as exc:
                                self.error(url + "#" + item.filename, exc)
                        if self.stop.is_set():
                            break
            else:
                archive.seek(0)
                with tarfile.open(fileobj=archive, mode="r:*") as tar:
                    for item in tar:
                        if item.isfile() and item.name.lower().endswith(".nam"):
                            handle = tar.extractfile(item)
                            if handle:
                                with handle:
                                    member(item.name, item.size, handle)
                        if self.stop.is_set():
                            break

    def run(self, sources):
        for source in sources:
            if self.stop.is_set():
                break
            try:
                kind = source.get("type", "github")
                if kind == "github":
                    self.collect_files(self.github(source))
                elif kind == "archive":
                    self.collect_archive(source)
                elif kind == "file":
                    url = source["url"]
                    path = urllib.parse.unquote(urllib.parse.urlparse(url).path)
                    self.collect_files([{"source_url": url, "download_url": url, "path": path,
                                         "creator": source.get("name") or urllib.parse.urlparse(url).netloc,
                                         "encoding": source.get("encoding") or "nam",
                                         "license": source.get("license") or "unknown",
                                         "license_url": source.get("license_url")}])
                else:
                    raise ValueError(f"unknown source type {kind!r}")
            except LimitReached as exc:
                log(str(exc))
                break
            except Exception as exc:
                self.error(source.get("repo") or source.get("url") or source, exc)
        summary = {"started_at": self.started, "finished_at": datetime.now(timezone.utc).isoformat(),
                   **self.stats, "unique_captures_available": len(self.hashes), "limit_reached": self.stop.is_set(),
                   "dry_run": self.args.dry_run, "errors_detail": self.errors}
        if not self.args.dry_run:
            temporary = self.output / "summary.json.part"
            temporary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
            os.replace(temporary, self.output / "summary.json")
        print(json.dumps(summary, indent=2))
        return 1 if self.stats["errors"] and not self.stats["catalog_records"] else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="JSON config with a sources list")
    parser.add_argument("--output", default="nams")
    parser.add_argument("--mirror-dir", help="hardlink captures into the app's models directory, copying if needed")
    parser.add_argument("--workers", type=int, default=4, choices=range(1, 5))
    parser.add_argument("--timeout", type=float, default=45)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--max-files", type=int, default=5000, help="maximum new library files in this run")
    parser.add_argument("--max-bytes", type=int, default=5 * 1024 ** 3,
                        help="per-run network and stored-byte ceiling (default 5 GiB)")
    parser.add_argument("--max-file-bytes", type=int, default=128 * MIB)
    parser.add_argument("--dry-run", action="store_true", help="discover file URLs without downloading captures")
    args = parser.parse_args(argv)
    if min(args.max_files, args.max_bytes, args.max_file_bytes, args.timeout) <= 0 or not 0 <= args.retries <= 6:
        parser.error("limits and timeout must be positive; retries must be 0..6")
    try:
        config = json.loads(Path(args.config).read_text(encoding="utf-8-sig"))
        sources = config["sources"]
        if not isinstance(sources, list) or not all(isinstance(source, dict) for source in sources):
            raise ValueError("sources must be a list of source objects")
        return Collector(args).run(sources)
    except KeyboardInterrupt:
        log("Interrupted. Completed captures are safe; rerun the same command to resume.")
        return 130
    except (OSError, ValueError, KeyError) as exc:
        log(f"Configuration error: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
