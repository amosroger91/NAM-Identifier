"""Freeze capture identities and source-note splits without rendering or scoring audio.

The sampled guitars have an unverified recording signal path.  This protocol
measures a fixed-library, cross-instrument stress test; assembled phrases are
not independent recording sessions or a verified dry-DI corpus.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
from datetime import datetime, timezone
import uuid
import wave

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAPTURES = ROOT / "benchmarks" / "baseline17.json"
DEFAULT_PROTOCOL = ROOT / "benchmarks" / "measurement_protocol.json"
SCHEMA = "nam-tone-match-measurement-v1"
RATE = 48000
BANK_SEEDS = tuple(range(7001, 7009))
GREEN_QUERY_SEEDS = tuple(range(1001, 1011))
BLACK_QUERY_SEEDS = tuple(range(2001, 2011))
COMMON_BANK_GAINS = (-6.0, 0.0, 6.0, 12.0)
QUERY_GAINS = (-6.0, 0.0, 6.0)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _relative(value, *, filename=False):
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise ValueError(f"invalid relative path: {value!r}")
    path = PurePosixPath(value)
    if (path.is_absolute() or PureWindowsPath(value).drive or ".." in path.parts
            or str(path) != value or value == "." or (filename and len(path.parts) != 1)):
        raise ValueError(f"unsafe relative path: {value!r}")
    return path


def _inside(root, relative):
    root = Path(root).resolve()
    path = root.joinpath(*_relative(relative).parts).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"source escapes its root: {relative!r}") from exc
    return path


def _validate_capture_entries(entries, models_root=None, expected_count=17):
    if not isinstance(entries, list) or len(entries) != expected_count or expected_count < 1:
        raise ValueError(f"capture manifest must contain exactly {expected_count} captures")
    names, hashes = set(), set()
    for item in entries:
        if not isinstance(item, dict):
            raise ValueError("invalid capture entry")
        name, digest, size = item.get("name"), item.get("sha256"), item.get("bytes")
        _relative(name, filename=True)
        if not name.lower().endswith(".nam"):
            raise ValueError(f"capture must have a .nam extension: {name}")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"invalid capture SHA256: {name}")
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            raise ValueError(f"invalid capture byte count: {name}")
        if name.casefold() in names or digest in hashes:
            raise ValueError(f"duplicate capture name or hash: {name}")
        names.add(name.casefold())
        hashes.add(digest)
        if models_root is not None:
            path = _inside(models_root, name)
            if not path.is_file() or path.stat().st_size != size:
                raise ValueError(f"capture missing or byte size changed: {name}")
            if sha256_file(path) != digest:
                raise ValueError(f"capture content changed: {name}")
    return entries


def validate_capture_manifest(manifest_path=DEFAULT_CAPTURES, models_root=None, expected_count=17):
    """Validate exactly 17 distinct captures, optionally checking the actual files."""
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8-sig"))
    return _validate_capture_entries(manifest.get("captures"), models_root, expected_count)


def _source_info(root, path):
    relative = path.relative_to(root).as_posix()
    resolved = _inside(root, relative)
    before = resolved.stat()
    if before.st_size <= 0:
        raise ValueError(f"empty source file: {relative}")
    with wave.open(str(resolved), "rb") as source:
        wav = {"sample_rate": source.getframerate(), "frames": source.getnframes(),
               "channels": source.getnchannels(), "sample_width": source.getsampwidth()}
        if wav["frames"] <= 0 or wav["sample_rate"] <= 0 or source.getcomptype() != "NONE":
            raise ValueError(f"empty or unsupported WAV source: {relative}")
    digest = sha256_file(resolved)
    after = resolved.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"source changed during freezing: {relative}")
    return {"path": relative, "sha256": digest, "bytes": after.st_size, "wav": wav}


def _inventory(root, instrument):
    paths = sorted(path for articulation in ("ord", "stac")
                   for path in (root / instrument / articulation).glob("*.wav"))
    if not paths:
        raise ValueError(f"no ordinary/staccato WAV notes found for {instrument}")
    return [_source_info(root, path) for path in paths]


def _phrase(name, role, instrument, seed, population, count):
    if len(population) < count:
        raise ValueError(f"not enough unused source notes for {name}: {len(population)} < {count}")
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(population), size=count, replace=False)
    timing_state = rng.bit_generator.state
    position, notes = 0, []
    for index in indices:
        notes.append({**population[int(index)], "offset_samples": position})
        position += int(RATE * rng.uniform(0.5, 0.9))
    return {"name": name, "role": role, "instrument": instrument, "seed": int(seed),
            "choice_population_size": len(population), "choice_indices": indices.tolist(),
            "timing_rng_state_after_choice": timing_state, "notes": notes,
            "output_samples": min(12 * RATE, position + RATE)}


def _unique_pool(inventory, excluded):
    seen, out = set(excluded), []
    for source in inventory:
        if source["sha256"] not in seen:
            out.append(source)
            seen.add(source["sha256"])
    return out


def _query_phrases(inventory, instrument, seeds, excluded, note_count):
    out, used = {}, set(excluded)
    for seed in seeds:
        name = f"q-{instrument}{seed}"
        phrase = _phrase(name, "query", instrument, seed, _unique_pool(inventory, used), note_count)
        used.update(note["sha256"] for note in phrase["notes"])
        out[name] = phrase
    return out


def _validate_protocol(doc):
    if not isinstance(doc, dict) or doc.get("schema") != SCHEMA:
        raise ValueError("unsupported measurement protocol")
    expected_id = _digest({key: value for key, value in doc.items() if key != "protocol_id"})
    if doc.get("protocol_id") != expected_id:
        raise ValueError("measurement protocol integrity mismatch")
    _validate_capture_entries(doc.get("captures"))
    phrases = doc.get("phrases")
    note_count = doc.get("audio", {}).get("note_count")
    if not isinstance(phrases, dict) or not phrases or not isinstance(note_count, int) or note_count < 1:
        raise ValueError("empty or invalid phrase definitions")
    seen_paths, bank_hashes, query_hashes = {}, set(), set()
    roles = {("bank", "green"): 0, ("query", "green"): 0, ("query", "black"): 0}
    for name, phrase in phrases.items():
        role = (phrase.get("role"), phrase.get("instrument"))
        if role not in roles or phrase.get("name") != name:
            raise ValueError(f"invalid phrase role/name: {name}")
        roles[role] += 1
        notes = phrase.get("notes")
        if not isinstance(notes, list) or len(notes) != note_count:
            raise ValueError(f"wrong note count: {name}")
        own_hashes = set()
        for note in notes:
            path = _relative(note.get("path"))
            if len(path.parts) != 3 or path.parts[0] != role[1] or path.parts[1] not in ("ord", "stac"):
                raise ValueError(f"source instrument/articulation mismatch: {name}")
            digest, size = note.get("sha256"), note.get("bytes")
            if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError(f"invalid source hash: {name}")
            if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
                raise ValueError(f"invalid source byte size: {name}")
            identity = (digest, size)
            if str(path) in seen_paths and seen_paths[str(path)] != identity:
                raise ValueError(f"inconsistent source identity: {path}")
            seen_paths[str(path)] = identity
            offset = note.get("offset_samples")
            if not isinstance(offset, int) or offset < 0:
                raise ValueError(f"invalid source offset: {name}")
            own_hashes.add(digest)
        if role[0] == "bank":
            bank_hashes.update(own_hashes)
        else:
            if len(own_hashes) != note_count or own_hashes & query_hashes:
                raise ValueError(f"source-note reuse between/within query composites: {name}")
            query_hashes.update(own_hashes)
    if not all(roles.values()) or bank_hashes & query_hashes:
        raise ValueError("empty split or bank/query source-note leakage")
    for role, instrument in roles:
        expected = doc.get("split", {}).get(f"{role}_{instrument}_seeds")
        actual = sorted(p["seed"] for p in phrases.values()
                        if p["role"] == role and p["instrument"] == instrument)
        if not expected or actual != sorted(expected):
            raise ValueError(f"phrase seeds disagree with the frozen split: {role}/{instrument}")
    return doc


def freeze_protocol(samples_root, captures_manifest=DEFAULT_CAPTURES,
                    output_path=DEFAULT_PROTOCOL, *, models_root=None, overwrite=False,
                    bank_seeds=BANK_SEEDS, green_query_seeds=GREEN_QUERY_SEEDS,
                    black_query_seeds=BLACK_QUERY_SEEDS, note_count=12):
    """Write a frozen protocol; does not decode, render, or score any model.

    ``phrases`` is keyed by the existing names (green7001, q-green1001,
    q-black2001, ...). Each phrase has ordered ``notes`` with root-relative
    paths, source SHA256/bytes, and exact sample offsets. Source roots remain
    caller parameters and are never persisted in the manifest.
    """
    root, destination = Path(samples_root).resolve(), Path(output_path)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"protocol already frozen: {destination}; use load_protocol")
    if not isinstance(note_count, int) or isinstance(note_count, bool) or note_count < 1:
        raise ValueError("note_count must be positive")
    seeds = [tuple(bank_seeds), tuple(green_query_seeds), tuple(black_query_seeds)]
    if any(not group or len(group) != len(set(group)) or any(not isinstance(s, int) for s in group)
           for group in seeds):
        raise ValueError("seed groups must be nonempty and distinct within each role")
    if set(seeds[0]) & set(seeds[1]):
        raise ValueError("green bank/query seed groups overlap")
    captures = validate_capture_manifest(captures_manifest, models_root)
    green, black = _inventory(root, "green"), _inventory(root, "black")
    phrases = {}
    for seed in seeds[0]:
        name = f"green{seed}"
        phrases[name] = _phrase(name, "bank", "green", seed, green, note_count)
    bank_hashes = {n["sha256"] for phrase in phrases.values() for n in phrase["notes"]}
    phrases.update(_query_phrases(green, "green", seeds[1], bank_hashes, note_count))
    dev_hashes = {n["sha256"] for phrase in phrases.values() for n in phrase["notes"]}
    phrases.update(_query_phrases(black, "black", seeds[2], dev_hashes, note_count))
    doc = {
        "schema": SCHEMA, "created_at": datetime.now(timezone.utc).isoformat(),
        "captures_manifest_sha256": sha256_file(captures_manifest), "captures": captures,
        "dataset": {"label": "Black and Green Guitars sampled-note stress library",
                    "signal_path": "unverified; not established as dry electric-guitar DI",
                    "green_role": "development instrument", "black_role": "held-out instrument stress test",
                    "inventory": {kind: {"files": len(items), "inventory_sha256": _digest(items)}
                                  for kind, items in (("green", green), ("black", black))}},
        "audio": {"sample_rate": RATE, "note_count": note_count, "buffer_samples": 12 * RATE,
                  "note_max_samples": int(1.6 * RATE), "fade": "linspace(1, 0, samples) ** 0.3",
                  "spacing_seconds": [0.5, 0.9], "tail_samples": RATE, "input_peak": 0.6,
                  "output_render_peak": 0.5, "decode": "decode_wav then effects.resample to analysis rate"},
        "split": {"bank_green_seeds": list(seeds[0]), "query_green_seeds": list(seeds[1]),
                  "query_black_seeds": list(seeds[2]),
                  "rule": "Preserve original green-bank choices. Query source hashes are disjoint from bank and all other queries."},
        "gains_db": {"bank_common": list(COMMON_BANK_GAINS), "query": list(QUERY_GAINS)},
        "phrases": phrases,
        "generated_inputs": {
            "probe": {"generator": "match.make_probe_di", "seed": 7, "seconds": 12.0},
            "synth_bank": {f"synth{i}": {"generator": "phrases.training_phrases", "seed": 9001 + i,
                                         "color_seed": 9001 + i, "seconds": 6.0} for i in range(8)},
            "synth_queries": {f"q-synth{seed}": {"generator": "phrases.make_phrase", "seed": seed,
                                                 "seconds": 9.0} for seed in range(3001, 3011)}},
        "uncertainty_scope": "Conditional on this fixed capture set and sampled-note library. Composite groups are not independent instruments, performers, or sessions.",
    }
    doc["protocol_id"] = _digest(doc)
    _validate_protocol(doc)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + "." + uuid.uuid4().hex + ".part")
    try:
        temporary.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return doc


def load_protocol(protocol_path=DEFAULT_PROTOCOL, samples_root=None, *, verify_sources=True,
                  captures_manifest=None, models_root=None):
    """Check protocol integrity and optionally verify all used WAV/capture bytes."""
    doc = _validate_protocol(json.loads(Path(protocol_path).read_text(encoding="utf-8-sig")))
    _validate_capture_entries(doc["captures"], models_root)
    if captures_manifest is not None and sha256_file(captures_manifest) != doc["captures_manifest_sha256"]:
        raise ValueError("capture manifest changed after protocol freezing")
    if verify_sources:
        if samples_root is None:
            raise ValueError("samples_root is required when verifying sources")
        checked = set()
        for phrase in doc["phrases"].values():
            for note in phrase["notes"]:
                if note["path"] in checked:
                    continue
                path = _inside(samples_root, note["path"])
                if not path.is_file() or path.stat().st_size != note["bytes"]:
                    raise ValueError(f"source missing or byte size changed: {note['path']}")
                if sha256_file(path) != note["sha256"]:
                    raise ValueError(f"source content changed: {note['path']}")
                checked.add(note["path"])
    return doc


def get_phrase(protocol, name):
    """Return one frozen recorded-note phrase definition by its existing name."""
    try:
        return protocol["phrases"][name]
    except KeyError as exc:
        raise ValueError(f"unknown frozen phrase: {name}") from exc
