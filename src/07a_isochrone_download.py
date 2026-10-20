from __future__ import annotations

import argparse
import re
import sys
import tarfile
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin

import requests

from config import DATA_DIR


MODEL_GRIDS_URL = "https://mist.science/model_grids.html"
ARCHIVE_TOKEN = "MIST_v1.2_vvcrit0.4_UBVRIplus.txz"
TARGET_FILES = {
    "+0.00": "MIST_v1.2_feh_p0.00_afe_p0.0_vvcrit0.4_UBVRIplus.iso.cmd",
    "+0.25": "MIST_v1.2_feh_p0.25_afe_p0.0_vvcrit0.4_UBVRIplus.iso.cmd",
}
FEH_CODE_RE = re.compile(r"feh_([pm]\d+\.\d+)_", re.IGNORECASE)
GAIA_BAND_RE = re.compile(r"^gaia_(g|bp|rp)_(.+)$", re.IGNORECASE)


class AnchorParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.anchors: list[dict[str, str]] = []
        self._href: str | None = None
        self._chunks: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        attr_map = dict(attrs)
        self._href = attr_map.get("href")
        self._chunks = []

    def handle_data(self, data: str) -> None:
        if self._href is not None:
            self._chunks.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() != "a" or self._href is None:
            return
        text = " ".join(part.strip() for part in self._chunks if part.strip())
        self.anchors.append({"href": self._href, "text": text})
        self._href = None
        self._chunks = []


class MissingTargetFilesError(RuntimeError):
    def __init__(self, missing: list[str], available_feh: list[str]) -> None:
        missing_str = ", ".join(missing)
        available_str = ", ".join(available_feh) if available_feh else "none detected"
        super().__init__(
            f"Did not find the requested files in the archive: {missing_str}. "
            f"Archive [Fe/H] values: {available_str}"
        )
        self.missing = missing
        self.available_feh = available_feh


class ArchiveInspection:
    def __init__(self, member_names: list[str], feh_counts: Counter[str]) -> None:
        self.member_names = member_names
        self.feh_counts = feh_counts


def human_size(size_bytes: int | None) -> str:
    if size_bytes is None:
        return "unknown size"
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(size_bytes)
    unit = units[0]
    for unit in units:
        if size < 1024 or unit == units[-1]:
            break
        size /= 1024
    if unit == "B":
        return f"{int(size)} {unit}"
    return f"{size:.1f} {unit}"


def fetch_archive_listing() -> tuple[str, str | None, int | None]:
    response = requests.get(MODEL_GRIDS_URL, timeout=60)
    response.raise_for_status()

    parser = AnchorParser()
    parser.feed(response.text)

    for anchor in parser.anchors:
        href = anchor["href"]
        text = anchor["text"]
        if ARCHIVE_TOKEN not in href:
            continue
        archive_url = urljoin(MODEL_GRIDS_URL, href)
        size_text_match = re.search(r"\(([^)]+)\)", text)
        size_text = size_text_match.group(1) if size_text_match else None
        expected_bytes = head_content_length(archive_url)
        return archive_url, size_text, expected_bytes

    raise RuntimeError("Could not find the official MIST v1.2 UBVRIplus archive on the model_grids page.")


def head_content_length(url: str) -> int | None:
    try:
        response = requests.head(url, allow_redirects=True, timeout=60)
        response.raise_for_status()
    except requests.RequestException:
        return None

    content_length = response.headers.get("Content-Length")
    if not content_length:
        return None
    try:
        return int(content_length)
    except ValueError:
        return None


def find_existing_target_files(iso_dir: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for label, filename in TARGET_FILES.items():
        path = iso_dir / filename
        if path.exists():
            found[label] = path
    return found


def download_archive(url: str, destination: Path) -> None:
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total_bytes = int(response.headers.get("Content-Length", "0") or "0") or None
        downloaded = 0
        chunk_size = 1024 * 1024

        with destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                handle.write(chunk)
                downloaded += len(chunk)
                print_progress(downloaded, total_bytes)

    if total_bytes is not None:
        print()


def print_progress(downloaded: int, total_bytes: int | None) -> None:
    if total_bytes:
        percent = 100.0 * downloaded / total_bytes
        message = (
            f"\rDownloading: {human_size(downloaded)} / {human_size(total_bytes)} "
            f"({percent:5.1f}%)"
        )
    else:
        message = f"\rDownloading: {human_size(downloaded)}"
    print(message, end="", flush=True)


def extract_feh_value(name: str) -> str | None:
    match = FEH_CODE_RE.search(name)
    if not match:
        return None
    code = match.group(1).lower()
    sign = "+" if code.startswith("p") else "-"
    return f"{sign}{code[1:]}"


def inspect_archive(archive_path: Path) -> ArchiveInspection:
    member_names: list[str] = []
    feh_counts: Counter[str] = Counter()

    with tarfile.open(archive_path, mode="r:xz") as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            basename = Path(member.name).name
            member_names.append(basename)
            feh_value = extract_feh_value(basename)
            if feh_value is not None:
                feh_counts[feh_value] += 1

    return ArchiveInspection(member_names=member_names, feh_counts=feh_counts)


def print_archive_summary(inspection: ArchiveInspection) -> None:
    print("First 40 archive member names:")
    for name in inspection.member_names[:40]:
        print(f"  {name}")

    print("[Fe/H] member counts:")
    if inspection.feh_counts:
        for feh_value in sorted(inspection.feh_counts):
            print(f"  {feh_value}: {inspection.feh_counts[feh_value]}")
    else:
        print("  none detected")


def extract_target_files(archive_path: Path, iso_dir: Path) -> dict[str, Path]:
    inspection = inspect_archive(archive_path)
    print_archive_summary(inspection)

    archive_members = set(inspection.member_names)
    missing = [filename for filename in TARGET_FILES.values() if filename not in archive_members]
    if missing:
        raise MissingTargetFilesError(missing=missing, available_feh=sorted(inspection.feh_counts))

    extracted: dict[str, Path] = {}
    with tarfile.open(archive_path, mode="r:xz") as archive:
        members = [member for member in archive.getmembers() if member.isfile()]
        for label, filename in TARGET_FILES.items():
            member = next((item for item in members if Path(item.name).name == filename), None)
            if member is None:
                raise MissingTargetFilesError(missing=[filename], available_feh=sorted(inspection.feh_counts))

            source = archive.extractfile(member)
            if source is None:
                raise RuntimeError(f"Could not extract {filename} from the archive.")

            destination = iso_dir / filename
            with source, destination.open("wb") as handle:
                handle.write(source.read())
            extracted[label] = destination

    return extracted


def read_header_lines(path: Path, max_lines: int = 300) -> list[str]:
    lines: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for _ in range(max_lines):
            line = handle.readline()
            if line == "":
                break
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                break
            lines.append(line.rstrip("\n"))
    return lines


def parse_column_names(header_lines: list[str]) -> list[str]:
    for line in reversed(header_lines):
        stripped = line.lstrip("#").strip()
        if not stripped:
            continue
        tokens = stripped.split()
        has_expected = any(token in {"EEP", "phase", "log_Teff", "log_L"} for token in tokens)
        if has_expected and len(tokens) >= 10:
            return tokens
    return []


def find_gaia_columns(column_names: list[str]) -> list[str]:
    gaia_columns: list[str] = []
    for name in column_names:
        match = GAIA_BAND_RE.match(name)
        if match:
            gaia_columns.append(name)
    return gaia_columns


def find_gaia_variants(gaia_columns: list[str]) -> list[str]:
    variants: set[str] = set()
    for name in gaia_columns:
        match = GAIA_BAND_RE.match(name)
        if match:
            variants.add(match.group(2))
    return sorted(variants)


def find_phase_column(column_names: list[str]) -> str | None:
    for name in column_names:
        if name.lower() == "phase":
            return name
    return None


def find_zero_point_statement(header_lines: list[str]) -> str:
    for line in header_lines:
        stripped = line.lstrip("#").strip()
        lowered = stripped.lower()
        if "vega" in lowered or re.search(r"\bab\b", lowered):
            return stripped
    return "Zero-point statement not found in header."


def print_file_summary(files_by_label: dict[str, Path]) -> None:
    print("Kept files:")
    for label, filename in TARGET_FILES.items():
        path = files_by_label[label]
        if path.name != filename:
            raise RuntimeError(f"Unexpected output file for {label}: {path.name}")
        print(f"  {label}: {path.name} ({human_size(path.stat().st_size)})")


def cleanup_iso_cmd_files(iso_dir: Path, keep_paths: dict[str, Path], archive_path: Path) -> None:
    keep_resolved = {path.resolve() for path in keep_paths.values()}
    for path in iso_dir.glob("*.iso.cmd"):
        if path.resolve() in keep_resolved:
            continue
        path.unlink()
    if len(keep_paths) == len(TARGET_FILES) and archive_path.exists():
        archive_path.unlink()


def print_target_details(path: Path) -> None:
    header_lines = read_header_lines(path)
    column_names = parse_column_names(header_lines)
    gaia_columns = find_gaia_columns(column_names)
    gaia_variants = find_gaia_variants(gaia_columns)
    phase_column = find_phase_column(column_names)
    zero_point_statement = find_zero_point_statement(header_lines)

    print(f"\nTarget file: {path.name}")
    print(f"Size: {human_size(path.stat().st_size)}")
    print("First 15 header lines:")
    for line in header_lines[:15]:
        print(line)

    print("Gaia G/BP/RP columns:")
    if gaia_columns:
        for name in gaia_columns:
            print(f"  {name}")
    else:
        print("  none found")

    print("Gaia variants present:")
    if gaia_variants:
        print(f"  {', '.join(gaia_variants)}")
    else:
        print("  none found")

    print("Phase column:")
    if phase_column is not None:
        print(f"  {phase_column}")
    else:
        print("  not present")

    print("Photometric zero-point statement:")
    print(f"  {zero_point_statement}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Find and optionally download the official MIST v1.2 UBVRIplus isochrone grid."
    )
    parser.add_argument(
        "--go",
        action="store_true",
        help="Download the archive if it is missing locally, then extract the exact target files.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete every other .iso.cmd file and delete the archive after both targets are present.",
    )
    args = parser.parse_args()

    iso_dir = DATA_DIR / "isochrones"
    iso_dir.mkdir(parents=True, exist_ok=True)

    archive_path = iso_dir / ARCHIVE_TOKEN
    existing = find_existing_target_files(iso_dir)

    if len(existing) == len(TARGET_FILES):
        print("already present")
        print_file_summary(existing)
        if args.clean:
            cleanup_iso_cmd_files(iso_dir, existing, archive_path)
            print("Removed other .iso.cmd files and deleted the archive because --clean was requested.")
        for label in TARGET_FILES:
            print_target_details(existing[label])
        return 0

    if archive_path.exists():
        print(f"Archive already on disk; skipping download: {archive_path} ({human_size(archive_path.stat().st_size)})")
    else:
        archive_url, size_text, expected_bytes = fetch_archive_listing()
        print(f"Archive URL: {archive_url}")
        if size_text and expected_bytes is not None:
            print(
                f"Expected size: {size_text} on model_grids page "
                f"({human_size(expected_bytes)} from HTTP headers)"
            )
        elif size_text:
            print(f"Expected size: {size_text} on model_grids page")
        elif expected_bytes is not None:
            print(f"Expected size: {human_size(expected_bytes)} from HTTP headers")
        else:
            print("Expected size: unknown")

        if not args.go:
            print("Archive is not on disk. Re-run with --go to download it.")
            return 0

        print(f"Downloading to: {archive_path}")
        download_archive(archive_url, archive_path)

    files_by_label = extract_target_files(archive_path, iso_dir)
    print_file_summary(files_by_label)

    if args.clean:
        cleanup_iso_cmd_files(iso_dir, files_by_label, archive_path)
        print("Removed other .iso.cmd files and deleted the archive because --clean was requested.")

    for label in TARGET_FILES:
        print_target_details(files_by_label[label])

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except requests.RequestException as exc:
        print(f"Network error: {exc}", file=sys.stderr)
        raise SystemExit(1)
    except (OSError, tarfile.TarError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
