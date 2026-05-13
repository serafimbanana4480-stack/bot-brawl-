"""
screenshot_recorder.py

Passive screenshot recorder for dataset pipeline.
Captures raw frames from the emulator via ADB screencap at a configurable
interval and saves them with metadata (timestamp, game state label if available).

Usage:
    python screenshot_recorder.py --adb-id 127.0.0.1:5555 --interval 2.0 --out ./dataset/raw

Output structure:
    dataset/raw/
        YYYYMMDD_HHMMSS_ms.png
        YYYYMMDD_HHMMSS_ms.json   ← metadata sidecar
"""

import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _adb_screencap(adb_path: str, adb_id: str, out_path: Path) -> bool:
    """
    Capture screenshot from emulator via ADB and save to out_path.
    Returns True on success.
    """
    try:
        result = subprocess.run(
            [adb_path, "-s", adb_id, "exec-out", "screencap", "-p"],
            capture_output=True,
            timeout=15,
        )
        if result.returncode != 0:
            logger.error(f"screencap failed: rc={result.returncode} stderr={result.stderr[:200]}")
            return False
        if len(result.stdout) < 1024:
            logger.error(f"screencap returned suspiciously small output ({len(result.stdout)} bytes)")
            return False
        out_path.write_bytes(result.stdout)
        return True
    except subprocess.TimeoutExpired:
        logger.error("screencap timed out")
        return False
    except Exception as e:
        logger.error(f"screencap exception: {e}")
        return False


def _get_adb_path() -> str:
    from .emulator_detector import get_adb_path
    return get_adb_path()


def record_passive(
    adb_id: str,
    output_dir: Path,
    interval_seconds: float = 2.0,
    max_frames: Optional[int] = None,
    adb_path: Optional[str] = None,
) -> None:
    """
    Continuously capture screenshots from the emulator.

    Args:
        adb_id:           ADB device ID (e.g. '127.0.0.1:5555')
        output_dir:       Directory to save frames + metadata
        interval_seconds: Time between captures
        max_frames:       Stop after N frames (None = run indefinitely)
        adb_path:         Path to adb executable (auto-detected if None)
    """
    if adb_path is None:
        adb_path = _get_adb_path()

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting passive recording: adb_id={adb_id}, interval={interval_seconds}s, output={output_dir}")

    frame_count = 0
    consecutive_failures = 0
    MAX_FAILURES = 5

    try:
        while True:
            ts = datetime.utcnow()
            stem = ts.strftime("%Y%m%d_%H%M%S_") + f"{ts.microsecond // 1000:03d}"
            img_path = output_dir / f"{stem}.png"
            meta_path = output_dir / f"{stem}.json"

            success = _adb_screencap(adb_path, adb_id, img_path)

            if success:
                consecutive_failures = 0
                frame_count += 1

                meta = {
                    "captured_at": ts.isoformat(),
                    "adb_id": adb_id,
                    "frame_index": frame_count,
                    "file": img_path.name,
                    "label": None,
                    "game_state": None,
                    "annotated": False,
                }
                meta_path.write_text(json.dumps(meta, indent=2))

                logger.info(f"Frame {frame_count:05d} saved: {img_path.name} ({img_path.stat().st_size} bytes)")

                if max_frames and frame_count >= max_frames:
                    logger.info(f"Reached max_frames={max_frames}, stopping.")
                    break
            else:
                consecutive_failures += 1
                logger.warning(f"Capture failed ({consecutive_failures}/{MAX_FAILURES})")
                if consecutive_failures >= MAX_FAILURES:
                    logger.error("Too many consecutive failures. Stopping recorder.")
                    break

            time.sleep(interval_seconds)

    except KeyboardInterrupt:
        logger.info(f"Recording interrupted. {frame_count} frames saved to {output_dir}")

    logger.info(f"Recording complete. Total frames: {frame_count}")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Passive screenshot recorder for Brawl Stars dataset.")
    parser.add_argument("--adb-id", required=True, help="ADB device ID (e.g. 127.0.0.1:5555)")
    parser.add_argument("--interval", type=float, default=2.0, help="Seconds between captures")
    parser.add_argument("--out", type=Path, default=Path("./dataset/raw"), help="Output directory")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames to capture")
    parser.add_argument("--adb-path", type=str, default=None, help="Path to adb executable")
    args = parser.parse_args()

    record_passive(
        adb_id=args.adb_id,
        output_dir=args.out,
        interval_seconds=args.interval,
        max_frames=args.max_frames,
        adb_path=args.adb_path,
    )


if __name__ == "__main__":
    main()
