#!/usr/bin/env python3
"""Download the latest HRRR pressure-level GRIB2 files and convert to Zarr.

Designed for GitHub Actions but runnable locally. By default it:
- picks the most recent HRRR cycle that is likely published (UTC now minus 2h)
- downloads specified forecast hours from the public NOAA HRRR bucket
- keeps only pressure-level fields
- writes a consolidated, chunked Zarr store
- optionally zips the Zarr directory for easier transport/storage
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import requests
import xarray as xr
from cfgrib.dataset import DatasetBuildError
from zarr.codecs import Blosc


logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
LOGGER = logging.getLogger("update_hrrr_zarr")


def parse_forecast_hours(text: str | None) -> List[int]:
    if not text:
        return []
    hours: List[int] = []
    for token in text.replace(",", " ").split():
        if not token.strip():
            continue
        hours.append(int(token))
    return sorted(set(hours))


def parse_shortnames(text: str | None) -> List[str]:
    if text is None:
        return []
    if text.strip() == "":
        return []
    return [tok.strip() for tok in text.replace(",", " ").split() if tok.strip()]


def parse_levels(text: str | None) -> List[int]:
    if not text:
        return []
    return [int(tok) for tok in text.replace(",", " ").split() if tok.strip()]


def default_forecast_string() -> str:
    """HRRR 48-hour forecasts available on 00/06/12/18Z cycles."""
    return " ".join(str(h) for h in range(0, 49))


def cycle_candidates(now: datetime, max_back_cycles: int) -> Iterable[Tuple[datetime.date, int]]:
    """Yield candidate (date, cycle_hour) pairs stepping back 6h at a time.

    HRRR 48-hour forecasts are only available on 00Z, 06Z, 12Z, 18Z cycles.
    Hourly cycles (01-05Z, 07-11Z, etc.) only have 18-hour forecasts.
    """
    for back in range(max_back_cycles + 1):
        candidate_time = now - timedelta(hours=6 * back)
        # Round to nearest 6-hour cycle (00, 06, 12, 18)
        cycle_hour = (candidate_time.hour // 6) * 6
        yield candidate_time.date(), cycle_hour


def build_url(base_url: str, date, cycle_hour: int, forecast_hour: int) -> str:
    """Build URL for HRRR GRIB2 file.

    HRRR URL pattern:
    https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.YYYYMMDD/conus/hrrr.tHHz.wrfprsf{forecast_hour}.grib2
    """
    date_str = date.strftime("%Y%m%d")
    path = f"hrrr.{date_str}/conus"
    filename = f"hrrr.t{cycle_hour:02d}z.wrfprsf{forecast_hour:02d}.grib2"
    return f"{base_url.rstrip('/')}/{path}/{filename}"


def download_file(url: str, dest: Path, retries: int = 2) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, timeout=30) as resp:
                if resp.status_code == 404:
                    raise FileNotFoundError(url)
                resp.raise_for_status()
                with dest.open("wb") as f:
                    for chunk in resp.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
            return
        except (requests.HTTPError, requests.ConnectionError, requests.Timeout) as exc:
            if attempt == retries:
                raise
            sleep_for = 2
            LOGGER.warning("%s (attempt %s/%s), retrying in %ss", exc, attempt, retries, sleep_for)
            import time

            time.sleep(sleep_for)


def load_pressure_dataset(grib_path: Path, shortnames: List[str], levels: Sequence[int]) -> xr.Dataset:
    """Open a GRIB, falling back to per-variable loads when cfgrib hits coord conflicts."""

    base_kwargs = {
        "filter_by_keys": {"typeOfLevel": "isobaricInhPa"},
        "indexpath": "",  # avoid writing index sidecars
        "errors": "ignore",  # prefer partial success over failure
    }

    # Auto-discover shortNames if none provided
    if not shortnames:
        try:
            # Open dataset to discover available variables (shortNames are variable names)
            temp_ds = xr.open_dataset(
                grib_path,
                engine="cfgrib",
                backend_kwargs=base_kwargs,
            )
            shortnames = sorted(list(temp_ds.data_vars.keys()))
            temp_ds.close()
            LOGGER.info("Discovered %d pressure-level shortNames", len(shortnames))
        except Exception:
            LOGGER.warning("Could not auto-discover shortNames; falling back to cfgrib full open")
            shortnames = []

    LOGGER.info("Opening %s", grib_path.name)
    try:
        ds = xr.open_dataset(
            grib_path,
            engine="cfgrib",
            backend_kwargs=base_kwargs,
        )
    except DatasetBuildError:
        LOGGER.warning("cfgrib merge conflict; retrying per-shortName subset")
        datasets = []
        levels_union = None
        target_levels = levels if levels else None
        for sn in shortnames:
            kw = dict(base_kwargs)
            kw["filter_by_keys"] = {**kw["filter_by_keys"], "shortName": sn}
            try:
                part = xr.open_dataset(grib_path, engine="cfgrib", backend_kwargs=kw)
                if "isobaricInhPa" in part.coords:
                    if target_levels:
                        part = part.sel(isobaricInhPa=[lev for lev in target_levels if lev in part.isobaricInhPa.values])
                    if levels_union is None:
                        levels_union = list(part.isobaricInhPa.values)
                    else:
                        levels_union = sorted(set(levels_union) | set(part.isobaricInhPa.values))
                datasets.append(part)
            except DatasetBuildError:
                LOGGER.warning("Skipping shortName=%s due to cfgrib error", sn)
                continue

        if not datasets:
            raise

        # Align to common pressure axis so merge succeeds
        if levels_union is not None:
            for i, part in enumerate(datasets):
                if "isobaricInhPa" in part.coords:
                    datasets[i] = part.reindex(isobaricInhPa=levels_union)

        ds = xr.merge(datasets, compat="override", combine_attrs="drop_conflicts")

    # Ensure pressure levels ascending and add valid_time for convenience
    if "isobaricInhPa" in ds.coords:
        if levels:
            keep = [lev for lev in levels if lev in ds.isobaricInhPa.values]
            if keep:
                ds = ds.sel(isobaricInhPa=keep)
        ds = ds.sortby("isobaricInhPa")
    if "time" in ds.coords and "step" in ds.coords:
        ds = ds.assign_coords(valid_time=ds.time + ds.step)
    # Light chunking to keep memory reasonable
    ds = ds.chunk({"isobaricInhPa": 8, "y": 90, "x": 90})
    return ds


def save_zarr(ds: xr.Dataset, output_path: Path, zip_output: bool, max_bytes: int | None = None) -> Path:
    if output_path.exists():
        shutil.rmtree(output_path)
    compressor = Blosc(cname="zstd", clevel=4, shuffle=2)
    encoding = {name: {"compressors": compressor, "dtype": ds[name].dtype} for name in ds.data_vars}
    LOGGER.info("Writing Zarr → %s", output_path)
    try:
        ds.to_zarr(output_path, mode="w", consolidated=True, encoding=encoding)
        LOGGER.info("Zarr write completed")
    except Exception as e:
        LOGGER.error("Zarr write failed: %s", e)
        raise
    if zip_output:
        archive_path = output_path.with_suffix(output_path.suffix + ".zip")
        if archive_path.exists():
            archive_path.unlink()
        LOGGER.info("Zipping Zarr → %s", archive_path.name)
        shutil.make_archive(
            base_name=str(output_path),
            format="zip",
            root_dir=output_path.parent,
            base_dir=output_path.name,
        )
        size_mb = archive_path.stat().st_size / 1e6
        LOGGER.info("Zarr archive size: %.1f MB", size_mb)
        if max_bytes and archive_path.stat().st_size > max_bytes:
            raise ValueError(
                f"Zarr archive {size_mb:.1f} MB exceeds max_bytes={max_bytes/1e6:.1f} MB"
            )
        return archive_path
    else:
        if max_bytes and output_path.stat().st_size > max_bytes:
            raise ValueError(
                f"Zarr store {output_path.stat().st_size/1e6:.1f} MB exceeds max_bytes={max_bytes/1e6:.1f} MB"
            )
        return output_path


def write_metadata(meta_path: Path, *, cycle_date, cycle_hour: int, forecast_hours: List[int], source_url: str):
    meta = {
        "cycle_date": cycle_date.strftime("%Y-%m-%d"),
        "cycle_hour_utc": cycle_hour,
        "forecast_hours": forecast_hours,
        "source": source_url,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    LOGGER.info("Wrote metadata → %s", meta_path)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--forecast-hours",
        dest="forecast_hours",
        default=os.getenv("FORECAST_HOURS", default_forecast_string()),
        help="Space separated forecast hours (0-48 available on 00/06/12/18Z cycles)",
    )
    parser.add_argument(
        "--cycle-offset",
        dest="cycle_offset",
        type=int,
        default=int(os.getenv("CYCLE_OFFSET_HOURS", 5)),
        help="Hours to back off from now(UTC) when picking cycle",
    )
    parser.add_argument(
        "--max-back-cycles",
        dest="max_back_cycles",
        type=int,
        default=3,
        help="How many 6-hour cycles to fall back if newest is missing",
    )
    parser.add_argument(
        "--base-url",
        dest="base_url",
        default=os.getenv("BASE_URL", "https://noaa-hrrr-bdp-pds.s3.amazonaws.com"),
    )
    parser.add_argument(
        "--output",
        dest="output",
        default=os.getenv("OUTPUT_ZARR", "hrrr_latest.zarr"),
    )
    parser.add_argument(
        "--zip",
        dest="zip_output",
        action=argparse.BooleanOptionalAction,
        default=os.getenv("ZIP_OUTPUT", "1") == "1",
        help="Also emit a .zip of the Zarr store",
    )
    parser.add_argument(
        "--tmp-dir",
        dest="tmp_dir",
        default=os.getenv("TMP_DIR", "/tmp/hrrr"),
    )
    parser.add_argument(
        "--params",
        dest="params",
        default=os.getenv("PARAM_SHORTNAMES", ""),
        help="Space separated shortName list to keep if cfgrib has conflicts; empty auto-discovers all pressure-level vars",
    )
    parser.add_argument(
        "--levels",
        dest="levels",
        default=os.getenv("LEVELS_HPA", ""),
        help="Space separated pressure levels (hPa) to retain; leave empty to keep all",
    )
    parser.add_argument(
        "--max-bytes",
        dest="max_bytes",
        type=int,
        default=int(os.getenv("MAX_ZARR_BYTES", 4_500_000_000)),
        help="Abort if zipped store exceeds this many bytes (prevents GitHub/LFS failures)",
    )
    parser.add_argument(
        "--dtype",
        dest="dtype",
        default=os.getenv("DTYPE", "float16"),
        help="Output dtype for data variables (e.g., float32, float16)",
    )
    args = parser.parse_args(argv)

    forecast_hours = parse_forecast_hours(args.forecast_hours)
    if not forecast_hours:
        raise SystemExit("No forecast hours provided")
    shortnames = parse_shortnames(args.params)
    levels = parse_levels(args.levels)

    now = datetime.now(timezone.utc) - timedelta(hours=args.cycle_offset)
    candidates = list(cycle_candidates(now, args.max_back_cycles))

    tmp_dir = Path(args.tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    def download_and_process(fh: int, cycle_date, cycle_hour: int) -> tuple:
        """Download and process a single forecast hour. Returns (fh, dataset) or (fh, None) on error."""
        url = build_url(args.base_url, cycle_date, cycle_hour, fh)
        grib_path = tmp_dir / Path(url).name
        try:
            download_file(url, grib_path)
            ds = load_pressure_dataset(grib_path, shortnames, levels)
            # CRITICAL: Load into memory NOW while we only have 1 dataset
            # This prevents I/O overload when writing 49 lazy datasets to Zarr
            ds = ds.compute()
            # Now safe to delete - data is in memory
            grib_path.unlink(missing_ok=True)
            LOGGER.info("Loaded and deleted %s", grib_path.name)
            return (fh, ds)
        except FileNotFoundError:
            return (fh, None)
        except Exception as e:
            LOGGER.warning("Failed to process forecast hour %d: %s", fh, e)
            return (fh, None)

    for cycle_date, cycle_hour in candidates:
        LOGGER.info("Trying cycle %s %02dZ", cycle_date.isoformat(), cycle_hour)
        try:
            # Download and process forecast hours sequentially
            # Each file is loaded into memory individually to avoid I/O overload
            datasets_dict = {}
            for i, fh in enumerate(forecast_hours):
                LOGGER.info("Processing forecast hour %d/%d (f%02d)", i+1, len(forecast_hours), fh)
                fh_result, ds = download_and_process(fh, cycle_date, cycle_hour)
                if ds is not None:
                    datasets_dict[fh_result] = ds
                else:
                    LOGGER.warning("Forecast hour %d not available for cycle %s %02dZ", fh, cycle_date, cycle_hour)
                    # If early hours are missing, cycle not ready
                    if fh < 6 and len(datasets_dict) < 6:
                        raise FileNotFoundError(f"Cycle {cycle_date} {cycle_hour:02d}Z not ready yet")
                    # If we have at least 24 hours, that's sufficient
                    if len(datasets_dict) >= 24:
                        LOGGER.info("Stopping at %d forecast hours (sufficient data)", len(datasets_dict))
                        break

            if not datasets_dict:
                raise FileNotFoundError(f"No data available for cycle {cycle_date} {cycle_hour:02d}Z")

            # If we have less than 24 hours and early hours are missing, try next cycle
            if len(datasets_dict) < 24:
                missing_early = any(fh < 12 for fh in forecast_hours[:12] if fh not in datasets_dict)
                if missing_early:
                    raise FileNotFoundError(f"Insufficient early forecast hours for cycle {cycle_date} {cycle_hour:02d}Z")

            # Sort datasets by forecast hour
            successful_hours = sorted(datasets_dict.keys())
            datasets = [datasets_dict[fh] for fh in successful_hours]
            LOGGER.info("Successfully retrieved %d forecast hours: %s", len(successful_hours), successful_hours)

            combined = xr.concat(datasets, dim="step", combine_attrs="drop_conflicts")
            if args.dtype:
                combined = combined.astype(args.dtype)
            output_path = Path(args.output)

            # Check disk space before write
            import subprocess
            df_result = subprocess.run(["df", "-h", str(output_path.parent)], capture_output=True, text=True)
            LOGGER.info("Disk space before Zarr save:\n%s", df_result.stdout)

            LOGGER.info("Starting Zarr save with %d forecast hours", len(successful_hours))
            archive_path = save_zarr(
                combined, output_path, zip_output=args.zip_output, max_bytes=args.max_bytes
            )
            LOGGER.info("Zarr save completed successfully")

            metadata_path = output_path.parent / "latest_metadata.json"
            write_metadata(
                metadata_path,
                cycle_date=cycle_date,
                cycle_hour=cycle_hour,
                forecast_hours=successful_hours,
                source_url=args.base_url,
            )

            # Clean up tmp directory
            shutil.rmtree(tmp_dir, ignore_errors=True)

            LOGGER.info("Done. Output at %s", archive_path)
            return 0
        except FileNotFoundError:
            LOGGER.warning("Cycle %s %02dZ not available (404). Trying previous cycle...", cycle_date, cycle_hour)
            continue
        except Exception:
            LOGGER.exception("Cycle %s %02dZ failed", cycle_date, cycle_hour)
            continue

    LOGGER.error("No available cycles in last %s attempts", len(candidates))
    return 1


if __name__ == "__main__":
    sys.exit(main())
