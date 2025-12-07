# HRRR Pressure Levels to Zarr

Automated pipeline to download NOAA HRRR (High-Resolution Rapid Refresh) pressure-level forecast data and convert it to Zarr format. Modeled after the [GFS Pressure Zarr](https://github.com/andrewnakas/Gfs_Pressure_Zarr) project.

## Overview

This project:
- Downloads HRRR pressure-level GRIB2 files from NOAA's AWS bucket
- Converts data to compressed Zarr format
- Maintains only the **latest forecast run** (no historical data accumulation)
- Runs automatically via GitHub Actions every 6 hours

## Data Details

- **Model**: NOAA HRRR (High-Resolution Rapid Refresh)
- **Domain**: Continental United States (CONUS)
- **Resolution**: 3 km
- **Forecast Hours**: 0-48 (hourly)
- **Update Frequency**: Every 6 hours
- **Levels**: All available pressure levels (isobaricInhPa)
- **Format**: Zarr (compressed with Blosc/Zstd)

## Dataset Access

The latest HRRR pressure level data is available on the `data` branch:

```python
import xarray as xr
import fsspec

# Download and open the latest dataset
url = "https://github.com/andrewnakas/hrrr_pressure_levels_to_zarr/raw/data/hrrr_latest.zarr.zip"

with fsspec.open(url, mode='rb') as f:
    ds = xr.open_zarr(f.read())

print(ds)
```

Or download directly:
```bash
wget https://github.com/andrewnakas/hrrr_pressure_levels_to_zarr/raw/data/hrrr_latest.zarr.zip
unzip hrrr_latest.zarr.zip
```

## Usage

### Local Execution

```bash
# Install dependencies
pip install -r requirements.txt

# Run with default settings (0-48 hour forecast)
python scripts/update_hrrr_zarr.py --output hrrr_latest.zarr --zip

# Customize forecast hours
python scripts/update_hrrr_zarr.py --forecast-hours "0 6 12 18 24 30 36 42 48" --output hrrr_subset.zarr

# Specify pressure levels (hPa)
python scripts/update_hrrr_zarr.py --levels "1000 925 850 700 500 250" --output hrrr_levels.zarr
```

### Configuration

Environment variables (or command-line flags):

- `FORECAST_HOURS`: Space-separated forecast hours (default: 0-48)
- `LEVELS_HPA`: Space-separated pressure levels in hPa (default: all)
- `PARAM_SHORTNAMES`: Specific variables to extract (default: auto-discover all)
- `CYCLE_OFFSET_HOURS`: Hours to subtract from current time when selecting cycle (default: 2)
- `MAX_ZARR_BYTES`: Maximum allowed size in bytes (default: 1,900,000,000)
- `DTYPE`: Output data type (default: float16)

## How It Works

### Cleanup Strategy

The project uses an **orphan branch strategy** to maintain only the latest data:

1. **Orphan Branch**: Each update creates a fresh `data` branch with no commit history
2. **Force Push**: Completely overwrites previous data
3. **No Accumulation**: Old forecasts are permanently deleted
4. **Git LFS**: Large files stored efficiently outside Git history

### Workflow Schedule

- Runs every 6 hours at :15 past the hour (00:15, 06:15, 12:15, 18:15 UTC)
- Can be triggered manually via GitHub Actions

### Disk Space Management

The workflow aggressively manages disk space:
- Frees ~40GB by removing unused software
- Deletes GRIB files immediately after processing
- Removes uncompressed Zarr directories
- Clears Python caches

## File Structure

```
hrrr_pressure_levels_to_zarr/
├── .github/
│   └── workflows/
│       └── update.yml           # GitHub Actions workflow
├── scripts/
│   └── update_hrrr_zarr.py     # Main data processing script
├── .gitattributes              # Git LFS configuration
├── .gitignore
├── README.md
└── requirements.txt            # Python dependencies
```

## Data Branch

The `data` branch contains:
- `hrrr_latest.zarr.zip`: Compressed Zarr archive of latest forecast
- `latest_metadata.json`: Metadata about the forecast run

Example metadata:
```json
{
  "cycle_date": "2025-12-06",
  "cycle_hour_utc": 12,
  "forecast_hours": [0, 1, 2, ..., 48],
  "source": "https://noaa-hrrr-bdp-pds.s3.amazonaws.com",
  "generated_utc": "2025-12-06T14:30:00+00:00"
}
```

## Requirements

- Python 3.11+
- xarray
- cfgrib
- zarr
- numcodecs
- dask
- requests
- libeccodes (system dependency)

## Related Projects

- [GFS Pressure Zarr](https://github.com/andrewnakas/Gfs_Pressure_Zarr) - GFS global forecast data
- [Dynamical HRRR Catalog](https://dynamical.org/catalog/noaa-hrrr-forecast-48-hour/) - Full HRRR archive

## License

MIT

## Data Source

HRRR data is provided by NOAA and hosted on AWS Open Data:
- **Source**: NOAA High-Resolution Rapid Refresh (HRRR)
- **Bucket**: `noaa-hrrr-bdp-pds`
- **License**: Public domain (US Government data)
