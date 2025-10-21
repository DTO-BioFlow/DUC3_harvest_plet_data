from __future__ import annotations

import os
import re
import yaml
import s3fs
import hashlib
import logging
import warnings
import unicodedata
import pandas as pd
import datetime as dt
from typing import Dict
from typing import List
from pathlib import Path
from datetime import date
from typing import Optional

from harvest_plet.ospar_comp import OSPARRegions
from harvest_plet.plet import PLETHarvester


def _safe_name(s: str) -> str:
    """
        Sanitize a string for safe use in filenames.

        Normalizes Unicode characters (e.g., ü → u), removes non-ASCII characters,
        replaces any sequence of non-alphanumeric characters with an underscore (_),
        and strips leading/trailing underscores.

        :param s: The input string to sanitize.
        :type s: str

        :returns: The sanitized string suitable for filenames.
        :rtype: str
        """
    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ascii", "ignore").decode()
    s = re.sub(r"[^\w\d]+", "_", s)
    return s.strip("_")


def _limit_filename(name: str, max_len: int = 100) -> str:
    """
    Shorten a filename string to a maximum length, optionally adding a hash
    for uniqueness if truncated.

    If the length of the input string exceeds `max_len`, the string is truncated
    and an 8-character hash derived from the full name is appended to preserve uniqueness.

    :param name: The original filename string.
    :type name: str
    :param max_len: Maximum allowed length for the filename stem (default: 100).
    :type max_len: int

    :returns: A string that does not exceed `max_len` characters (plus hash if truncated).
    :rtype: str
    """
    if len(name) <= max_len:
        return name
    h = hashlib.md5(name.encode("utf-8")).hexdigest()[:8]
    return name[:max_len] + "_" + h


def harvest_for_assessment(start_date: date,
                           end_date: date,
                           out_dir: str | None = None,
                           overwrite: bool = False,
                           logs_dir: str | None = None
                           ) -> None:
    """
      Harvest datasets for all OSPAR regions within a given date range, with caching
      and optional logging.

      Data for each dataset and region combination is retrieved and stored in
      a local cache directory. If the file already exists and `overwrite` is False,
      the data is skipped. Logs are written to a timestamped log file.

      :param start_date: Start date of the data harvest (inclusive).
      :type start_date: date
      :param end_date: End date of the data harvest (inclusive).
      :type end_date: date
      :param out_dir: Directory to store cached CSV files. Defaults to '.cache' if None.
      :type out_dir: Optional[str]
      :param overwrite: Whether to overwrite existing cached files. Defaults to False.
      :type overwrite: bool
      :param logs_dir: Directory to store log files. Defaults to a 'logs' folder
          in the package if None.
      :type logs_dir: Optional[str]

      :returns: None
      :rtype: None
      """
    # default cache dir
    if out_dir is None:
        out_dir = Path(".cache")
    else:
        out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # default logs dir
    if logs_dir is None:
        logs_dir = Path(__file__).resolve().parent / "logs"
    else:
        logs_dir = Path(logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # logfile
    timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = logs_dir / f"logs-{timestamp}.txt"
    logging.basicConfig(
        filename=log_file,
        filemode="w",
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO
    )

    # harvester and regions
    comp_regions = OSPARRegions()
    plet_harvester = PLETHarvester()
    region_ids = comp_regions.get_all_ids()
    dataset_names = plet_harvester.get_dataset_names()

    logging.info(f"Starting harvest run with {len(region_ids)} regions and {len(dataset_names)} datasets.")
    logging.info(f"Cache/output directory: {out_dir}")
    logging.info(f"Logs directory: {logs_dir}")
    logging.info(f"Overwrite enabled: {overwrite}")

    for dataset_idx, dataset_name in enumerate(dataset_names, start=1):
        logging.info(f"=== DATASET {dataset_idx}/{len(dataset_names)}: {dataset_name} ===")
        dataset_safe = _safe_name(dataset_name)

        for region_idx, region_id in enumerate(region_ids, start=1):
            t0 = dt.datetime.now()
            region_safe = _safe_name(region_id)

            # construct filename stem
            filename_stem = (
                f"Dataset_{dataset_safe}_Region_{region_safe}_"
                f"START_{start_date.strftime('%Y-%m-%d')}_STOP_{end_date.strftime('%Y-%m-%d')}"
            )
            filename_stem = _limit_filename(filename_stem, max_len=100)

            filepath = out_dir / f"{filename_stem}.csv"

            try:
                if not overwrite and filepath.exists():
                    logging.info(f"   --- REGION {region_idx}/{len(region_ids)}: {region_id} --- [CACHED] {filepath}")
                    continue

                region_wkt = comp_regions.get_wkt(id=region_id, simplify=True)
                plet_harvester.harvest_data(
                    start_date=start_date,
                    end_date=end_date,
                    wkt=region_wkt,
                    dataset_name=dataset_name,
                    csv=True,
                    out_dir=str(out_dir),
                    name=filename_stem
                )
                t1 = dt.datetime.now()
                logging.info(f"   --- REGION {region_idx}/{len(region_ids)}: {region_id} --- [OK] Duration: {t1 - t0} | Output: {filepath}")

            except Exception as e:
                logging.error(f"   --- REGION {region_idx}/{len(region_ids)}: {region_id} --- [FAILED] Error: {e}")


def _load_and_merge(csv_dir: str | Path) -> pd.DataFrame:
    """
    Internal helper to load all CSVs from a directory, extract dataset/region IDs,
    and merge into a single dataframe with consistent column order.

    :param csv_dir: Directory containing CSV files.
    :type csv_dir: str | Path
    :return: Merged dataframe with dataset_name and region_id as first columns.
    :rtype: pd.DataFrame
    """
    csv_dir = Path(csv_dir)
    if not csv_dir.is_dir():
        raise ValueError(f"Directory does not exist: {csv_dir}")

    csv_files = [f for f in os.listdir(csv_dir) if f.endswith(".csv")]
    if not csv_files:
        raise ValueError(f"No CSV files found in: {csv_dir}")

    dataframes = []
    for file in csv_files:
        file_path = csv_dir / file
        try:
            # skip HTML/error CSVs
            with open(file_path, "r", encoding="utf-8") as f:
                if f.readline().lstrip().startswith("<"):
                    warnings.warn(f"Skipping HTML/error file: {file}")
                    continue

            df = pd.read_csv(file_path, encoding="utf-8")

            # Extract dataset name and region id
            dataset_match = re.search(r"Dataset_(.+?)_Region_", file)
            region_match = re.search(r"_Region_(.+?)_START_", file)

            dataset_name = dataset_match.group(1) if dataset_match else "unknown_dataset"
            region_id = region_match.group(1) if region_match else "unknown_region"

            df.insert(0, "dataset_name", dataset_name)
            df.insert(1, "region_id", region_id)

            df["__source_file__"] = file  # optional trace
            dataframes.append(df)

        except Exception as e:
            warnings.warn(f"Skipping file {file} due to error: {e}")

    if not dataframes:
        raise RuntimeError("No valid CSV files could be read.")

    return pd.concat(dataframes, ignore_index=True)


def to_parquet(
    csv_dir: str = ".cache",
    out_path: str = "merged-data.parquet",
    use_s3: bool = False,
    bucket: Optional[str] = None,
    endpoint_url: Optional[str] = None,
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None,
    aws_session_token: Optional[str] = None,
) -> None:
    """
    Converts all CSV files in a directory to a single merged Parquet file.

    If `use_s3=True`, the file is stored in the given S3/MinIO bucket.

    :param csv_dir: Directory containing CSV files. Defaults to ".cache".
    :param out_path: Output path (local or remote). Defaults to "merged-data.parquet".
    :param use_s3: Whether to save to S3/MinIO.
    :param bucket: S3/MinIO bucket name.
    :param endpoint_url: S3/MinIO endpoint URL.
    :param aws_access_key_id: Access key ID.
    :param aws_secret_access_key: Secret access key.
    :param aws_session_token: Optional session token.
    """
    merged_df = _load_and_merge(csv_dir)

    if use_s3:
        if not all([bucket, endpoint_url, aws_access_key_id, aws_secret_access_key]):
            raise ValueError("Missing S3 credentials or bucket configuration.")

        storage_options = {
            "key": aws_access_key_id,
            "secret": aws_secret_access_key,
            "token": aws_session_token,
            "client_kwargs": {"endpoint_url": endpoint_url},
        }

        s3_path = f"s3://{bucket}/{out_path.lstrip('/')}"
        merged_df.to_parquet(s3_path, index=False, engine="pyarrow", storage_options=storage_options)
        print(f"✅ Parquet file saved to S3: {s3_path}")
    else:
        merged_df.to_parquet(out_path, index=False, engine="pyarrow")
        print(f"✅ Parquet file saved locally at: {out_path}")


def export_to_csv(csv_dir: Optional[str] = None,
                  out_path: Optional[str] = None) -> None:
    """
    Merge CSV files into a single CSV file with dataset/region columns first.

    :param csv_dir: Directory containing CSV files. Defaults to '.cache'.
    :type csv_dir: Optional[str]
    :param out_path: Output CSV file path. Defaults to 'merged.csv'.
    :type out_path: Optional[str]
    :return: None
    :rtype: None
    """
    if csv_dir is None:
        csv_dir = Path(".cache")

    merged_df = _load_and_merge(csv_dir)

    if out_path is None:
        out_path = "merged.csv"

    merged_df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"CSV file saved at: {out_path}")


if __name__ == "__main__":
    from _config import Config

    config = Config()
    print(config.settings)

    start_date = date(2015, 1, 1)
    end_date = date(2025, 1, 1)

    # harvest_for_assessment(start_date=start_date,
    #                        end_date=end_date)

    export_to_csv(
        csv_dir=".cache",
        out_path="C:/Users/willem.boone/Downloads/merged_PLET.csv"
    )

    # Export merged Parquet directly to MinIO/S3
    # to_parquet(
    #     csv_dir=".cache",
    #     out_path="data/merged-data3.parquet",
    #     use_s3=True,
    #     bucket=config.settings.bucket,
    #     endpoint_url=config.settings.endpoint_url,
    #     aws_access_key_id=config.settings.aws_access_key_id,
    #     aws_secret_access_key=config.settings.aws_secret_access_key,
    #     aws_session_token=config.settings.aws_session_token
    # )