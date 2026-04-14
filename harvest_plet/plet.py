import os
import re
import csv
import io
import time
import json
import copy
import requests
import warnings
from typing import List
from typing import Dict
from typing import Union
from typing import Optional
from datetime import date
from pathlib import Path
from bs4 import BeautifulSoup
from shapely import wkt as wkt_loader


class PLETHarvester:
    """
    Interface to interact with the DASSH PLET database for harvesting marine
    biological datasets.

    Provides functionality to list datasets, query specific data using
    spatiotemporal filters (including OSPAR COMP areas), and save results to
    CSV.

    The default API endpoint can be overridden per instance by assigning to
    ``base_url`` or ``BASE_URL`` after construction.
    """

    DEFAULT_INSTANCE: str = "PLET"
    BASE_URL: str = "https://www.dassh.ac.uk/plet/cgi-bin/get_form.py"
    SITE_URL: str = "https://www.dassh.ac.uk/lifeforms/"
    _instances_cache: Optional[Dict[str, Dict[str, str]]] = None

    def __init__(self) -> None:
        """
        Initializes the PLETHarvester instance with a persistent requests
        session.
        """
        self.session = requests.Session()
        self._instance_name: Optional[str] = None
        self.set_instance(self.DEFAULT_INSTANCE)

    @classmethod
    def _get_endpoints_path(cls) -> Path:
        """Return the local path of the endpoint instance registry."""
        return Path(__file__).resolve().parent / "config" / "endpoints.json"

    @classmethod
    def _load_instances(cls) -> Dict[str, Dict[str, str]]:
        """Load endpoint instances from config and cache them."""
        if cls._instances_cache is None:
            endpoints_path = cls._get_endpoints_path()
            try:
                with endpoints_path.open("r", encoding="utf-8") as f_in:
                    data = json.load(f_in)
            except FileNotFoundError as exc:
                raise RuntimeError(
                    f"Endpoint config not found: {endpoints_path}") from exc
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Endpoint config is not valid JSON: {endpoints_path}") from exc

            if not isinstance(data, dict):
                raise RuntimeError(
                    "Endpoint config must contain a top-level JSON object")
            cls._instances_cache = data
        return cls._instances_cache

    @classmethod
    def get_instances(cls) -> Dict[str, Dict[str, str]]:
        """Return all configured endpoint instances from endpoints.json."""
        return copy.deepcopy(cls._load_instances())

    def set_instance(self, instance_name: str) -> None:
        """Switch both URLs to a named instance from endpoints.json."""
        if not isinstance(instance_name, str) or not instance_name.strip():
            raise ValueError("instance_name must be a non-empty string")

        instance_key = instance_name.strip()
        instances = self._load_instances()
        if instance_key not in instances:
            available = ", ".join(sorted(instances.keys()))
            raise ValueError(
                f"Unknown instance '{instance_key}'. Available instances: {available}")

        config = instances[instance_key]
        try:
            self.base_url = config["BASE_URL"]
            self.site_url = config["SITE_URL"]
        except KeyError as exc:
            raise ValueError(
                f"Instance '{instance_key}' is missing required key: {exc}") from exc

        self._instance_name = instance_key

    def get_instance(self) -> Optional[str]:
        """Return the currently active instance key, if selected."""
        return self._instance_name

    @property
    def base_url(self) -> str:
        """Return the currently configured harvest endpoint."""
        return self.BASE_URL

    @base_url.setter
    def base_url(self, value: str) -> None:
        """Override the harvest endpoint for this instance."""
        if not value:
            raise ValueError("base_url must be a non-empty string")
        self.BASE_URL = value

    def set_base_url(self, value: str) -> None:
        """Convenience method for updating the harvest endpoint."""
        self.base_url = value

    def get_base_url(self) -> str:
        """Convenience method for reading the harvest endpoint."""
        return self.base_url

    @property
    def site_url(self) -> str:
        """Return the currently configured website endpoint."""
        return self.SITE_URL

    @site_url.setter
    def site_url(self, value: str) -> None:
        """Override the website endpoint for this instance."""
        if not value:
            raise ValueError("site_url must be a non-empty string")
        self.SITE_URL = value

    def set_site_url(self, value: str) -> None:
        """Convenience method for updating the website endpoint."""
        self.site_url = value

    def get_site_url(self) -> str:
        """Convenience method for reading the website endpoint."""
        return self.site_url

    def get_dataset_names(self) -> List[str]:
        """
        Retrieve all available dataset names from the DASSH website.

        :returns: A list of dataset names available for download.
        :rtype: List[str]
        """
        response = self.session.get(self.site_url)
        response.encoding = response.apparent_encoding  # Fix encoding issues
        soup = BeautifulSoup(response.text, 'html.parser')

        select_element = soup.find("select",
                                   {"id": "abundance_dataset"})
        if not select_element:
            print("Select element not found.")
            return []

        options = [
            option.text.strip()
            for option in select_element.find_all("option")
            if
            option.text.strip()
            and "select a dataset"
            not in option.text.lower()
        ]
        return options

    def harvest_data(
            self,
            start_date: date,
            end_date: date,
            wkt: str,
            dataset_name: str,
            csv: bool = False,
            out_dir: Optional[str] = None,
            name: Optional[str] = None,
            retries: int = 3,
            backoff_factor: float = 60.0,
            timeout: float = 600.0
    ) -> Optional[str]:
        """
        Harvest dataset from DASSH API for a given time range and spatial region.
        Optionally write output to a CSV file.

        :param start_date: Start date of query.
        :param end_date: End date of query.
        :param wkt: WKT string representing the polygon region.
        :param dataset_name: Dataset name to retrieve.
        :param csv: If True, save as CSV file. If False, return as string.
        :param out_dir: Directory to save the CSV file (required if csv=True).
        :param name: Name of the CSV file (required if csv=True).
        :param retries: Number of retry attempts on failure.
        :param backoff_factor: Backoff multiplier for retry wait time.
        :param timeout: Request timeout in seconds.
        :returns: CSV string if csv=False, otherwise None.
        :raises ValueError: For invalid inputs.
        :raises RuntimeError: If request fails after all retries.
        """
        if end_date <= start_date:
            raise ValueError("end_date must be after start_date")

        try:
            geom = wkt_loader.loads(wkt)
            if not geom.is_valid:
                raise ValueError("Provided WKT geometry is invalid")
        except Exception as e:
            raise ValueError(f"Invalid WKT format: {e}, WKT: {wkt}")

        params = {
            "startdate": start_date.strftime("%Y-%m-%d"),
            "enddate": end_date.strftime("%Y-%m-%d"),
            "wkt": wkt,
            "abundance_dataset": dataset_name,
            "format": "csv",
        }

        for attempt in range(1, retries + 1):
            try:
                response = self.session.get(self.base_url, params=params,
                                            timeout=timeout)
                response.raise_for_status()
                csv_data = response.text

                if "<h2>" in csv_data and "Error:" in csv_data:
                    warnings.warn(
                        "API returned an error page instead of data: "
                        "likely no samples found for the given parameters.",
                        RuntimeWarning
                    )

                if csv:
                    if not out_dir or not name:
                        raise ValueError(
                            "out_dir and name must be provided when csv=True")
                    os.makedirs(out_dir, exist_ok=True)
                    clean_name = self._sanitize_path_name(name)
                    dest_path = os.path.join(out_dir, f"{clean_name}.csv")
                    self._write_csv_from_string(csv_data, dest_path)
                    return None
                else:
                    return csv_data

            except requests.RequestException as e:
                print(f"[Attempt {attempt}] Request failed: {e}")
                if attempt == retries:
                    raise RuntimeError(
                        f"Request failed after {retries} attempts: {e}")
                sleep_time = backoff_factor * (2 ** (attempt - 1))
                print(f"Retrying in {sleep_time:.1f} seconds...")
                time.sleep(sleep_time)

    def _harvest_all_datasets(
        self,
        start_date: date = date(2010, 1, 1),
        end_date: date = date(2021, 1, 1),
        wkt: str = "POLYGON ((-180 -90,-180 90,180 90,180 -90,-180 -90))",
        out_dir: str = "",
        retries: int = 3,
        backoff_factor: float = 60.0,
        timeout: float = 600.0
    ) -> Dict[str, List[str]]:
        """
        old & depricated code
        """
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir, exist_ok=True)
            print(f"Created output directory: {out_dir}")

        succeeded: List[str] = []
        failed: List[str] = []

        for dataset in self.get_dataset_names():
            print(f"Attempting to harvest dataset: {dataset}")
            try:
                self.harvest_as_csv(
                    dataset_name=dataset,
                    start_date=start_date,
                    end_date=end_date,
                    wkt=wkt,
                    out_dir=out_dir,
                    retries=retries,
                    backoff_factor=backoff_factor,
                    timeout=timeout,
                )
                succeeded.append(dataset)
            except Exception as e:
                failed.append(dataset)
                print(f"Failed to harvest dataset: {dataset}. Error: {e}")

        return {"succeeded": succeeded, "failed": failed}

    @staticmethod
    def _sanitize_path_name(name: str) -> str:
        """
        Clean string for safe use as file or directory name.

        :param name: String to sanitize.
        :type name: str

        :returns: Sanitized name in lowercase with no special characters.
        :rtype: str
        """
        name = name.replace(" ", "_")
        name = re.sub(r'[^\w\-]', '', name)
        return name

    @staticmethod
    def _write_csv_from_string(
        data_str: str,
        output_file: Union[str, bytes, os.PathLike],
    ) -> None:
        """
        Write CSV-formatted string data to disk.

        :param data_str: String containing CSV data.
        :type data_str: str
        :param output_file: Path to the output file.
        :type output_file: Union[str, bytes, os.PathLike]

        :returns: None
        :rtype: None
        """
        csv_buffer = io.StringIO(data_str)
        reader = csv.reader(csv_buffer)

        with open(output_file, 'w', newline='', encoding='utf-8') as f_out:
            writer = csv.writer(f_out)
            for row in reader:
                writer.writerow(row)

        # print(f"CSV written to {output_file}")


if __name__ == "__main__":
    plet_harvester = PLETHarvester()

    # test dateset names
    dataset_names = plet_harvester.get_dataset_names()
    print(dataset_names)
