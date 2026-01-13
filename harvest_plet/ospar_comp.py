import os
import requests
from shapely import wkt
from typing import List
from typing import Optional
from staticmap import Polygon as st_Polygon
from staticmap import StaticMap
import matplotlib.pyplot as plt
from shapely.geometry import shape
from shapely.geometry import Polygon
from shapely.geometry import MultiPolygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union


class OSPARRegions:
    """
    A class for fetching and handling OSPAR WFS component data.
    Description of data: https://odims.ospar.org/en/submissions/ospar_comp_au_2023_01/
    json url: https://odims.ospar.org/geoserver/odims/wfs?service=WFS&version=2.0.0&request=GetFeature&typeName=ospar_comp_au_2023_01_001&outputFormat=application/json
    """

    def __init__(self) -> None:
        self.url = ("https://odims.ospar.org/geoserver/odims/wfs?service=WFS&"
                    "version=2.0.0&request=GetFeature&"
                    "typeName=ospar_comp_au_2023_01_001"
                     "&outputFormat=application/json")
        self.data = self._get_json()

    def _get_json(self) -> dict:
        try:
            response = requests.get(self.url)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise IOError(f"Error accessing or reading GeoJSON data: {e}")

    def _get_feature_by_id(self, id: str) -> Optional[dict]:
        for feature in self.data.get("features", []):
            if feature["properties"].get("ID") == id:
                return feature
        return None

    def _get_geometry(self, id: str) -> Optional[BaseGeometry]:
        feature = self._get_feature_by_id(id)
        if feature:
            return shape(feature["geometry"])
        return None

    def get_wkt(self, id: str, simplify: bool = False) -> Optional[str]:
        """
        Retrieve the WKT (Well-Known Text) geometry string for a given feature ID.

        Optionally simplifies the geometry to reduce its size while preserving
        topology. Tiny polygons are removed and coordinates are rounded to
        0.01 precision.

        :param id: The unique identifier of the feature.
        :type id: str
        :param simplify: Whether to simplify the geometry before returning.
        :type simplify: bool

        :returns: The WKT string of the geometry, or None if not found.
        :rtype: Optional[str]
        """
        geometry: BaseGeometry = self._get_geometry(id)
        if geometry is None:
            return None

        if simplify:
            tolerance = 0.01  # starting simplification tolerance
            max_len = 5000  # target max WKT length
            simplified = geometry

            while True:
                # Simplify geometry with topology preserved
                simplified = simplified.simplify(tolerance,
                                                 preserve_topology=True)

                # Merge close/overlapping polygons
                if isinstance(simplified, (MultiPolygon, Polygon)):
                    simplified = unary_union(simplified)

                # Remove tiny polygons (area threshold)
                if isinstance(simplified, MultiPolygon):
                    simplified = MultiPolygon(
                        [p for p in simplified.geoms if p.area > 0.0001])
                    if len(simplified.geoms) == 1:
                        simplified = simplified.geoms[0]

                # Fix any invalid geometries
                simplified = simplified.buffer(0)

                # Round coordinates to 0.01 precision
                simplified = wkt.loads(
                    wkt.dumps(simplified, rounding_precision=2))

                # Stop if WKT is small enough or tolerance too high
                if len(simplified.wkt) <= max_len or tolerance >= 1.0:
                    break

                tolerance *= 2  # increase simplification tolerance

            geometry = simplified
        else:
            # Even without simplification, round coordinates to 0.01 precision
            geometry = wkt.loads(wkt.dumps(geometry, rounding_precision=2))

        return geometry.wkt

    def get_all_ids(self) -> List[str]:
        """
        Get a list of all feature IDs in the dataset.

        :returns: A list of feature IDs.
        :rtype: List[str]
        """
        return [
            feature["properties"].get("ID")
            for feature in self.data.get("features", [])
            if feature["properties"].get("ID")
        ]

    def plot_map(
            self,
            id: Optional[str] = None,
            show: bool = True,
            output_dir: Optional[str] = None
    ) -> None:
        """
        Plot the geometry of a specific feature ID or all features on a static map.

        If an ID is provided, only that feature is plotted. Otherwise, all
        features in the dataset are plotted.

        :param id: Feature ID to plot. If None, plots all features.
        :type id: Optional[str]
        :param show: Whether to display the plot interactively.
        :type show: bool
        :param output_dir: Directory to save the plot image. If None, plot is not saved.
        :type output_dir: Optional[str]
        :returns: None
        :rtype: None
        """
        # Get the list of features to plot
        features = self.data.get("features", [])
        if not features:
            raise ValueError("No features found in dataset.")

        # Filter by ID if provided
        if id:
            features = [f for f in features if f["properties"].get("ID") == id]
            if not features:
                raise ValueError(
                    f"No feature with ID '{id}' found in dataset.")
            filename = f"{id}.png"
            title = f"OSPAR Region ID: {id}"
        else:
            filename = "ospar_all_regions.png"
            title = "All OSPAR Regions"

        # Initialize map
        m = StaticMap(800, 800)

        # Helper function for geometry coordinate extraction
        def extract_coords(geom):
            """Convert shapely geometry to list(s) of (lon, lat) tuples."""
            if geom.is_empty:
                return []

            if geom.geom_type == "Polygon":
                return [(x, y) for x, y in geom.exterior.coords]

            elif geom.geom_type == "MultiPolygon":
                return [
                    [(x, y) for x, y in poly.exterior.coords]
                    for poly in geom.geoms
                    if not poly.is_empty
                ]

            else:
                return []

        # Add polygons to map
        for feature in features:
            try:
                geom = shape(feature["geometry"])
            except Exception as e:
                print(
                    f"Warning: could not parse geometry for feature {feature.get('id')}: {e}")
                continue

            coords = extract_coords(geom)
            if not coords:
                continue

            if geom.geom_type == "Polygon":
                polygon = st_Polygon(coords,
                                  "#FF000080",
                                  "#FF0000",
                                  )
                m.add_polygon(polygon)

            elif geom.geom_type == "MultiPolygon":
                for poly_coords in coords:
                    polygon = st_Polygon(poly_coords,
                                      "#FF000080",
                                      "#FF0000"
                                      )
                    m.add_polygon(polygon)

        # Render map image
        try:
            image = m.render()
        except Exception as e:
            raise RuntimeError(f"Error rendering static map: {e}")

        # Save image to file if requested
        if output_dir:
            try:
                os.makedirs(output_dir, exist_ok=True)
                file_path = os.path.join(output_dir, filename)
                image.save(file_path)
                print(f"✅ Saved map image to {file_path}")
            except Exception as e:
                print(
                    f"⚠️ Warning: Could not save plot to '{output_dir}'. Error: {e}")

        # Display the image interactively
        if show:
            plt.imshow(image)
            plt.axis("off")
            plt.title(title)
            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    comp_regions = OSPARRegions()
    print(comp_regions.get_wkt("NAAC2", simplify=False))
    print('-----')
    print(comp_regions.get_wkt("NAAC2", simplify=True))
    comp_regions.plot_map("NAAC2")
    id_list = comp_regions.get_all_ids()
    for item in id_list:
        print(item)



