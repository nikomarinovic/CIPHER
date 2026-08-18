import json
import os

import requests


class GeoRouter:
    """
    Geocodes place names and fetches driving-route alternatives between
    them, using two free public services:

      - Nominatim (OpenStreetMap) for geocoding place names -> coordinates
        https://nominatim.org
      - OSRM's public demo server for turn-by-turn routing
        https://project-osrm.org

    Both are free, rate-limited public services meant for light,
    non-commercial use — fine for a personal assistant, not for
    high-volume production traffic. If this ever needs to scale, swap
    the URLs for a self-hosted Nominatim/OSRM instance or a paid
    provider (Mapbox, Google, etc.) — the interface below stays the same.
    """

    NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
    OSRM_URL = "https://router.project-osrm.org/route/v1/driving/{coords}"

    HEADERS = {
        # Nominatim's usage policy requires a real, identifying User-Agent.
        "User-Agent": "CIPHER-personal-assistant/1.0 (personal project; no contact configured)"
    }

    # Nominatim result "class"/"type" pairs that represent an actual
    # settlement (city/town/village/etc). When a plain place name like
    # "Moskva" is searched, Nominatim's #1 result is often some
    # institution located there (a university, an embassy...) rather
    # than the city itself — we prefer an actual settlement match when
    # one exists among the top results.
    SETTLEMENT_TYPES = {
        "city", "town", "village", "hamlet", "municipality",
        "administrative", "state", "county", "suburb",
    }

    def geocode(self, place: str) -> dict | None:
        place = (place or "").strip()

        if not place:
            return None

        try:
            response = requests.get(
                self.NOMINATIM_URL,
                params={"q": place, "format": "jsonv2", "limit": 5},
                headers=self.HEADERS,
                timeout=10,
            )

            response.raise_for_status()
            results = response.json()

        except (requests.RequestException, ValueError):
            return None

        if not results:
            return None

        result = self._best_match(results)

        try:
            return {
                "query": place,
                "name": result.get("display_name", place),
                "lat": float(result["lat"]),
                "lon": float(result["lon"]),
            }
        except (KeyError, TypeError, ValueError):
            return None

    def _best_match(self, results: list[dict]) -> dict:
        for result in results:
            if result.get("class") == "place" and result.get("type") in self.SETTLEMENT_TYPES:
                return result

        for result in results:
            if result.get("type") in self.SETTLEMENT_TYPES:
                return result

        return results[0]

    def get_routes(self, origin: dict, destination: dict, limit: int = 3) -> list[dict]:
        coords = f"{origin['lon']},{origin['lat']};{destination['lon']},{destination['lat']}"

        try:
            response = requests.get(
                self.OSRM_URL.format(coords=coords),
                params={
                    "alternatives": "true",
                    "overview": "full",
                    "geometries": "geojson",
                    "steps": "false",
                },
                timeout=15,
            )

            response.raise_for_status()
            data = response.json()

        except (requests.RequestException, ValueError):
            return []

        if data.get("code") != "Ok":
            return []

        routes = []

        for route in data.get("routes", []):
            distance_km = route.get("distance", 0) / 1000
            duration_min = route.get("duration", 0) / 60

            # geojson coordinates come as [lon, lat] pairs — flip to
            # (lat, lon) so the rest of the code stays consistent with
            # how origin/destination points are stored.
            raw_coords = route.get("geometry", {}).get("coordinates", [])
            geometry = [(point[1], point[0]) for point in raw_coords]

            routes.append({
                "distance_km": round(distance_km, 1),
                "duration_min": round(duration_min),
                "geometry": geometry,
            })

        routes.sort(key=lambda item: item["duration_min"])

        return routes[:limit]


def _short_name(full_name: str) -> str:
    """Nominatim display names are long ('Zagreb, Grad Zagreb, Croatia').
    Take just the first, human-relevant part."""
    return full_name.split(",")[0].strip()


def _format_duration(duration_min: float) -> str:
    total_minutes = int(round(duration_min))
    hours, minutes = divmod(total_minutes, 60)

    if hours:
        return f"{hours}h {minutes}min"

    return f"{minutes} min"


def _bresenham(x0: int, y0: int, x1: int, y1: int):
    """Classic Bresenham line algorithm — yields every pixel on the
    straight line between two points, used to draw the route path across
    the canvas one pixel at a time."""
    points = []

    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    error = dx + dy

    x, y = x0, y0

    while True:
        points.append((x, y))

        if x == x1 and y == y1:
            break

        doubled_error = 2 * error

        if doubled_error >= dy:
            error += dy
            x += sx

        if doubled_error <= dx:
            error += dx
            y += sy

    return points


_BORDERS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "world_borders.json",
)

_borders_cache = None


def _load_borders() -> list[dict]:
    """Lazily load + cache the bundled low-resolution world country
    outlines (simplified from Natural Earth's 110m dataset, ~180KB).
    Shipped as a local file rather than fetched at runtime — borders
    don't change day-to-day, and this keeps the map working even if
    OpenStreetMap/OSRM are slow or unavailable."""
    global _borders_cache

    if _borders_cache is not None:
        return _borders_cache

    try:
        with open(_BORDERS_PATH, "r", encoding="utf-8") as file:
            _borders_cache = json.load(file)

    except (OSError, json.JSONDecodeError):
        _borders_cache = []

    return _borders_cache


def _rings_in_view(min_lat, max_lat, min_lon, max_lon) -> list[list]:
    """Pre-filter to just the polygon rings anywhere near the viewport,
    so the scanline fill below doesn't waste time on other continents."""
    countries = _load_borders()

    lat_margin = (max_lat - min_lat) * 0.6
    lon_margin = (max_lon - min_lon) * 0.6

    view_min_lat, view_max_lat = min_lat - lat_margin, max_lat + lat_margin
    view_min_lon, view_max_lon = min_lon - lon_margin, max_lon + lon_margin

    def in_view(lat, lon):
        return view_min_lat <= lat <= view_max_lat and view_min_lon <= lon <= view_max_lon

    rings = []

    for country in countries:
        for ring in country["rings"]:
            if any(in_view(lat, lon) for lon, lat in ring):
                rings.append(ring)

    return rings


def _land_mask(rings, px_cols, px_rows, min_lat, max_lat, min_lon, max_lon) -> list[list[bool]]:
    """Rasterize country polygons into a px_cols x px_rows boolean grid
    (True = land) with a classic even-odd scanline fill — one horizontal
    scan per pixel row, filling between each pair of edge crossings.
    This is what turns 'a squiggly outline' into an actual land/sea map."""
    lat_range = max(max_lat - min_lat, 1e-6)
    lon_range = max(max_lon - min_lon, 1e-6)

    mask = [[False] * px_cols for _ in range(px_rows)]

    for row in range(px_rows):
        lat = max_lat - (row + 0.5) / px_rows * lat_range

        crossings = []

        for ring in rings:
            count = len(ring)

            for i in range(count):
                lon1, lat1 = ring[i]
                lon2, lat2 = ring[(i + 1) % count]

                if abs(lon2 - lon1) > 180 or lat1 == lat2:
                    continue

                if (lat1 <= lat < lat2) or (lat2 <= lat < lat1):
                    t = (lat - lat1) / (lat2 - lat1)
                    crossings.append(lon1 + t * (lon2 - lon1))

        if not crossings:
            continue

        crossings.sort()

        for i in range(0, len(crossings) - 1, 2):
            x_start = int((crossings[i] - min_lon) / lon_range * (px_cols - 1))
            x_end = int((crossings[i + 1] - min_lon) / lon_range * (px_cols - 1))

            x_start = max(0, min(px_cols - 1, x_start))
            x_end = max(0, min(px_cols - 1, x_end))

            for x in range(x_start, x_end + 1):
                mask[row][x] = True

    return mask


_RESET = "\033[0m"

# Foreground/background ANSI codes per pixel "type" — sea=blue, land=
# green (classic atlas palette), route=red so it pops against both.
_FG = {"sea": "34", "land": "32", "route": "91"}
_BG = {"sea": "44", "land": "42", "route": "41"}
_MARKER_STYLE = "\033[1;97;41m"  # bold white text on red — a map "pin"


def draw_route_map(origin: dict, destination: dict, route: dict, cols: int = 64, rows: int = 26) -> str:
    """Render an actual colored land/sea map — not an outline sketch —
    with the real road path drawn on top. Uses the Unicode upper-half-
    block trick (▀ with independent foreground/background colors) to
    get double the vertical resolution out of each terminal character,
    the same technique terminal image viewers use. Still a schematic
    equirectangular projection (not to scale, no elevation), but this
    is about as close to 'a real map' as plain colored text gets."""

    geometry = route.get("geometry") or []

    points = geometry if geometry else [
        (origin["lat"], origin["lon"]),
        (destination["lat"], destination["lon"]),
    ]

    lats = [point[0] for point in points]
    lons = [point[1] for point in points]

    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)

    lat_pad = max((max_lat - min_lat) * 0.18, 0.06)
    lon_pad = max((max_lon - min_lon) * 0.18, 0.06)

    min_lat -= lat_pad
    max_lat += lat_pad
    min_lon -= lon_pad
    max_lon += lon_pad

    lat_range = max(max_lat - min_lat, 1e-6)
    lon_range = max(max_lon - min_lon, 1e-6)

    px_cols = cols
    px_rows = rows * 2  # half-block doubles vertical resolution

    def project_px(lat: float, lon: float) -> tuple[int, int]:
        x = int((lon - min_lon) / lon_range * (px_cols - 1))
        y = int((max_lat - lat) / lat_range * (px_rows - 1))
        x = min(max(x, 0), px_cols - 1)
        y = min(max(y, 0), px_rows - 1)
        return x, y

    rings = _rings_in_view(min_lat, max_lat, min_lon, max_lon)
    land = _land_mask(rings, px_cols, px_rows, min_lat, max_lat, min_lon, max_lon)

    pixel_type = [
        ["land" if land[y][x] else "sea" for x in range(px_cols)]
        for y in range(px_rows)
    ]

    projected_points = [project_px(lat, lon) for lat, lon in points]

    for (x0, y0), (x1, y1) in zip(projected_points, projected_points[1:]):
        for x, y in _bresenham(x0, y0, x1, y1):
            pixel_type[y][x] = "route"

            # A single-pixel-wide line tends to visually disappear
            # against the fill colors — nudge it slightly thicker.
            if x + 1 < px_cols:
                pixel_type[y][x + 1] = "route"

    origin_px = project_px(origin["lat"], origin["lon"])
    destination_px = project_px(destination["lat"], destination["lon"])

    marker_cells = {
        (origin_px[0], origin_px[1] // 2): "A",
        (destination_px[0], destination_px[1] // 2): "B",
    }

    origin_label = _short_name(origin["name"])
    destination_label = _short_name(destination["name"])
    duration_text = _format_duration(route["duration_min"])
    distance_text = f"{route['distance_km']:.1f} km"

    lines = []
    lines.append("┌" + "─" * cols + "┐")

    for row in range(rows):
        top_row = pixel_type[row * 2]
        bottom_index = row * 2 + 1
        bottom_row = pixel_type[bottom_index] if bottom_index < px_rows else top_row

        rendered = []

        for col in range(cols):
            marker = marker_cells.get((col, row))

            if marker:
                rendered.append(f"{_MARKER_STYLE}{marker}{_RESET}")
                continue

            top_type = top_row[col]
            bottom_type = bottom_row[col]
            rendered.append(f"\033[{_FG[top_type]};{_BG[bottom_type]}m\u2580{_RESET}")

        lines.append("│" + "".join(rendered) + "│")

    lines.append("├" + "─" * cols + "┤")

    legend = f" A = {origin_label}    B = {destination_label} "
    stats = f" {distance_text}  \u00b7  {duration_text} "

    lines.append("│" + legend[:cols].ljust(cols) + "│")
    lines.append("│" + stats.ljust(cols) + "│")
    lines.append("└" + "─" * cols + "┘")
    lines.append("(shematski prikaz, nije razmjeren — kopno/more iz Natural Earth, ruta preko OSRM-a)")
    lines.append("Izvor: OpenStreetMap (Nominatim) + OSRM + Natural Earth")

    return "\n".join(lines)


def format_routes(origin: dict, destination: dict, routes: list[dict], width: int = 66) -> str:
    """Render route alternatives as a clean box for terminal display.
    Kept as a fallback/summary view — draw_route_map() is the primary
    display now, but this is handy when comparing several alternatives
    at once (the map only plots the fastest one)."""

    origin_label = _short_name(origin["name"])
    destination_label = _short_name(destination["name"])

    lines = []
    lines.append("┌" + "─" * width + "┐")

    title = f" {origin_label}  →  {destination_label} "
    lines.append("│" + title.center(width) + "│")
    lines.append("├" + "─" * width + "┤")

    for index, route in enumerate(routes, start=1):
        label = "Najbrza ruta" if index == 1 else f"Alternativa {index - 1}"
        duration_text = _format_duration(route["duration_min"])
        distance_text = f"{route['distance_km']:.1f} km"

        row = f"  {index}. {label:<16}{distance_text:>10}{duration_text:>12}  "

        if len(row) > width:
            row = row[:width]

        lines.append("│" + row.ljust(width) + "│")

    lines.append("└" + "─" * width + "┘")
    lines.append("Izvor: OpenStreetMap (Nominatim) + OSRM — © OpenStreetMap contributors")

    return "\n".join(lines)