def reverse_geocode(lat: float, lon: float) -> dict:
    try:
        return Geocode(site_config.MAP_API_PROVIDER).reverse(lat, lon)
    except Exception as e:
        util.logger.warning(f"Error while reverse geocoding: {e}")
        return {}

