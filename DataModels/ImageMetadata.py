from typing import List, Dict
from .Region import Region

class ImageMetadata:
    def __init__(self, filename: str, size: int, regions: List[Dict]):
        self.filename = filename
        self.size = size
        self.regions = [Region(region.get("shape_attributes", {}), region.get("region_attributes", {})) for region in regions]

    def to_dict(self):
        return {
            "filename": self.filename,
            "size": self.size,
            "regions": [region.to_dict() for region in self.regions]  # Convert Region objects to dictionaries
        }

    def __repr__(self):
        return f"ImageMetadata(Filename={self.filename}, Size={self.size}, Regions={self.regions})"
