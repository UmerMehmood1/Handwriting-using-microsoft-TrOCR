from typing import Dict
from .ShapeAttributes import ShapeAttributes
from .RegionAttributes import RegionAttributes

class Region:
    def __init__(self, shape_attributes: Dict, region_attributes: Dict):
        self.shape_attributes = ShapeAttributes(
            shape_attributes.get("x"),
            shape_attributes.get("y"),
            shape_attributes.get("width"),
            shape_attributes.get("height"),
        )
        self.region_attributes = RegionAttributes(
            region_attributes.get("Language", "English"),
            region_attributes.get("Dosage", ""),
            region_attributes.get("Dignostic", ""),
            region_attributes.get("Symptoms", ""),
            region_attributes.get("Medicine Name", ""),
            region_attributes.get("Text", ""),
            region_attributes.get("Personal Information", "N/A"),
            region_attributes.get("Numeric Data", "N/A"),
        )

    def to_dict(self):
        return {
            "shape_attributes": self.shape_attributes.to_dict(),
            "region_attributes": self.region_attributes.to_dict(),
        }

    def __repr__(self):
        return f"Region({self.shape_attributes}, {self.region_attributes})"
