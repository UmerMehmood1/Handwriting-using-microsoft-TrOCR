from typing import List, Dict, Optional
from .ImageMetadata import ImageMetadata

class AnnotatedData:
    def __init__(self, data: Dict):
        self.image_ids = data.get("_via_image_id_list", [])
        self.metadata = {
            img_id: ImageMetadata(
                data["_via_img_metadata"][img_id]["filename"],
                data["_via_img_metadata"][img_id]["size"],
                data["_via_img_metadata"][img_id].get("regions", [])
            ) for img_id in self.image_ids if img_id in data["_via_img_metadata"]
        }

    def __repr__(self):
        return f"AnnotatedData(Images={list(self.metadata.keys())})"