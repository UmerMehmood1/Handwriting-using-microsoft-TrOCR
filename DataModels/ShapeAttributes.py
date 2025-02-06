class ShapeAttributes:
    def __init__(self, x: int, y: int, width: int, height: int):
        self.x = x
        self.y = y
        self.width = width
        self.height = height

    def to_dict(self):
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
        }

    def __repr__(self):
        return f"ShapeAttributes(x={self.x}, y={self.y}, width={self.width}, height={self.height})"
