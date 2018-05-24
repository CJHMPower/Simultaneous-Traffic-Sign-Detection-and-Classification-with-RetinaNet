class BoundingBox:
    def __init__(self, left, top, right, bottom, image_width, image_height, label):
        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom
        self.width = right - left
        self.height = bottom - top
        self.image_width = image_width
        self.image_height = image_height
        self.label = label

    def __repr__(self):
        return '(x1: {}, y1: {}, x2: {}, y2: {} ({}))'.format(self.left, self.top, self.right, self.bottom, self.label)

    def flip(self):
        left = self.image_width - self.right
        top = self.image_height - self.bottom
        right = self.image_width - self.left
        bottom = self.image_height - self.top
        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom
        return self

    def resize(self, width, height):
        width_ratio = width / float(self.image_width)
        height_ratio = height / float(self.image_height)
        self.left = int(self.left * width_ratio)
        self.top = int(self.top * height_ratio)
        self.right = int(self.right * width_ratio)
        self.bottom = int(self.bottom * height_ratio)
        return self