from dataclasses import dataclass

import math
from PIL import Image, ImageDraw


@dataclass
class BoundingBox:
    xmin: float
    xmax: float
    ymin: float
    ymax: float


def apply_rescale(coords: tuple[float, float], scale: float, offset_x: float, offset_y: float,
                  extra_offset_x: float = 0, extra_offset_y: float = 0) -> tuple[float, float]:
    return (coords[0] - offset_x) * scale + extra_offset_x, (coords[1] - offset_y) * scale + extra_offset_y


class LWordRenderer:
    def __init__(self, width: int, height: int):
        self.__width = width
        self.__height = height

    def __compute_bounding_box(self, lword: str, angle: float, distance: float) -> tuple[BoundingBox, bool]:
        angle = math.radians(angle)

        x = self.__width // 2
        y = self.__height
        direction = -math.pi
        state_stack = []

        xmin = math.inf
        xmax = -math.inf
        ymin = math.inf
        ymax = -math.inf

        lines_drawn = set()
        doubled_lines = False

        xmin = min(xmin, x)
        xmax = max(xmax, x)
        ymin = min(ymin, y)
        ymax = max(ymax, y)

        for symbol in lword:
            if symbol == "F":
                x_new = x + distance * math.sin(direction)
                y_new = y + distance * math.cos(direction)

                xmin = min(xmin, x_new)
                xmax = max(xmax, x_new)
                ymin = min(ymin, y_new)
                ymax = max(ymax, y_new)

                line = (x, y, x_new, y_new)
                if line not in lines_drawn:
                    lines_drawn.add((x, y, x_new, y_new))
                    lines_drawn.add((x_new, y_new, x, y))
                else:
                    doubled_lines = True

                x, y = x_new, y_new
            elif symbol == "+":
                direction += angle
            elif symbol == "-":
                direction -= angle
            elif symbol == "[":
                state_stack.append((x, y, direction))
            elif symbol == "]":
                x, y, direction = state_stack.pop()

        return BoundingBox(xmin, xmax, ymin, ymax), doubled_lines

    def validate_word(self, lword: str, angle: float, distance: float) -> bool:
        # TODO: implement other validations
        _, doubled_lines = self.__compute_bounding_box(lword, angle, distance)

        return not doubled_lines

    def render(self, lword: str, angle: float, distance: float, rescale: bool, padding: float = 0.9) -> Image:
        scale = 1
        offset_x, extra_offset_x = 0, 0
        offset_y, extra_offset_y = 0, 0

        if rescale:
            bounding_box, _ = self.__compute_bounding_box(lword, angle, distance)

            scale_x = self.__width / (bounding_box.xmax - bounding_box.xmin)
            scale_y = self.__height / (bounding_box.ymax - bounding_box.ymin)
            scale = min(scale_x, scale_y)

            offset_x = bounding_box.xmin
            offset_y = bounding_box.ymin

            upper_left_corner = apply_rescale((bounding_box.xmin, bounding_box.ymin), scale, offset_x, offset_y)
            lower_right_corner = apply_rescale((bounding_box.xmax, bounding_box.ymax), scale, offset_x, offset_y)
            extra_offset_x = (self.__width - (lower_right_corner[0] - upper_left_corner[0])) / 2
            extra_offset_y = (self.__height - (lower_right_corner[1] - upper_left_corner[1])) / 2

        image = Image.new('L', (self.__width, self.__height), color='white')
        draw = ImageDraw.Draw(image)

        angle = math.radians(angle)

        x = self.__width // 2
        y = self.__height
        direction = -math.pi
        state_stack = []

        draw.rectangle([(bounding_box.xmin, bounding_box.ymin), (bounding_box.xmax, bounding_box.ymax)],
                       outline='black')
        draw.rectangle([apply_rescale((bounding_box.xmin, bounding_box.ymin), scale, offset_x, offset_y,
                                      extra_offset_x, extra_offset_y),
                        apply_rescale((bounding_box.xmax, bounding_box.ymax), scale, offset_x, offset_y,
                                      extra_offset_x, extra_offset_y)], outline='black')

        for symbol in lword:
            if symbol == "F":
                x_new = x + distance * math.sin(direction)
                y_new = y + distance * math.cos(direction)

                draw.line(
                    [
                        apply_rescale((x, y), scale, offset_x, offset_y, extra_offset_x, extra_offset_y),
                        apply_rescale((x_new, y_new), scale, offset_x, offset_y, extra_offset_x, extra_offset_y)
                    ],
                    fill='black'
                )

                x, y = x_new, y_new
            elif symbol == "+":
                direction += angle
            elif symbol == "-":
                direction -= angle
            elif symbol == "[":
                state_stack.append((x, y, direction))
            elif symbol == "]":
                x, y, direction = state_stack.pop()

        return image
