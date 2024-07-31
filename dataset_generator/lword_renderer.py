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


def are_points_close(point1, point2, tol=1e-8):
    return abs(point1[0] - point2[0]) < tol and abs(point1[1] - point2[1]) < tol


class LWordRenderer:
    def __init__(self, width: int, height: int):
        self.__width = width
        self.__height = height

    def __compute_bounding_box(self, lword: str, angle: float, distance: float) -> tuple[BoundingBox, dict]:
        angle = math.radians(angle)

        x = self.__width // 2
        y = self.__height
        direction = -math.pi
        state_stack = []

        xmin = math.inf
        xmax = -math.inf
        ymin = math.inf
        ymax = -math.inf

        lines_drawn = {}
        doubled_lines = {}

        xmin = min(xmin, x)
        xmax = max(xmax, x)
        ymin = min(ymin, y)
        ymax = max(ymax, y)

        for i, symbol in enumerate(lword):
            if symbol == "F":
                x_new = x + distance * math.sin(direction)
                y_new = y + distance * math.cos(direction)

                xmin = min(xmin, x_new)
                xmax = max(xmax, x_new)
                ymin = min(ymin, y_new)
                ymax = max(ymax, y_new)

                line = (x, y, x_new, y_new)
                reversed_line = (x_new, y_new, x, y)
                line_exists = any(
                    are_points_close(line[:2], key[:2]) and are_points_close(line[2:], key[2:]) for key in lines_drawn
                )

                if not line_exists:
                    lines_drawn[line] = i
                    lines_drawn[reversed_line] = i
                else:
                    found_index = next(
                        (lines_drawn[key] for key in lines_drawn if are_points_close(line[:2], key[:2]) and are_points_close(line[2:], key[2:])),
                        lines_drawn[next(key for key in lines_drawn if are_points_close(reversed_line[:2], key[:2]) and are_points_close(reversed_line[2:], key[2:]))]
                    )

                    if found_index not in doubled_lines:
                        doubled_lines[found_index] = []

                    doubled_lines[found_index].append(i)

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

    def validate_lword_geometrically(self, lword: str, angle: float, distance: float) -> dict:
        _, doubled_lines = self.__compute_bounding_box(lword, angle, distance)

        return doubled_lines

    def fix_lword_geometrically(self, lword: str, angle: float, distance: float) -> str:
        result = lword
        doubled_lines = self.validate_lword_geometrically(result, angle, distance)

        while doubled_lines:
            new_lword = ""
            deleted_indices = set()

            for i in doubled_lines:
                deleted_indices.update(doubled_lines[i])

            for i, symbol in enumerate(result):
                if i not in deleted_indices:
                    new_lword += symbol

            result = new_lword
            doubled_lines = self.validate_lword_geometrically(result, angle, distance)

        return result

    def render(self, lword: str, angle: float, distance: float, rescale: bool, padding: float = 0.95) -> Image:
        scale = 1
        offset_x, extra_offset_x = 0, 0
        offset_y, extra_offset_y = 0, 0

        if rescale:
            bounding_box, _ = self.__compute_bounding_box(lword, angle, distance)

            scale_x = self.__width / (bounding_box.xmax - bounding_box.xmin)
            scale_y = self.__height / (bounding_box.ymax - bounding_box.ymin)
            scale = min(scale_x, scale_y) * padding

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
