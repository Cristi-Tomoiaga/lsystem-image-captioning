from lsystem import LSystem
from lword_renderer import LWordRenderer


width = 512
height = 512
distance = 5
angle = 25
num_iterations = 5

lsystem = LSystem(
    "X",
    [
        "X->F+[[X]-X]-F[-FX]+X",
        "F->FF"
    ]
)
# lsystem = LSystem(
#     "F",
#     [
#         "F->F[+F]F[-F]F"
#     ]
# )
lword = lsystem.generate(num_iterations)
print(lword)
print()

renderer = LWordRenderer(width, height)
print(renderer.validate_word(lword, angle, distance))

image = renderer.render(lword, angle, distance, rescale=True)
image.save("test4.png")
