from lsystem import LSystem
from lword_renderer import LWordRenderer


width = 512
height = 512
distance = 5
angle = 25
num_iterations = 7

# lsystem = LSystem(
#     "X",
#     [
#         "X->F+[[X]-X]-F[-FX]+X",
#         "F->FF"
#     ]
# )
# lsystem = LSystem(
#     "F",
#     [
#         "F->F[+F]F[-F]F"
#     ],
#     False
# )
# lsystem = LSystem(
#     "F",
#     [
#         "F:0.33->F[+F]F[-F]F",
#         "F:0.33->F[+F]F",
#         "F:0.33->F[-F]F"
#     ],
#     True
# )
lsystem = LSystem(
    "X",
    [
        "X:0.40->F[-X][+X]",
        "X:0.30->F-X",
        "X:0.30->F+X",
    ],
    True
)
lword = lsystem.generate(num_iterations, clean_lword=True)
print(lword)
print()

renderer = LWordRenderer(width, height)
print(renderer.validate_word(lword, angle, distance))

image = renderer.render(lword, angle, distance, rescale=True)
image.save("test7.png")
