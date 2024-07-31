from dataset_generator.lsystem import LSystem
from dataset_generator.lword_renderer import LWordRenderer
from dataset_generator.lword_preprocessor import LwordPreprocessor

width = 512
height = 512
distance = 5
angle = 25
num_iterations = 5

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
# lsystem = LSystem(
#     "X",
#     [
#         "X:0.40->F[-X][+X]",
#         "X:0.30->F-X",
#         "X:0.30->F+X",
#     ],
#     True
# )
# lsystem = LSystem(
#     "FX",
#     [
#         "X:0.40->[-FX][+FX]",
#         "X:0.30->-FX",
#         "X:0.30->+FX",
#     ],
#     True
# )
# also good in main:
# lsystem = LSystem(
#     "FX",
#     [
#         "X:0.55->[-FX][+FX]",
#         "X:0.225->-FX",
#         "X:0.225->+FX",
#     ],
#     True
# )
lsystem = LSystem(
    "FX",
    [
        "X:0.34->[-FX][+FX]",
        "X:0.33->-FX",
        "X:0.33->+FX",
    ],
    True
)
lword = lsystem.generate(num_iterations, clean_lword=True)
print(lword)
print()

renderer = LWordRenderer(width, height)
image = renderer.render(lword, angle, distance, rescale=True)
image.save("test8.png")

# lword = "F-F[-F+F+F[-F-F+][+F-F-]][+F-F-F+F[-F+][+F+]]"
# lword = "F[-F[-F+F+F-F-F+][+F[-F[-F+F-F-][+F+F-F-]][+F[-F+F+F+][+F-F-F]]]][+F[-F-F+F+F-F][+F-F[-F[-F-F+][+F[-F+][+F-]]][+F[-F[-F-][+F]][+F+F-]]]]"
# angle = 32
# angle = 15
# distance = 100

# doubled_lines = renderer.validate_lword_geometrically(lword, angle, distance)
# print(not doubled_lines)
# image = renderer.render(lword, angle, distance, rescale=True)
# image.save("test_overlap_initial.png")
#
# lword_modified = renderer.fix_lword_geometrically(lword, angle, distance)
# print(lword, lword_modified)
# doubled_lines = renderer.validate_lword_geometrically(lword_modified, angle, distance)
# print(not doubled_lines)
# image = renderer.render(lword_modified, angle, distance, rescale=True)
# image.save("test_overlap_final.png")

# print(LwordPreprocessor.check_canceling_rotations(""))
# print(LwordPreprocessor.check_canceling_rotations("F-[+][-+F]-+++-++-"))
# print(LwordPreprocessor.fix_canceling_rotations("F-[+][-+F]-+++-++-"))
# print(LwordPreprocessor.check_canceling_rotations("F-[+][--F]+F-F"))
# print()
# print(LwordPreprocessor.check_empty_branches(""))
# print(LwordPreprocessor.check_empty_branches("F"))
# print(LwordPreprocessor.check_empty_branches("F+[F]"))
# print(LwordPreprocessor.check_empty_branches("F+[F][-F+[F]]F-F[-[+][F]]"))
# print(LwordPreprocessor.fix_empty_branches("F+[F][-F+[F]]F-F[-[+][F]]"))
# print(LwordPreprocessor.check_empty_branches("F+[F[]]"))
# print(LwordPreprocessor.fix_empty_branches("F+[F[]]"))
# print(LwordPreprocessor.check_empty_branches("F+[F[-[F]]]"))
# print(LwordPreprocessor.check_empty_branches("F+[F[-[F][]]]"))
# print(LwordPreprocessor.fix_empty_branches("F+[F[-[F][]]]"))
# print(LwordPreprocessor.check_empty_branches("F+[F[-[+]-]]"))
# print(LwordPreprocessor.fix_empty_branches("F+[F[-[+]-]]"))
# print()
# print(LwordPreprocessor.check_ordered_branches("F[+F][-F]"))
# print(LwordPreprocessor.fix_ordered_branches("F[+F][-F]"))
# print(LwordPreprocessor.check_ordered_branches("F[-F][+F]"))
# print(LwordPreprocessor.check_ordered_branches("F[+F][-F]F-F[+F][-F-F[+F][-F]]F[-F]"))
# print(LwordPreprocessor.fix_ordered_branches("F[+F][-F]F-F[+F][-F-F[+F][-F]]F[-F]"))
# print(LwordPreprocessor.check_ordered_branches("F[-F][+F]F-F[-F][+F-F[+F][-F]]F[-F]"))
# print(LwordPreprocessor.fix_ordered_branches("F[-F][+F]F-F[-F][+F-F[+F][-F]]F[-F]"))
# print(LwordPreprocessor.check_ordered_branches("F[-F][+F]F-F[-F][+F-F[-F][+F]]F[-F]"))
# print()
# print(LwordPreprocessor.check_ending_subbranches("F"))
# print(LwordPreprocessor.check_ending_subbranches("F[-F]"))
# print(LwordPreprocessor.fix_ending_subbranches("F[-F]"))
# print(LwordPreprocessor.check_ending_subbranches("F[-F][+F]"))
# print(LwordPreprocessor.check_ending_subbranches("F[-F][+F]F[-F]F-F+F[-F[+F]][+F]"))
# print(LwordPreprocessor.fix_ending_subbranches("F[-F][+F]F[-F]F-F+F[-F[+F]][+F]"))
# print(LwordPreprocessor.check_ending_subbranches("F[-F][+F]F[-F]F-F+F[-F[+F][-F]][+F]"))
# print(LwordPreprocessor.check_ending_subbranches("F[-F][+F]F[-F]F-F+F[-F[+F][-F]][+F]F"))
# print(LwordPreprocessor.check_ending_subbranches("F[-F][+F]F[-F]F-F+F[-F[+F][-F]][+F]F[-F]"))
# print(LwordPreprocessor.fix_ending_subbranches("F[-F][+F]F[-F]F-F+F[-F[+F][-F]][+F]F[-F]"))
# print(LwordPreprocessor.check_ending_subbranches("F[-F][+F]F[-F]F-F+F[-F[+F-F]][+F]F[-F]"))
# print(LwordPreprocessor.fix_ending_subbranches("F[-F][+F]F[-F]F-F+F[-F[+F-F]][+F]F[-F]"))
