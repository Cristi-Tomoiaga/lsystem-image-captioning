import re
from typing import Callable


class LWordPreprocessor:
    @staticmethod
    def check_canceling_rotations(lword: str) -> bool:
        regex = re.compile(r'\+-|-\+')

        return len(regex.findall(lword)) == 0

    @staticmethod
    def check_empty_branches(lword: str) -> bool:
        i = 0
        stack = []

        while i < len(lword):
            if lword[i] == '[':
                stack.append("")
            elif lword[i] == ']':
                if stack and 'F' not in stack[-1]:
                    return False

                branch = stack.pop()
                if stack:
                    stack[-1] += branch
            elif stack:
                stack[-1] += lword[i]

            i += 1

        return True

    @staticmethod
    def __verify_condition_for_branches(lword: str, condition: Callable[[list[str], bool], bool]) -> bool:
        queue = [[lword]]

        while queue:
            current_list = queue.pop(0)

            for current in current_list:
                i = 0
                branches = []

                while i < len(current):
                    if current[i] == '[':
                        j = i + 1
                        count = 1

                        while j < len(current) and count > 0:
                            if current[j] == '[':
                                count += 1
                            elif current[j] == ']':
                                count -= 1

                            j += 1

                        branch = current[i + 1:j - 1]
                        branches.append(branch)

                        i = j
                    else:
                        if branches:
                            if not condition(branches, False):
                                return False

                            filtered_branches = filter(lambda x: '[' in x, branches)
                            if filtered_branches:
                                queue.append(filtered_branches)

                            branches = []

                        i += 1

                if branches:
                    if not condition(branches, True):
                        return False

                    filtered_branches = filter(lambda x: '[' in x, branches)
                    if filtered_branches:
                        queue.append(filtered_branches)

        return True

    @staticmethod
    def check_ordered_branches(lword: str) -> bool:
        def ordered_branches_condition(branches: list[str], _):
            if len(branches) == 2:
                return sorted(branches, reverse=True) == branches

            return True

        return LWordPreprocessor.__verify_condition_for_branches(lword, ordered_branches_condition)

    @staticmethod
    def check_ending_subbranches(lword: str) -> bool:
        def ending_subbranches_condition(branches: list[str], last: bool):
            if len(branches) == 1 and last:
                return False

            return True

        return LWordPreprocessor.__verify_condition_for_branches(lword, ending_subbranches_condition)

    @staticmethod
    def fix_canceling_rotations(lword: str) -> str:
        return re.sub(r'\+-|-\+', '', lword)

    @staticmethod
    def __convert_to_nested_list(lword: str) -> list:
        stack = []
        current = []

        for char in lword:
            if char == '[':
                stack.append(current)
                current = []
            elif char == ']':
                sublist = current

                current = stack.pop()
                current.append(sublist)
            else:
                if (current and not isinstance(current[-1], str)) or (not current):
                    current.append("")

                current[-1] += char

        return current

    @staticmethod
    def __convert_to_lword(nested_list: list | str) -> str:
        if isinstance(nested_list, list):
            return "[" + "".join(map(LWordPreprocessor.__convert_to_lword, nested_list)) + "]"

        return nested_list

    @staticmethod
    def fix_empty_branches(lword: str) -> str:
        def helper(sublist: list) -> tuple[list, bool]:
            result = []
            is_empty = True

            for item in sublist:
                if isinstance(item, list):
                    fixed_item, is_item_empty = helper(item)

                    if not is_item_empty:
                        is_empty = False
                        result.append(fixed_item)
                else:
                    if "F" in item:
                        is_empty = False

                    result.append(item)

            return result, is_empty

        converted_lword = LWordPreprocessor.__convert_to_nested_list(lword)
        converted_lword, _ = helper(converted_lword)

        return LWordPreprocessor.__convert_to_lword(converted_lword)[1:-1]

    @staticmethod
    def fix_ordered_branches(lword: str) -> str:
        def helper(sublist: list) -> list:
            result = []
            branches = []

            for item in sublist:
                if isinstance(item, list):
                    branches.append(helper(item))
                else:
                    if branches:
                        branches.sort(key=lambda x: LWordPreprocessor.__convert_to_lword(x), reverse=True)
                        result.extend(branches)

                        branches = []

                    result.append(item)

            if branches:
                branches.sort(key=lambda x: LWordPreprocessor.__convert_to_lword(x), reverse=True)
                result.extend(branches)

            return result

        converted_lword = LWordPreprocessor.__convert_to_nested_list(lword)
        converted_lword = helper(converted_lword)

        return LWordPreprocessor.__convert_to_lword(converted_lword)[1:-1]

    @staticmethod
    def fix_ending_subbranches(lword: str) -> str:
        def helper(sublist: list) -> list:
            result = []

            for i, item in enumerate(sublist):
                if isinstance(item, list):
                    fixed_item = helper(item)

                    if i == len(sublist) - 1 and (len(sublist) == 1 or (i > 0 and not isinstance(sublist[i - 1], list))):
                        result.extend(fixed_item)
                    else:
                        result.append(fixed_item)
                else:
                    result.append(item)

            return result

        converted_lword = LWordPreprocessor.__convert_to_nested_list(lword)
        converted_lword = helper(converted_lword)

        return LWordPreprocessor.__convert_to_lword(converted_lword)[1:-1]

    @staticmethod
    def process_lword(lword: str) -> str:
        if not LWordPreprocessor.check_canceling_rotations(lword):
            lword = LWordPreprocessor.fix_canceling_rotations(lword)

        if not LWordPreprocessor.check_empty_branches(lword):
            lword = LWordPreprocessor.fix_empty_branches(lword)

        if not LWordPreprocessor.check_ordered_branches(lword):
            lword = LWordPreprocessor.fix_ordered_branches(lword)

        if not LWordPreprocessor.check_ending_subbranches(lword):
            lword = LWordPreprocessor.fix_ending_subbranches(lword)

        return lword

    @staticmethod
    def process_lword_repeatedly(lword: str) -> str:
        done = False

        while not done:
            done = True

            if not LWordPreprocessor.check_canceling_rotations(lword):
                done = False
                lword = LWordPreprocessor.fix_canceling_rotations(lword)

            if not LWordPreprocessor.check_empty_branches(lword):
                done = False
                lword = LWordPreprocessor.fix_empty_branches(lword)

            if not LWordPreprocessor.check_ordered_branches(lword):
                done = False
                lword = LWordPreprocessor.fix_ordered_branches(lword)

            if not LWordPreprocessor.check_ending_subbranches(lword):
                done = False
                lword = LWordPreprocessor.fix_ending_subbranches(lword)

        return lword
