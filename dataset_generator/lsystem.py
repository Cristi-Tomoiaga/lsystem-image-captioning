import random


class LSystem:
    def __init__(self, axiom: str, rules: list[str], stochastic: bool):
        self.__axiom = axiom
        self.__rules = {}
        self.__stochastic = stochastic
        self.__graphical_symbols = {"F", "+", "-", "[", "]"}

        for rule in rules:
            if stochastic:
                [lhs, replacement] = rule.split("->")
                [symbol, probability] = lhs.split(":")

                if symbol not in self.__rules:
                    self.__rules[symbol] = []

                self.__rules[symbol].append((replacement, float(probability)))
            else:
                [symbol, replacement] = rule.split("->")

                self.__rules[symbol] = replacement

    def __choose_replacement(self, symbol):
        if symbol not in self.__rules:
            return symbol

        replacements, probabilities = zip(*self.__rules[symbol])
        return random.choices(replacements, weights=probabilities, k=1)[0]

    def __clean_lword(self, lword: str):
        return "".join(symbol for symbol in lword if symbol in self.__graphical_symbols)

    def generate(self, num_iterations: int, clean_lword: bool) -> str:
        result = self.__axiom

        for _ in range(num_iterations):
            if self.__stochastic:
                result = "".join(self.__choose_replacement(symbol) for symbol in result)
            else:
                result = "".join(self.__rules.get(symbol, symbol) for symbol in result)

        return self.__clean_lword(result) if clean_lword else result
