class LSystem:  # TODO: implement stochastic
    def __init__(self, axiom: str, rules: list[str]):
        self.__axiom = axiom
        self.__rules = {}

        for rule in rules:
            [symbol, replacement] = rule.split("->")
            self.__rules[symbol] = replacement

    def generate(self, num_iterations: int) -> str:
        result = self.__axiom

        for _ in range(num_iterations):
            result = "".join(self.__rules.get(symbol, symbol) for symbol in result)

        return result
