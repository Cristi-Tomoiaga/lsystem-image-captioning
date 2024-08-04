import argparse
import pickle
import re


class LWordTokenizer:
    def __init__(self):
        self.__tokens = {'F', '+F', '-F', '[', ']'}

    def __call__(self, lword):
        regex = re.compile(r'F|\+F|-F|\[|]')

        return regex.findall(lword)


class Vocabulary:
    def __init__(self):
        self.__token2idx = {}
        self.__idx2token = {}
        self.__index = 0
        self.__tokenizer = LWordTokenizer()

        self.__build_vocabulary()

    @property
    def tokenizer(self):
        return self.__tokenizer

    def __add_token(self, token):
        if token not in self.__token2idx:
            self.__token2idx[token] = self.__index
            self.__idx2token[self.__index] = token
            self.__index += 1

    def __build_vocabulary(self):
        self.__add_token("<pad>")
        self.__add_token("<bos>")
        self.__add_token("<eos>")
        self.__add_token("F")
        self.__add_token("+F")
        self.__add_token("-F")
        self.__add_token("[")
        self.__add_token("]")

    def __call__(self, token):
        return self.__token2idx[token]

    def __len__(self):
        return len(self.__token2idx)

    def convert_from_lword(self, lword):
        tokens = self.__tokenizer(lword)

        return [self("<bos>")] + [self(token) for token in tokens] + [self("<eos>")]

    def convert_to_lword(self, tokens_idx):
        tokens = []

        for token_idx in tokens_idx:
            token = self.__idx2token[token_idx]
            tokens.append(token)

            if token == "<eos>":
                break

        return "".join(tokens)


def main(args):
    vocab = Vocabulary()

    with open(args.vocab_path, 'wb') as f:
        pickle.dump(vocab, f)

    print(f'Saved the vocabulary to {args.vocab_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_path', default='../generated_datasets/vocab.pkl', type=str, help='The path where the vocabulary is saved')

    parsed_args = parser.parse_args()
    main(parsed_args)
