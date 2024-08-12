import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence


class EncoderCNN(nn.Module):
    def __init__(self, feature_size):
        super(EncoderCNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        # self.conv3 = nn.Conv2d(32, 64, kernel_size=5, stride=2)
        # self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, dilation=2)  # investigate, also batch norm, dropout
        # self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, dilation=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, dilation=2)

        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.max_pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()
        self.linear = nn.Linear(64 * 7 * 7, feature_size)

    def forward(self, images):
        c1 = self.max_pool1(F.relu(self.conv1(images)))  # (batch_size, 16, 127, 127)
        c2 = self.max_pool2(F.relu(self.conv2(c1)))  # (batch_size, 32, 31, 31)
        c3 = self.max_pool3(F.relu(self.conv3(c2)))  # (batch_size, 64, 7, 7)
        features = self.linear(self.flatten(c3))  # (batch_size, feature_size)

        return features


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size):
        super(DecoderRNN, self).__init__()

        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions, lengths):
        embeddings = self.embed(captions)  # (batch_size, max_target_length, embed_size)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), dim=1)  # (batch_size, max_target_length+1, embed_size)
        # cuts off the <end> token because the features are the first token now
        packed_embeddings = pack_padded_sequence(embeddings, lengths, batch_first=True)  # (*, embed_size)

        hiddens, _ = self.lstm(packed_embeddings)  # (*, hidden_size)
        outputs = self.linear(hiddens[0])  # (*, vocab_size)

        return outputs

    def generate_caption(self, features, max_sequence_length, return_idx=True):
        generated_idx = []
        inputs = features.unsqueeze(1)  # (batch_size, 1, feature_size)
        states = None

        for _ in range(max_sequence_length):
            hiddens, states = self.lstm(inputs, states)  # (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))  # (batch_size, vocab_size)

            predicted_idx = outputs.argmax(dim=1)  # (batch_size)
            generated_idx.append(predicted_idx if return_idx else outputs)

            inputs = self.embed(predicted_idx)  # (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)  # (batch_size, 1, embed_size)

        generated_idx = torch.stack(generated_idx, dim=1)  # (batch_size, max_sequence_length) or (batch_size, max_sequence_length, vocab_size)

        return generated_idx
