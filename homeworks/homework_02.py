import torch
import torch.nn as nn


class NET(nn.Module):
    def __init__(self, input_size, hidden_size=15):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.context = 7
        self.epoches = 10
        self.input = nn.Embedding(input_size, hidden_size)
        self.input_hidden = nn.Linear(hidden_size, hidden_size)
        self.hidden = nn.Tanh()
        self.hidden_hidden = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, input_size)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01)

    def fit(self, X, words):
        self.train()
        for epoche in range(self.epoches):
            for i in range(len(X)-self.context):
                context = X[i:i+self.context]
                target = X[i+self.context]
                tmp = torch.zeros((1, self.hidden_size))
                for word in context:
                    embedding = self.input(torch.tensor([words.index(word)]))
                    tmp = self.hidden(
                        self.input_hidden(embedding) + self.hidden_hidden(tmp)
                    )
                output = self.output(tmp)
                loss = self.criterion(output, torch.tensor([words.index(target)]))
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def predict(self, context, words):
        self.eval()
        with torch.no_grad():
            tmp = torch.zeros((1, self.hidden_size))
            for word in context:
                embedding = self.input(torch.tensor([words.index(word)]))
                tmp = self.hidden(
                    self.input_hidden(embedding) + self.hidden_hidden(tmp)
                )
            output = self.output(tmp)
            result = nn.functional.softmax(output)
            return words[torch.argmax(result)]


def main():
    words = []
    with open('saltan.txt') as file:
        for line in file:
            for symb in '.,;:-!?"':
                line = line.replace(symb, '')
                line = line.replace('\n', ' ')
            words.extend(line.split(' '))
            words = [word for word in words if word != '']
    model = NET(len(words), 64)
    model.fit(words[:1000], words)
    print(model.predict(words[1000:1007], words))
    print(words[1007])


if __name__ == "__main__":
    main()