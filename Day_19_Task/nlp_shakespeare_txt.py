import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import os

file_path = "shakespeare_sonnets.txt"

if not os.path.exists(file_path):
    raise FileNotFoundError(f"{file_path} not found. Please download it manually and place it in the same directory as this script.")

with open(file_path, "r", encoding="utf-8") as file:
    text = file.read()

start_index = text.find("THE SONNETS")
end_index = text.find("End of the Project Gutenberg EBook of Shakespeare's Sonnets")
text = text[start_index:end_index].strip()

chars = sorted(list(set(text)))
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}
encoded_text = np.array([char_to_idx[ch] for ch in text])

class SonnetDataset(Dataset):
    def __init__(self, text, seq_length):
        self.text = text
        self.seq_length = seq_length

    def __len__(self):
        return len(self.text) - self.seq_length

    def __getitem__(self, idx):
        return (torch.tensor(self.text[idx:idx+self.seq_length]),
                torch.tensor(self.text[idx+1:idx+self.seq_length+1]))

seq_length = 100
batch_size = 64
dataset = SonnetDataset(encoded_text, seq_length)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

class VanillaRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super(VanillaRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, hidden):
        x = self.embedding(x)
        x = self.dropout(x)
        out, hidden = self.rnn(x, hidden)
        out = self.dropout(out)
        out = self.fc(out.reshape(out.size(0)*out.size(1), out.size(2)))
        return out, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(num_layers, batch_size, hidden_dim).to(device)

class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, hidden):
        x = self.embedding(x)
        x = self.dropout(x)
        out, hidden = self.lstm(x, hidden)
        out = self.dropout(out)
        out = self.fc(out.reshape(out.size(0)*out.size(1), out.size(2)))
        return out, hidden

    def init_hidden(self, batch_size):
        return (torch.zeros(num_layers, batch_size, hidden_dim).to(device),
                torch.zeros(num_layers, batch_size, hidden_dim).to(device))

class GRU(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super(GRU, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, hidden):
        x = self.embedding(x)
        x = self.dropout(x)
        out, hidden = self.gru(x, hidden)
        out = self.dropout(out)
        out = self.fc(out.reshape(out.size(0)*out.size(1), out.size(2)))
        return out, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(num_layers, batch_size, hidden_dim).to(device)

def train_model(model, dataloader, criterion, optimizer, num_epochs, rnn_type='vanilla'):
    training_losses = []
    for epoch in range(num_epochs):
        for inputs, targets in dataloader:
            batch_size = inputs.size(0)
            hidden = model.init_hidden(batch_size)

            if rnn_type == 'lstm' or rnn_type == 'gru':
                hidden = tuple([h.detach() for h in hidden])  # Detach hidden state for LSTM and GRU
            else:
                hidden = hidden.detach()  # Detach hidden state for Vanilla RNN

            inputs, targets = inputs.to(device), targets.to(device).long()
            optimizer.zero_grad()
            outputs, hidden = model(inputs, hidden)
            loss = criterion(outputs, targets.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            optimizer.step()
        training_losses.append(loss.item())
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')
    return training_losses

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab_size = len(chars)
embedding_dim = 128
hidden_dim = 256
num_layers = 2
num_epochs = 20

# Initialize the model, criterion, and optimizer for Vanilla RNN
model = VanillaRNN(vocab_size, embedding_dim, hidden_dim, num_layers).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
training_losses = train_model(model, dataloader, criterion, optimizer, num_epochs, rnn_type='vanilla')


def generate_text(model, start_str, length):
    model.eval()
    # Filter the start_str to include only characters present in char_to_idx
    start_str = ''.join([ch for ch in start_str if ch in char_to_idx])
    if len(start_str) == 0:
        raise ValueError("The start string contains no valid characters.")

    input_seq = torch.tensor([char_to_idx[ch] for ch in start_str]).unsqueeze(0).to(device)
    batch_size = input_seq.size(0)
    hidden = model.init_hidden(batch_size)
    generated_text = start_str

    for _ in range(length):
        output, hidden = model(input_seq, hidden)
        output = output[-1, :].detach().cpu().numpy()
        char_idx = np.argmax(output)
        generated_text += idx_to_char[char_idx]
        input_seq = torch.tensor([[char_idx]]).to(device)

    return generated_text


start_str = "Shall I compare thee to a summer's day?\n"
generated_text = generate_text(model, start_str, 200)
print(generated_text)

def plot_training_loss(training_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(training_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

plot_training_loss(training_losses)
