import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import csv
import random
import numpy as np
import matplotlib.pyplot as plt
import re
import pickle

from torch.utils.data import TensorDataset, DataLoader


from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


prompts_test = [
    "great melody and rhythm",
    "vocals are outstanding",
    "could use more variety in instruments",
    "too repetitive",
    "lyrics are powerful",
    "catchy chorus",
    "the beat is too slow",
    "amazing production quality",
    "needs better mixing",
    "very original sound",
    "not catchy enough",
    "excellent harmonies",
    "the intro is too long",
    "impressive guitar solo",
    "bass line is too weak",
    "unique vocal style",
    "song feels unfinished",
    "outstanding arrangement",
    "sounds overproduced",
    "beautiful piano section",
    "lacks emotional depth",
    "energy drops in the middle",
    "memorable hook",
    "melody is forgettable",
    "chorus stands out",
    "bridge is unnecessary",
    "great dynamics",
    "the song is too short",
    "loved the drum patterns",
    "missing instrumental break",
    "great use of effects",
    "not enough vocal clarity",
    "lyrics are cliché",
    "track feels too long",
    "catchy and upbeat",
    "melancholic atmosphere",
    "the drop is unexpected",
    "great build-up",
    "too much auto-tune",
    "production is clean",
    "needs more layers",
    "vocals lack emotion",
    "creative sound design",
    "strong opening",
    "the outro is abrupt",
    "chorus is underwhelming",
    "harmonies are weak",
    "groove is infectious",
    "arrangement feels crowded",
    "unique genre fusion",
    "needs a stronger climax"
]


def normalized_positivity(text):
    analyzer = SentimentIntensityAnalyzer()
    score = analyzer.polarity_scores(text)['compound']  # -1 (négatif) à 1 (positif)
    normalized = (score + 1) / 2  # entre 0 et 1
    return normalized


# Neural Network for Reward Model
class RewardModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(RewardModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

def train_reward_model(model, data_loader, num_epochs=10, learning_rate=0.001):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for inputs, targets in data_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * inputs.size(0)
        avg_loss = epoch_loss / len(data_loader.dataset)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {(1 - avg_loss) * 100:.2f}%')
    return model


def evaluate_reward_model(model, data_loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            outputs = model(inputs)
            loss = F.mse_loss(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
    avg_loss = total_loss / len(data_loader.dataset)
    print(f'Validation Loss: {avg_loss:.4f}')
    print("Accuracy: {:.2f}%".format((1 - avg_loss) * 100))
    return avg_loss


# CSV Handling Functions
def create_csv(file_path, n=20):
    csv_dict = {'User': [], 'Prompt': [], 'Mark': [], 'Reward': []}

    def assign_reward(prompt, mark):
        reward_mapping = {
            "good": min(5, mark * 2),
            "bad": max(1, mark - 1),
            "average": mark,
            "more technical": min(5, mark + 1)
        }
        return int(max(1, min(5, reward_mapping.get(prompt, mark))))

    for i in range(n):
        user = f'User {i}'
        current_prompt = random.choice(prompts_test)
        mark = random.randint(1, 5)
        reward = assign_reward(current_prompt, mark)

        csv_dict['User'].append(user)
        csv_dict['Prompt'].append(current_prompt)
        csv_dict['Mark'].append(mark)
        csv_dict['Reward'].append(reward / 5)

    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['User', 'Prompt', 'Mark', 'Reward'])
        for i in range(n):
            writer.writerow([csv_dict['User'][i], csv_dict['Prompt'][i], csv_dict['Mark'][i], csv_dict['Reward'][i]])


def create_csv_empty(file_path, col=["User", "Prompt", "Mark", "Reward"]):
    with open(file_path, "w", newline='') as my_empty_csv:
        writer = csv.writer(my_empty_csv)
        writer.writerow(col)


def add_line_in_csv(file_path, line_data):
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(line_data)

def read_csv(file_path):
    with open(file_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        headers = next(reader)
        data = [row for row in reader]
    return headers, data

def take_columns(file_path, columns):
    headers, data = read_csv(file_path)
    column_data = {col: [] for col in columns}
    for row in data:
        for col in columns:
            if col in headers:
                index = headers.index(col)
                column_data[col].append(row[index])
    return column_data

# Tokenization and Embedding Functions
def tokenize(text):
    pattern = re.compile(r'[A-Za-z]+[\w^\']*|[\w^\']*[A-Za-z]+[\w^\']*')
    return pattern.findall(text.lower())

def mapping(tokens):
    unique_tokens = sorted(set(tokens))
    word_to_id = {token: i for i, token in enumerate(unique_tokens)}
    id_to_word = {i: token for token, i in word_to_id.items()}
    return word_to_id, id_to_word

def one_hot_encode(index, vocab_size):
    vec = np.zeros(vocab_size)
    vec[index] = 1
    return vec

def generate_training_data(tokens, word_to_id, window=2):
    X, y = [], []
    vocab_size = len(word_to_id)
    for i, center_word in enumerate(tokens):
        center_index = word_to_id[center_word]
        for j in range(max(0, i - window), min(len(tokens), i + window + 1)):
            if i != j:
                context_word = tokens[j]
                context_index = word_to_id[context_word]
                X.append(one_hot_encode(center_index, vocab_size))
                y.append(one_hot_encode(context_index, vocab_size))
    return np.array(X), np.array(y)

# Skip-gram Model Functions
def init_network(vocab_size, embedding_dim):
    return {
        "w1": np.random.randn(vocab_size, embedding_dim),
        "w2": np.random.randn(embedding_dim, vocab_size)
    }

def softmax(X):
    exp = np.exp(X - np.max(X, axis=1, keepdims=True))
    return exp / np.sum(exp, axis=1, keepdims=True)

def forward(model, X, return_cache=True):
    a1 = X @ model["w1"]
    a2 = a1 @ model["w2"]
    z = softmax(a2)
    if return_cache:
        return {"a1": a1, "a2": a2, "z": z}
    return z

def cross_entropy(z, y):
    return -np.sum(np.log(z + 1e-9) * y)

def backward(model, X, y, alpha):
    cache = forward(model, X)
    dz = cache["z"] - y
    dw2 = cache["a1"].T @ dz
    da1 = dz @ model["w2"].T
    dw1 = X.T @ da1
    model["w1"] -= alpha * dw1
    model["w2"] -= alpha * dw2
    return cross_entropy(cache["z"], y)

def get_prompt_embedding(prompt, word_to_id, embeddings):
    prompt_embedding = []
    for word in prompt:
        if word in word_to_id:
            embedding = embeddings[word_to_id[word]]
            prompt_embedding.append(embedding)
    if prompt_embedding:
        return np.mean(prompt_embedding, axis=0)
    else:
        # Si aucun mot connu, retourne la moyenne des embeddings du vocabulaire (embedding “neutre”)
        return np.mean(embeddings, axis=0)



def create_random_dataset(n=20, prompts_test=prompts_test):
    # Generate random data for n users
    users = [f"User {i}" for i in range(1, n + 1)]
    prompts = [random.choice(prompts_test) for _ in range(n)]
    marks = [normalized_positivity(prompt) for prompt in prompts]
    rewards = [random.uniform(0, 1) for _ in range(n)]
    return list(zip(users, prompts, marks, rewards))


def save_model(model, file_path):
    torch.save(model.state_dict(), file_path)



if __name__ == "__main__":
    # Example Usage
    create_csv_empty("data/reward_data.csv")
    for i in range(20):
        add_line_in_csv("data/reward_data.csv", create_random_dataset(1, prompts_test)[0])

    headers, data = read_csv('data/reward_data.csv')
    print("CSV Headers and Data:")
    print(headers)
    print(data)

    datas = take_columns('data/reward_data.csv', ['User', 'Prompt', 'Mark', 'Reward'])
    print("\nSelected Columns:")
    print(datas)

    # Tokenization and Embedding
    tokens_nested = [tokenize(prompt) for prompt in datas['Prompt']]
    tokens = [word for sublist in tokens_nested for word in sublist]
    word_to_id, id_to_word = mapping(tokens)

    X, y = generate_training_data(tokens, word_to_id, window=2)

    # Training the Skip-gram Model
    np.random.seed(42)
    vocab_size = len(word_to_id)
    embedding_dim = 10
    model = init_network(vocab_size, embedding_dim)

    history = [backward(model, X, y, alpha=0.05) for _ in range(50)]

    plt.plot(range(len(history)), history)
    plt.title("Loss over time")
    plt.xlabel("Iteration")
    plt.ylabel("Cross-Entropy Loss")
    plt.show()

    # Extracting and Printing Embeddings
    embeddings = model["w1"]
    prompt_embeddings = [get_prompt_embedding(prompt, word_to_id, embeddings) for prompt in tokens_nested]

    print("\nPrompt Embeddings:")
    for i, embedding in enumerate(prompt_embeddings):
        print(f"Prompt {i + 1} Embedding: {embedding}")

    datas['Prompt Embeddings'] = prompt_embeddings
    X_embedding = torch.tensor([l for l in datas['Prompt Embeddings']], dtype=torch.float32)
    X_mark = torch.tensor([float(m) for m in datas['Mark']], dtype=torch.float32)
    X_reward = torch.tensor(np.array([float(r) for r in datas['Reward']]), dtype=torch.float32)

    # data for reward model
    X = torch.cat((X_embedding, X_mark.unsqueeze(1)), dim=1)
    y = X_reward.unsqueeze(1)

    from sklearn.model_selection import train_test_split

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Prépare les DataLoaders pour train et val
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=4, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=4)

    # Modèle
    input_dim = X_train.shape[1]
    hidden_dim = 64
    output_dim = 1
    reward_model = RewardModel(input_dim, hidden_dim, output_dim)

    # Train
    reward_model = train_reward_model(reward_model, train_loader, num_epochs=10, learning_rate=0.001)

    # Evaluation
    evaluate_reward_model(reward_model, val_loader)
    
    save_model(reward_model, "reward_model.pth")
    import pickle
    with open("word_to_id.pkl", "wb") as f:
        pickle.dump(word_to_id, f)
    np.save("embeddings.npy", embeddings)

