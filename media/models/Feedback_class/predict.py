import torch
import numpy as np
from train import RewardModel, tokenize, get_prompt_embedding  # à ajuster selon où sont tes fonctions

# === Charger word_to_id et embeddings ===
import pickle

with open("word_to_id.pkl", "rb") as f:
    word_to_id = pickle.load(f)
embeddings = np.load("embeddings.npy")  # c’est model["w1"] sauvegardé

def predict_reward(model_path, word_to_id, embeddings):
    input_dim = embeddings.shape[1] + 1  # Embedding + Mark
    hidden_dim = 64
    output_dim = 1
    model = RewardModel(input_dim, hidden_dim, output_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    while True:
        prompt = input("\nEntrez un prompt (ou 'quit' pour sortir): ")
        if prompt.strip().lower() == 'quit':
            break
        try:
            note = float(input("Entrez une note (1 à 5): "))
            if not (1 <= note <= 5):
                print("La note doit être entre 1 et 5.")
                continue
        except ValueError:
            print("Merci d’entrer un nombre pour la note.")
            continue

        # Normaliser la note
        norm_note = (note - 1) / 4

        # Embedding du prompt
        prompt_tokens = tokenize(prompt)
        prompt_emb = get_prompt_embedding(prompt_tokens, word_to_id, embeddings)
        # Ici, plus besoin de tester si c'est un vecteur nul !

        if np.all(prompt_emb == 0):
            print("Aucun mot du prompt n'est connu du vocabulaire !")
            continue

        # Créer l'entrée modèle
        X = torch.tensor(np.concatenate([prompt_emb, [norm_note]]), dtype=torch.float32).unsqueeze(0)

        # Prédire
        with torch.no_grad():
            pred_reward = model(X).item()

        print(f"\nReward prédit (normalisé 0-1) : {pred_reward:.3f}")
        break

if __name__ == "__main__":
    predict_reward("reward_model.pth", word_to_id, embeddings)
