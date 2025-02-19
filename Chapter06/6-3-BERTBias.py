import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM
from transformers import Trainer, TrainingArguments
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def load_bert_model():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    return tokenizer, model

def get_word_embedding(word, tokenizer, model):
    inputs = tokenizer(word, return_tensors="pt")
    
    # Check if we're using BertForMaskedLM or BertModel
    if isinstance(model, BertForMaskedLM):
        # Get the base BERT model from the MLM model
        outputs = model.bert(**inputs)
    else:
        outputs = model(**inputs)
        
    return outputs.last_hidden_state[0][1:-1].mean(dim=0).detach().numpy()

def analyze_gender_bias(model=None, tokenizer=None):
    if model is None or tokenizer is None:
        tokenizer, model = load_bert_model()
    
    # Define gender-specific word pairs
    male_words = ["he", "man", "father", "son", "brother", "uncle"]
    female_words = ["she", "woman", "mother", "daughter", "sister", "aunt"]
    
    # Define profession words
    professions = ["doctor", "nurse", "engineer", "teacher", "scientist", "assistant"]
    
    # Get embeddings
    male_embeddings = np.array([get_word_embedding(w, tokenizer, model) for w in male_words])
    female_embeddings = np.array([get_word_embedding(w, tokenizer, model) for w in female_words])
    profession_embeddings = np.array([get_word_embedding(w, tokenizer, model) for w in professions])
    
    # Calculate gender direction
    gender_direction = (male_embeddings.mean(axis=0) - female_embeddings.mean(axis=0))
    gender_direction = gender_direction / np.linalg.norm(gender_direction)
    
    # Calculate bias scores for professions
    bias_scores = {}
    for prof, emb in zip(professions, profession_embeddings):
        # Project profession embedding onto gender direction
        bias = np.dot(emb, gender_direction)
        bias_scores[prof] = bias
    
    return bias_scores

class GenderNeutralDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer):
        self.training_texts = [
            "The doctor performed surgery.",
            "The nurse helped the patient.",
            "The engineer designed the bridge.",
            "The teacher educated the students.",
            "The scientist conducted research.",
            "The assistant organized the meeting."
        ]
        
        self.encodings = tokenizer(
            self.training_texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = item['input_ids'].clone()
        return item
        
    def __len__(self):
        return len(self.training_texts)

def fine_tune_bert():
    """Fine-tune BERT on gender-neutral dataset"""
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForMaskedLM.from_pretrained('bert-base-uncased')
    
    # Create dataset
    dataset = GenderNeutralDataset(tokenizer)
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./bert_fine_tuned",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        logging_steps=10,
        save_steps=50,
        learning_rate=2e-5
    )
    
    # Create trainer and fine-tune
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset
    )
    
    trainer.train()
    return model, tokenizer

# Demonstrate bias before and after fine-tuning
if __name__ == "__main__":
    print("Original BERT bias scores:")
    original_bias = analyze_gender_bias()
    for prof, score in original_bias.items():
        print(f"{prof}: {score:.3f}")
    
    # Fine-tune BERT
    fine_tuned_model, tokenizer = fine_tune_bert()
    fine_tuned_model.to("cpu")
    
    print("\nFine-tuned BERT bias scores:")
    fine_tuned_bias = analyze_gender_bias(fine_tuned_model, tokenizer)
    for prof, score in fine_tuned_bias.items():
        print(f"{prof}: {score:.3f}")