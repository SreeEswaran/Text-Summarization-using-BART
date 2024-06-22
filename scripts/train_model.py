import pandas as pd
from transformers import BartTokenizer, BartForConditionalGeneration, Trainer, TrainingArguments
from datasets import Dataset

def train_model(train_file, val_file, model_dir):
    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)
    
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    
    def preprocess_function(examples):
        inputs = [doc for doc in examples['text']]
        model_inputs = tokenizer(inputs, max_length=1024, truncation=True, padding="max_length")
        return model_inputs
    
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    
    train_dataset = train_dataset.map(preprocess_function, batched=True)
    val_dataset = val_dataset.map(preprocess_function, batched=True)
    
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
    
    training_args = TrainingArguments(
        output_dir=model_dir,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=1,
        load_best_model_at_end=True,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    
    trainer.train()
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

if __name__ == "__main__":
    train_model('data/processed/train.csv', 'data/processed/val.csv', 'models/bart_model')
