import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

# Step 1: Load the dataset
splits = {'train': 'train.jsonl.gz', 'validation': 'validation.jsonl.gz'}
dataset = load_dataset("json", data_files=splits)

# Step 2: Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Step 3: Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)

# Apply tokenization to the dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Step 4: Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    logging_dir="./logs",  # Directory to store logs
    logging_steps=500,  # Log every 500 steps (can adjust as needed)
    save_steps=500,  # Save checkpoints every 500 steps
    save_total_limit=3,  # Save only the 3 most recent checkpoints
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    disable_tqdm=False,  # Set this to False to show progress bars
    report_to="tensorboard",  # Optional: Log metrics to TensorBoard
)

# Step 5: Load the model
model = AutoModelForCausalLM.from_pretrained("gpt2")

# Step 6: Set up the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
)

# Step 7: Train the model
trainer.train()

# Step 8: Save the fine-tuned model and tokenizer
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")

# Optional: Evaluate the model
trainer.evaluate()
