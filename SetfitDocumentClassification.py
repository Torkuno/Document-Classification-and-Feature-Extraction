import os
import torch
import evaluate
import pandas as pd
import numpy as np
from setfit import SetFitModel, SetFitTrainer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from ClassifyPDF import extract_text_from_pdf
import warnings
warnings.filterwarnings('ignore')

# Create initial dataframe with 'label' and 'text' columns
df = pd.DataFrame({
    'label': [],
    'text': []
})

label2id = {"contract": 0, "email": 1, "invoice": 2, "resume": 3}
id2label = {v: k for k, v in label2id.items()}

# Print sampling results
print("\nAfter sampling:")
print(df['label'].value_counts())

# Add extracted text from PDFs in testing directory
test_dir = "./img_classification_testing_2"
pdf_texts = []
pdf_labels = []
label = None

# Go through each subfolder in the testing directory
for subfolder in os.listdir(test_dir):
    subfolder_path = os.path.join(test_dir, subfolder)
    if os.path.isdir(subfolder_path):
        # Go through each PDF file in the subfolder
        for file in os.listdir(subfolder_path):
            if file.endswith('.pdf'):
                label = label2id[subfolder]

                # Extract text from the PDF
                pdf_path = os.path.join(subfolder_path, file)
                text = extract_text_from_pdf(pdf_path)

                # Add to the list of texts and files
                pdf_texts.append(text)
                pdf_labels.append(label)

print(f"Successfully extracted text from {len(pdf_texts)} PDF files")

# Add extracted texts to dataframe
test_texts_df = pd.DataFrame({
    'label': pdf_labels,
    'text': pdf_texts
})

# Concatenate the extracted texts with the original dataframe
df = pd.concat([df, test_texts_df], ignore_index=True)

# Create train-test split
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

# Filter out unnecessary columns
train_df = train_df[['text', 'label']]
test_df = test_df[['text', 'label']]

# Convert back to HuggingFace datasets
from datasets import Dataset
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# Print dataset information
print("\nDataset Overview:")
print(f"Training samples: {len(train_dataset)}")
print(f"Testing samples: {len(test_dataset)}")
print("\nLabel distribution in training set:")
labels = train_dataset['label']
unique_labels, counts = np.unique(labels, return_counts=True)
for label, count in zip(unique_labels, counts):
    print(f"{label}: {count} samples")

# Convert datasets to pandas DataFrames for preview
train_df_preview = train_dataset.to_pandas()
test_df_preview = test_dataset.to_pandas()

# Print the first few lines of the datasets
print("Train Dataset Preview:")
print(train_df_preview.head())

print("Test Dataset Preview:")
print(test_df_preview.head())

# Initialize the model
model = SetFitModel.from_pretrained("sentence-transformers/paraphrase-MiniLM-L6-v2")

# Create the trainer with smaller batch size
trainer = SetFitTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    metric="accuracy",    # Use accuracy for evaluation
    batch_size=16,        # Reduced from 16
    num_iterations=20,    # Number of text pairs to generate for contrastive learning
    num_epochs=2,         # Number of epochs to use for contrastive learning
    column_mapping={"text": "text", "label": "label"}  # Specify column mapping
)

# Train the model
trainer.train()

# Get predictions for the test set in smaller batches
batch_size = 32
all_predictions = []
for i in range(0, len(test_dataset), batch_size):
    batch = test_dataset[i:i+batch_size]["text"]
    # Debugging: Print the batch of text being fed into the model
    print("Batch being fed into the model:", batch)
    predictions = trainer.model(batch)
    all_predictions.extend(predictions)

# Check if predictions are integers and handle them directly
all_predictions_indices = [int(pred) if isinstance(pred, torch.Tensor) else pred for pred in all_predictions]

# Debugging: Print the first few converted predictions to verify
print("Converted Predictions (first 10):", all_predictions_indices[:10])

# Evaluate the model using indices
accuracy = evaluate.load("accuracy")
results = accuracy.compute(predictions=all_predictions_indices, references=test_dataset["label"])
print(f"\nOverall Accuracy: {results['accuracy']:.3f}")

# Print detailed classification report
label_names = [key for key in label2id.keys()]
print("\nDetailed Classification Report:")

print(classification_report(test_dataset["label"], all_predictions_indices, target_names=label_names))

# Save the model
trainer.model.save_pretrained("./docs_classifier_model")
