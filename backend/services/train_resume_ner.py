"""
Training Script for Resume BERT NER Model
Fine-tunes BERT for token-level classification with BIO tagging
Based on Resume_BERT_NER_Modeling_Task documentation

Run this script to train a custom resume NER model.
"""

from transformers import (
    BertTokenizerFast,
    BertForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification
)
from datasets import Dataset
import torch
import json
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Resume-specific entity schema (BIO tagging) - COMPREHENSIVE
LABEL_LIST = [
    "O",
    "B-NAME", "I-NAME",
    "B-DESIGNATION", "I-DESIGNATION",
    "B-PHONE", "I-PHONE",
    "B-EMAIL", "I-EMAIL",
    "B-EDU", "I-EDU",
    "B-COMPANY", "I-COMPANY",
    "B-LOCATION", "I-LOCATION",
    "B-SKILL-PRIMARY", "I-SKILL-PRIMARY",
    "B-SKILL-SECONDARY", "I-SKILL-SECONDARY",
    "B-EXP-MONTHS", "I-EXP-MONTHS",
    "B-JOB-TITLE", "I-JOB-TITLE",
    "B-PROJECT", "I-PROJECT",
    "B-CERT", "I-CERT",
]

LABEL2ID = {l: i for i, l in enumerate(LABEL_LIST)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}


def tokenize_and_align_labels(example, tokenizer):
    """
    Tokenize text and align BIO labels with subword tokens
    
    Args:
        example: Dict with 'tokens' (list of words) and 'ner_tags' (list of labels)
        tokenizer: BertTokenizerFast instance
        
    Returns:
        Dict with tokenized inputs and aligned labels
    """
    tokenized = tokenizer(
        example["tokens"],
        is_split_into_words=True,
        truncation=True,
        max_length=512
    )
    
    labels = []
    word_ids = tokenized.word_ids()
    
    previous_word_id = None
    for word_id in word_ids:
        if word_id is None:
            # Special tokens get -100 (ignored in loss)
            labels.append(-100)
        elif word_id != previous_word_id:
            # First subword token gets the label
            labels.append(LABEL2ID[example["ner_tags"][word_id]])
        else:
            # Subsequent subword tokens get -100 or I- tag
            labels.append(-100)
        
        previous_word_id = word_id
    
    tokenized["labels"] = labels
    return tokenized


def create_sample_dataset():
    """
    Create a sample annotated dataset for comprehensive resume extraction
    In production, replace with real annotated resume data
    """
    examples = [
        {
            "tokens": ["John", "Doe"],
            "ner_tags": ["B-NAME", "I-NAME"]
        },
        {
            "tokens": ["Senior", "Software", "Engineer"],
            "ner_tags": ["B-DESIGNATION", "I-DESIGNATION", "I-DESIGNATION"]
        },
        {
            "tokens": ["Contact", ":", "+1", "555", "1234", "|", "john.doe@email.com"],
            "ner_tags": ["O", "O", "B-PHONE", "I-PHONE", "I-PHONE", "O", "B-EMAIL"]
        },
        {
            "tokens": ["MS", "Computer", "Science", ",", "Stanford", "University"],
            "ner_tags": ["B-EDU", "I-EDU", "I-EDU", "O", "I-EDU", "I-EDU"]
        },
        {
            "tokens": ["Currently", "working", "at", "Google"],
            "ner_tags": ["O", "O", "O", "B-COMPANY"]
        },
        {
            "tokens": ["Based", "in", "San", "Francisco", ",", "CA"],
            "ner_tags": ["O", "O", "B-LOCATION", "I-LOCATION", "O", "I-LOCATION"]
        },
        {
            "tokens": ["Primary", "skills", ":", "Python", ",", "Django", ",", "AWS"],
            "ner_tags": ["O", "O", "O", "B-SKILL-PRIMARY", "O", "B-SKILL-PRIMARY", "O", "B-SKILL-PRIMARY"]
        },
        {
            "tokens": ["Also", "familiar", "with", "Docker", "and", "Kubernetes"],
            "ner_tags": ["O", "O", "O", "B-SKILL-SECONDARY", "O", "B-SKILL-SECONDARY"]
        },
        {
            "tokens": ["5", "years", "of", "experience"],
            "ner_tags": ["B-EXP-MONTHS", "I-EXP-MONTHS", "O", "O"]
        },
        {
            "tokens": ["Led", "backend", "development", "as", "Tech", "Lead"],
            "ner_tags": ["O", "O", "O", "O", "B-JOB-TITLE", "I-JOB-TITLE"]
        },
        {
            "tokens": ["Developed", "E-commerce", "Platform", "using", "microservices"],
            "ner_tags": ["O", "B-PROJECT", "I-PROJECT", "O", "O"]
        },
        {
            "tokens": ["AWS", "Certified", "Solutions", "Architect"],
            "ner_tags": ["B-CERT", "I-CERT", "I-CERT", "I-CERT"]
        }
    ]
    
    return Dataset.from_list(examples)


def train_resume_ner_model(
    output_dir: str = "./models/resume-bert-ner",
    base_model: str = "bert-base-uncased",
    num_epochs: int = 3,
    learning_rate: float = 2e-5,
    batch_size: int = 8
):
    """
    Train BERT model for resume NER
    
    Args:
        output_dir: Directory to save trained model
        base_model: Base BERT model to fine-tune
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        batch_size: Training batch size
    """
    logger.info("=" * 60)
    logger.info("Resume BERT NER Training Pipeline")
    logger.info("=" * 60)
    
    # Step 1: Load tokenizer
    logger.info(f"Loading tokenizer: {base_model}")
    tokenizer = BertTokenizerFast.from_pretrained(base_model)
    
    # Step 2: Create dataset (replace with real data in production)
    logger.info("Creating sample dataset...")
    dataset = create_sample_dataset()
    train_dataset = dataset.map(
        lambda x: tokenize_and_align_labels(x, tokenizer),
        batched=False
    )
    
    logger.info(f"Dataset size: {len(train_dataset)} examples")
    
    # Step 3: Initialize model
    logger.info(f"Initializing BERT model for token classification...")
    model = BertForTokenClassification.from_pretrained(
        base_model,
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )
    
    # Step 4: Configure training
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="no",  # No validation set in sample
        push_to_hub=False,
    )
    
    # Step 5: Initialize trainer
    data_collator = DataCollatorForTokenClassification(tokenizer)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator
    )
    
    # Step 6: Train
    logger.info("Starting training...")
    trainer.train()
    
    # Step 7: Save model
    logger.info(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save label mapping
    with open(os.path.join(output_dir, "label_mapping.json"), "w") as f:
        json.dump({
            "label2id": LABEL2ID,
            "id2label": ID2LABEL,
            "label_list": LABEL_LIST
        }, f, indent=2)
    
    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info(f"Model saved to: {output_dir}")
    logger.info("=" * 60)
    
    return trainer


def test_inference(model_path: str):
    """
    Test the trained model on sample resume text
    """
    from transformers import pipeline
    
    logger.info(f"Loading trained model from {model_path}")
    ner_pipeline = pipeline(
        "ner",
        model=model_path,
        tokenizer=model_path,
        aggregation_strategy="simple"
    )
    
    test_text = """
    I have 5 years of Software Engineering experience.
    Skilled in Python and Docker.
    Completed MS Data Science from EPITA.
    AWS Solutions Architect certified.
    """
    
    logger.info("Testing inference...")
    logger.info(f"Input: {test_text}")
    
    results = ner_pipeline(test_text)
    
    logger.info("\nExtracted Entities:")
    for entity in results:
        logger.info(f"  {entity['entity_group']}: {entity['word']} (score: {entity['score']:.2f})")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Resume BERT NER Model")
    parser.add_argument("--output_dir", type=str, default="./models/resume-bert-ner",
                        help="Output directory for trained model")
    parser.add_argument("--base_model", type=str, default="bert-base-uncased",
                        help="Base BERT model to fine-tune")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Training batch size")
    parser.add_argument("--test", action="store_true",
                        help="Test the model after training")
    
    args = parser.parse_args()
    
    # Train model
    trainer = train_resume_ner_model(
        output_dir=args.output_dir,
        base_model=args.base_model,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size
    )
    
    # Test inference if requested
    if args.test:
        test_inference(args.output_dir)
