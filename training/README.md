# Transformer Model Training

This directory contains the implementation and training code for a Transformer-based sequence classification model, specifically designed for spam detection tasks.

## 📁 Project Structure

```
training/
├── README.md                    # This file - project overview
├── MODEL_ARCHITECTURE.md        # 📖 Detailed model architecture documentation
├── models.py                    # 🏗️ Transformer model implementation
├── inference.py                 # 🔮 Model inference and prediction
├── train.py                     # 🚀 Training script
├── config.py                    # ⚙️ Model configuration
├── data_loader.py              # 📊 Data loading utilities
└── utils.py                    # 🛠️ Helper functions
```

## 🏗️ Model Architecture

Our implementation follows the **encoder-only Transformer architecture** (similar to BERT) for sequence classification. The model consists of:

- **Embeddings Layer**: Converts token IDs to dense vectors with positional encoding
- **12 Encoder Layers**: Each with multi-head attention and feed-forward networks
- **Classification Head**: Uses [CLS] token for sequence-level predictions

### Key Features

- **BERT-base Configuration**: 768 hidden size, 12 attention heads, 12 layers
- **Pre-LayerNorm Architecture**: Better training stability
- **Multi-Head Attention**: 12 parallel attention heads for diverse relationship modeling
- **Position-wise Feed-Forward**: 768 → 3072 → 768 with GELU activation

## 📖 Detailed Documentation

For comprehensive architecture explanations, mathematical formulas, dimension flows, and component deep-dives, see:

**[📋 MODEL_ARCHITECTURE.md](MODEL_ARCHITECTURE.md)**

This document includes:

- Complete mathematical formulations
- Step-by-step processing explanations
- Mermaid architecture diagrams
- BERT-base configuration details
- Dimension flow examples
- Component interaction patterns

## 🚀 Quick Start

### 1. Model Configuration

```python
from config import ModelConfig

config = ModelConfig(
    vocab_size=30522,           # BERT-base vocabulary
    hidden_size=768,            # Embedding dimension
    num_attention_heads=12,     # Multi-head attention
    num_hidden_layers=12,       # Encoder layers
    intermediate_size=3072,     # Feed-forward hidden size
    max_position_embeddings=512, # Max sequence length
    num_labels=2                # Binary classification
)
```

### 2. Model Initialization

```python
from models import TransformerForSequenceClassification

model = TransformerForSequenceClassification(config)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
```

### 3. Training

```python
python train.py --config config.json --data_path ../datasets/TREC2007/
```

### 4. Inference

```python
from inference import EmailClassifier

classifier = EmailClassifier("path/to/trained/model")
result = classifier.predict("This is a test email message")
print(f"Prediction: {result['label']} (confidence: {result['confidence']:.3f})")
```

## 📊 Data Processing Pipeline

The model expects preprocessed email data in the following format:

```
[CLS] email_from: sender@domain.com [SEP] email_to: recipient@domain.com [SEP] subject: Email Subject [SEP] message: Email content here
```

### Input Processing Flow:

1. **Tokenization**: Text → Token IDs using BERT tokenizer
2. **Embedding Lookup**: Token IDs → Dense vectors (768-dim)
3. **Position Encoding**: Add positional information
4. **Encoder Processing**: 12 layers of attention + feed-forward
5. **Classification**: [CLS] token → Binary prediction

## 🎯 Model Performance

The model is designed for binary email classification:

- **Class 0**: Ham (legitimate email)
- **Class 1**: Spam (unwanted email)

### Training Metrics:

- **Loss Function**: CrossEntropyLoss
- **Optimizer**: AdamW with learning rate scheduling
- **Evaluation**: Accuracy, Precision, Recall, F1-Score

## 🔧 Configuration Options

Key hyperparameters in `config.py`:

| Parameter             | Default | Description               |
| --------------------- | ------- | ------------------------- |
| `hidden_size`         | 768     | Model embedding dimension |
| `num_attention_heads` | 12      | Number of attention heads |
| `num_hidden_layers`   | 12      | Number of encoder layers  |
| `intermediate_size`   | 3072    | Feed-forward hidden size  |
| `hidden_dropout_prob` | 0.1     | Dropout probability       |
| `learning_rate`       | 2e-5    | Training learning rate    |
| `batch_size`          | 16      | Training batch size       |
| `max_length`          | 512     | Maximum sequence length   |

## 🛠️ Development

### Code Structure

- **`models.py`**: Clean, well-documented model implementation
- **`train.py`**: Complete training loop with validation
- **`inference.py`**: Production-ready inference pipeline
- **`data_loader.py`**: Efficient data loading and preprocessing
- **`utils.py`**: Utility functions for training and evaluation

### Key Design Principles:

1. **Modularity**: Each component is independently testable
2. **Documentation**: Comprehensive docstrings and comments
3. **Efficiency**: Optimized for both training and inference
4. **Flexibility**: Easy to modify for different tasks

## 📈 Usage Examples

### Training a New Model

```bash
# Train with default configuration
python train.py

# Train with custom config
python train.py --config custom_config.json --epochs 5 --batch_size 32

# Resume training from checkpoint
python train.py --resume checkpoints/model_epoch_3.pt
```

### Model Inference

```python
# Single prediction
classifier = EmailClassifier("trained_model.pt")
result = classifier.predict("Your email text here")

# Batch prediction
results = classifier.predict_batch([
    "Email 1 content",
    "Email 2 content",
    "Email 3 content"
])
```

### Model Analysis

```python
# Get attention weights for visualization
model.eval()
with torch.no_grad():
    outputs = model(input_ids)
    attention_weights = model.encoder.layers[0].attention.att_mats
```

## 🔍 Architecture Diagrams

The model architecture is visualized in detail in [MODEL_ARCHITECTURE.md](MODEL_ARCHITECTURE.md), including:

- **Complete Model Flow**: Input → Embeddings → Encoder → Classification
- **Multi-Head Attention**: Parallel processing of 12 attention heads
- **Encoder Layer**: Pre-LayerNorm design with residual connections
- **Dimension Flow**: Tensor shapes through each component

## 📚 References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer paper
- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805) - BERT architecture
- [Layer Normalization](https://arxiv.org/abs/1607.06450) - Pre-LayerNorm benefits

---

For detailed technical documentation and mathematical explanations, see **[MODEL_ARCHITECTURE.md](MODEL_ARCHITECTURE.md)**.
