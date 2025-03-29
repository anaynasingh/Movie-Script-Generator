# MovieScriptGPT

MovieScriptGPT is a Transformer-based model that generates movie scripts given a prompt. Built with PyTorch, it leverages a GPT-style architecture to generate coherent and contextually relevant screenplay content.

## Features
- Uses a Transformer model with multi-head self-attention
- Trained on movie scripts dataset
- Supports prompt-based generation with temperature scaling and top-k sampling
- Can be trained from scratch or loaded from a saved checkpoint

## Installation

### Prerequisites
- Python 3.8+
- PyTorch
- Datasets library
- tqdm

### Install Dependencies
```bash
pip install torch datasets tqdm
```

## Usage

### Training the Model
To train the model from scratch, run:
```bash
python script.py "<initial prompt>"
```
This will automatically train the model if a pre-trained checkpoint is not found.

### Generating Movie Scripts
If the model is already trained, you can generate a script using:
```bash
python script.py "Once upon a time in Hollywood..."
```
This will generate a screenplay continuation based on the given prompt.

## Model Architecture
- **Embedding Layer**: Converts input tokens into dense vector representations.
- **Multi-Head Attention**: Enables contextual understanding of words.
- **Feedforward Network**: Enhances feature extraction.
- **Stacked Transformer Blocks**: Improves deep learning capabilities.
- **Output Layer**: Predicts the next token in sequence.

## Configuration
The model is configured with:
```python
class Config:
    vocab_size = 50257  # GPT-2 tokenizer vocab size
    n_embd = 384        # Embedding dimension
    n_head = 6          # Number of attention heads
    n_layer = 6         # Number of transformer layers
    block_size = 256    # Context window length
    dropout = 0.1       # Dropout rate
    lr = 3e-4           # Learning rate
    batch_size = 32     # Training batch size
```

## Training Details
- The model is trained using cross-entropy loss.
- AdamW optimizer is used with a learning rate of `3e-4`.

## Saving and Loading Model
The trained model is saved as `movie_gpt.pth`.

## Example Output
### Input Prompt:
```text
INT. DARK ALLEY - NIGHT
A lone detective walks cautiously, gripping his gun tightly.
```
### Generated Output:
```text
DETECTIVE HARRIS: (whispering) Show yourself.

A faint rustling echoes through the alley. Suddenly, a shadow moves.

MYSTERIOUS FIGURE: (laughs) You’re too late, detective.

Harris steadies his grip, eyes scanning the darkness...
```

## Future Improvements
- Fine-tuning on more diverse movie scripts
- Improved text coherence with larger models
- Web-based UI for interactive script generation

## License
This project is licensed under the MIT License.

