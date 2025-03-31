# Atari JEPA - Frame-to-Action Prediction

This project implements a Joint Embedding Predictive Architecture (JEPA) for predicting player actions from Atari Breakout game frames using the Atari-HEAD dataset.

## Project Structure

```
atari_jepa/
│
├── data/                   # Data loading and processing
│   ├── __init__.py
│   ├── data_loader.py      # Dataset loaders
│   └── data_processing.py  # Data extraction and preprocessing
│
├── models/                 # Model definitions
│   ├── __init__.py
│   ├── context_encoder.py  # Frame encoder
│   ├── target_encoder.py   # Action encoder
│   ├── predictor.py        # Predictor network
│   └── jepa.py             # Full JEPA implementation
│
├── utils/                  # Utility functions
│   ├── __init__.py
│   ├── visualization.py    # Visualization tools
│   └── metrics.py          # Evaluation metrics
│
├── scripts/                # Training and evaluation scripts
│   ├── train.py            # Training script
│   └── evaluate.py         # Evaluation script
│
├── notebooks/              # Jupyter notebooks for exploration and demos
│
├── venv/                   # Virtual environment
│
└── requirements.txt        # Project dependencies
```

## I-JEPA Alignment Considerations

This implementation has been compared with Facebook Research's I-JEPA implementation ([facebookresearch/ijepa](https://github.com/facebookresearch/ijepa)). Here are key considerations and potential improvements to align our implementation more closely with I-JEPA:

### Key Differences

1. **Encoder Architecture**: 
   - **Facebook I-JEPA**: Uses Vision Transformers (ViT) with positional embeddings and self-attention.
   - **Current Implementation**: Uses CNN/ResNet for frames and MLP/embedding for actions.

2. **Predictor Design**:
   - **Facebook I-JEPA**: Uses transformer-based predictor with masked tokens and multiple attention blocks.
   - **Current Implementation**: Uses simpler MLP or residual block-based predictor network.

3. **Masking Strategy**:
   - **Facebook I-JEPA**: Predicts masked image patches from unmasked patches (self-supervised).
   - **Current Implementation**: Directly predicts action embeddings from frame embeddings (supervised).

4. **Target Space**:
   - **Facebook I-JEPA**: Same-modality prediction (image patches to image patches).
   - **Current Implementation**: Cross-modality prediction (frames to actions).

### Planned Improvements

1. **Enhanced Predictor Architecture**:
   - Replace the simple MLP predictor with a transformer-based architecture.
   - Add self-attention mechanisms to capture relationships within frames.

2. **Masking Strategies**:
   - Implement frame masking where parts of input frames are masked.
   - Support multi-block masking as in I-JEPA.

3. **Positional Encodings**:
   - Add sinusoidal positional encodings to both encoders.
   - Support 2D-aware position encoding for image frames.

4. **Self-supervised Pretraining**:
   - Add self-supervised pretraining phase where the model predicts masked frame regions.
   - Fine-tune on action prediction as a downstream task.

5. **Loss Function Refinements**:
   - Focus on cosine similarity between predicted and target embeddings.
   - Implement InfoNCE-style contrastive losses.

## Implementation Status

- [x] Basic JEPA implementation
- [x] CNN/ResNet context encoder
- [x] MLP/Embedding target encoder
- [x] Basic predictor networks
- [ ] Vision Transformer context encoder
- [ ] Transformer-based predictor
- [ ] Masking strategies
- [ ] Self-supervised pretraining

## Setup

1. Create a virtual environment:
   ```
   python -m venv venv
   ```

2. Activate the virtual environment:
   ```
   # On Linux/Mac
   source venv/bin/activate
   
   # On Windows
   .\venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Data Preparation

The code expects the Atari-HEAD dataset in the following format:
- Breakout game data extracted from the zip file
- Game frames accessible as image files
- Label files containing action data

## Training

To train the JEPA model on the Breakout game data:

```
python scripts/train.py --data_path /media/robomotic/bumbledisk/github/game-jepa/Atari --game_name breakout
```

For advanced I-JEPA style training with masking and transformer-based models:

```
python scripts/train.py --data_path /media/robomotic/bumbledisk/github/game-jepa/Atari --game_name breakout --context_encoder vit --predictor transformer --use_masking
```

## Evaluation

To evaluate the trained model:

```
python scripts/evaluate.py --model_path /path/to/saved/model
```

## Approach

This implementation follows a Joint Embedding Predictive Architecture (JEPA) approach:

1. **Context Encoder**: Encodes game frames into a latent representation
2. **Target Encoder**: Encodes actions into a latent representation
3. **Predictor**: Predicts the action embedding from the frame embedding
4. **Training**: Minimizes the distance between predicted action embeddings and actual action embeddings

This approach allows the model to learn meaningful representations that capture the relationship between game states and human actions without having to predict exact pixel values or action labels directly.

### JEPA vs I-JEPA Adaptation

While the original Facebook I-JEPA focuses on self-supervised learning within the same modality (predicting masked image patches from visible patches), our adaptation applies JEPA principles to cross-modal prediction (frames to actions). 

Our implementation offers:

1. **Cross-Modal Learning**: Learn relationships between visual frames and discrete actions
2. **Behavorial Cloning**: Capture human decision-making patterns from the Atari-HEAD dataset
3. **Multiple Encoder Options**: 
   - CNN/ResNet for performance on limited hardware
   - Vision Transformers for higher representational capacity
4. **Multiple Predictor Options**:
   - MLP/Residual predictors for simple mapping
   - Transformer-based predictors for more complex relationships
5. **Optional Masking**: Apply masking strategies for self-supervised pretraining
