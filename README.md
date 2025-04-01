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

### Main Implementation (train_combined.py)
- [x] Basic JEPA implementation
- [x] CNN/ResNet/Vision Transformer context encoders
- [x] MLP/Embedding target encoders
- [x] MLP/Residual/Transformer predictor networks
- [x] Data processing for Atari-HEAD dataset
- [x] Self-supervised learning with masking
- [x] Unified training script with multiple modes
- [x] Support for both real and synthetic data

### Supported Features
- [x] Simplified synthetic data generation
- [x] Multiple data processing approaches (real, fixed, synthetic)
- [x] Vision Transformer context encoder
- [x] Transformer-based predictor
- [x] Support for masking strategies

### Future Work
- [x] Integrate all advanced features into the main training script
- [x] Self-supervised pretraining with masking
- [ ] Multi-game training
- [ ] Comprehensive evaluation metrics
- [ ] Performance optimizations for large datasets
- [ ] Curriculum learning strategies
- [ ] Integration with other game environments

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

The project provides a unified training script that supports multiple data processing modes and model architectures:

```bash
python atari_jepa/scripts/train_combined.py [OPTIONS]
```

### Training Modes

1. **Real Data Mode**: Process and train on the actual Atari-HEAD dataset.
   ```bash
   python atari_jepa/scripts/train_combined.py --data_mode real --data_path /path/to/atari_head_dataset --game_name breakout
   ```

2. **Fixed Data Mode**: Use a more robust data processing approach.
   ```bash
   python atari_jepa/scripts/train_combined.py --data_mode fixed --data_path /path/to/atari_head_dataset --game_name breakout
   ```

3. **Synthetic Data Mode**: Generate synthetic data for quick testing without needing the dataset.
   ```bash
   python atari_jepa/scripts/train_combined.py --data_mode synthetic --game_name breakout --synthetic_samples 1000
   ```

### Common Options

- `--data_mode`: Data processing mode to use (real, synthetic, or fixed)
- `--data_path`: Path to the Atari-HEAD dataset (required for real and fixed modes)
- `--game_name`: Name of the game to train on (default: breakout)
- `--synthetic_samples`: Number of synthetic samples to generate (synthetic mode only)
- `--output_dir`: Directory to save outputs
- `--batch_size`: Batch size for training
- `--epochs`: Number of epochs to train
- `--lr`: Learning rate
- `--device`: Device to train on (cuda or cpu)

### Model Architecture Options

- `--context_encoder`: Type of context encoder (cnn, resnet, vit)
- `--target_encoder`: Type of target encoder (standard, embedding)
- `--predictor`: Type of predictor (mlp, residual, transformer)
- `--embedding_dim`: Dimension of embeddings
- `--hidden_dim`: Dimension of hidden layers
- `--use_masking`: Enable masking for self-supervised training
- `--mask_ratio`: Ratio of tokens/patches to mask during training

## Example training:

Here's an example of training the model with advanced features using synthetic data:

```bash
python atari_jepa/scripts/train_combined.py --data_mode synthetic --synthetic_samples 100 --context_encoder vit --predictor transformer --embedding_dim 256 --hidden_dim 512 --use_masking --mask_ratio 0.3 --batch_size 8 --epochs 20
```

### Parameter Explanation:

- `--data_mode synthetic`: Uses synthetic data generation instead of real Atari-HEAD dataset
- `--synthetic_samples 100`: Generates 100 random frame-action pairs for training
- `--context_encoder vit`: Uses Vision Transformer architecture for encoding frames
- `--predictor transformer`: Uses Transformer-based predictor with self-attention
- `--embedding_dim 256`: Sets the dimension of latent embeddings to 256
- `--hidden_dim 512`: Sets the dimension of hidden layers to 512
- `--use_masking`: Enables self-supervised learning with patch masking
- `--mask_ratio 0.3`: Masks 30% of the input patches during training
- `--batch_size 8`: Processes 8 samples per training iteration
- `--epochs 20`: Trains for 20 complete passes through the dataset

### What's Happening During Training:

1. **Data Preparation**: The script generates 100 synthetic frame-action pairs and splits them into training (80%) and validation (20%) sets.

2. **Model Creation**: A JEPA model is created with:
   - Vision Transformer context encoder for processing frames
   - Standard action embedding target encoder
   - Transformer-based predictor for mapping frame embeddings to action embeddings

3. **Training Process**:
   - During each epoch, the model processes batches of frames with 30% of patches randomly masked
   - The Vision Transformer encoder extracts features from the partially masked frames
   - The Transformer predictor maps these features to predicted action embeddings
   - Loss is calculated using contrastive learning (comparing predicted vs. actual action embeddings)

4. **Validation**: Every 5 epochs, the model is evaluated on the validation set with masking disabled

### Training Results:

```
Final evaluation...
Test Loss: 1.8572
Test Accuracy: 0.1500
Test MRR: 0.3760
```

- **Test Loss**: Final contrastive loss value (lower is better)
- **Test Accuracy**: Percentage of correctly predicted actions (with synthetic random data, this is near random chance)
- **Test MRR**: Mean Reciprocal Rank - a measure of ranking quality (higher is better)

The model saves checkpoints during training, with the final model stored at `../outputs/[timestamp]/checkpoints/final_model.pth`.

> **Note**: When using synthetic data, performance metrics are primarily for testing the pipeline functionality rather than actual predictive performance. For meaningful results, use real Atari-HEAD data with `--data_mode real`.

## Evaluation

Evaluation is not ready yet as I have to plug a game simulator but in principle will work like this:

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
