# Google Play Review Sentiment Classifier

This repository contains the `NLP_f.ipynb` notebook used to fine-tune a BERT-based classifier on Google Play Store reviews. The workflow loads raw reviews, maps the original 1–5 star ratings to *negative / neutral / positive* sentiments, and trains a `bert-base-cased` encoder with a lightweight classification head to predict the sentiment of unseen reviews.

## Notebook Highlights
- **Environment setup** – installs Hugging Face Transformers, PyTorch, scikit-learn, Seaborn, and other utilities for data wrangling and visualization (`NLP_f.ipynb:1-2`).
- **Data ingestion + labeling** – reads `reviews.csv`, inspects shape/NULLs, visualizes score distribution, and derives ternary sentiment labels with `to_sentiment` (`NLP_f.ipynb:1832-1905`).
- **Tokenization pipeline** – loads the `bert-base-cased` tokenizer, explores token lengths, and fixes `MAX_LEN = 160` before building a custom `GPReviewDataset` plus DataLoaders (`NLP_f.ipynb:2476-2736`).
- **Model architecture** – wraps a pretrained `BertModel` with dropout and a linear classifier (`SentimentClassifier`) to output three sentiment logits (`NLP_f.ipynb:2841-2928`).
- **Training loop** – trains for 10 epochs with `AdamW`, linear warmup scheduling, accuracy tracking, and validation after each epoch (`NLP_f.ipynb:2975-3185`).
- **Evaluation & inference** – prints a `classification_report`, plots a confusion matrix, and runs free-form inference on sample text (e.g., "I love completing my todos!") (`NLP_f.ipynb:3227-3402`).

## Data Requirements
The notebook expects a file named `reviews.csv` reachable at `/content/reviews.csv` (adjust the path near the top of the notebook if you store it elsewhere). The source data contains **12,495** Google Play reviews for the productivity app `prox.lab.calclock` collected between 2013-01-14 and 2020-10-28, with columns such as:

| Column | Description |
| --- | --- |
| `reviewId`, `userName`, `userImage` | Metadata that uniquely identifies each review. |
| `content` | Free-text review body – the main feature fed into BERT. |
| `score` | Original 1–5 star rating mapped into {0: negative, 1: neutral, 2: positive}. |
| `thumbsUpCount` | Number of helpful votes. |
| `reviewCreatedVersion`, `appId` | App version and identifier (e.g., `prox.lab.calclock`). |
| `at`, `repliedAt`, `replyContent` | Review timestamp plus developer reply and timestamp. |
| `sortOrder` | Ordering flag from the Google Play scraper.

Only the `content` and derived `sentiment` columns are used for modeling, but the remaining fields are helpful for future feature engineering.

## Environment Setup
1. Create a virtual environment (Python 3.9+ recommended):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   ```
2. Install required libraries:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # pick the wheel that matches your CUDA/CPU
   pip install transformers==4.* scikit-learn pandas numpy matplotlib seaborn
   ```
3. (Optional) Install Jupyter tooling:
   ```bash
   pip install notebook jupyterlab
   ```

## Running the Notebook
1. Place `reviews.csv` where the notebook can access it (update the path assigned to `pd.read_csv` if needed).
2. Launch Jupyter:
   ```bash
   jupyter notebook NLP_f.ipynb
   ```
3. Run the cells sequentially. GPU acceleration is recommended; the code automatically selects `cuda` when available.
4. Monitor training metrics printed every epoch plus the accuracy/loss curves and confusion matrix plots drawn with Matplotlib/Seaborn.
5. Use the final inference cell to test custom review snippets or adapt it into an API/CLI wrapper.

## Reproducing & Extending
- Adjust `MAX_LEN`, `BATCH_SIZE`, and `EPOCHS` if you experiment with longer reviews or limited compute.
- Swap in an alternative pretrained model (e.g., `distilbert-base-uncased`) by updating `MODEL_NAME` and re-running the notebook.
- Augment the dataset with additional app IDs or languages; ensure you rebalance the training split (currently an 80/10/10 train/val/test split with `RANDOM_SEED = 42`).
- Export the trained weights via `torch.save(model.state_dict(), "sentiment_classifier.pt")` for downstream deployment.

## Troubleshooting
- **CUDA OOM**: Reduce `BATCH_SIZE` or `MAX_LEN` and restart the kernel.
- **Tokenizer errors**: Confirm that the Transformers cache is accessible and reinstall the package if the tokenizer files are missing.
- **Imbalanced classes**: Inspect the output of the sentiment count plots and consider class-weighted loss or upsampling if neutral reviews dominate your data slice.

## Repository Layout
```
.
├── NLP_f.ipynb   # Main end-to-end workflow
└── README.md     # Project overview and usage guide
```

Happy fine-tuning!
