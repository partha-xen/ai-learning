"""
Knowledge Distillation with TF-IDF Student Model (KD-TF-IDF)
===========================================================

This script implements true knowledge distillation (KD) using temperature scaling
to transfer knowledge from a pre-trained teacher model (DistilBERT) to a much
smaller student model (TF-IDF + MLP).

KNOWLEDGE DISTILLATION FUNDAMENTALS:
====================================
Knowledge distillation is a model compression technique where:
1. TEACHER MODEL: Large, complex model with high accuracy (DistilBERT)
2. STUDENT MODEL: Smaller, faster model (TF-IDF + MLP)
3. KNOWLEDGE TRANSFER: Student learns from teacher's "soft" predictions

WHY KNOWLEDGE DISTILLATION?
==========================
- DEPLOYMENT: Large models are too slow/heavy for production
- EFFICIENCY: Need fast inference for real-time applications
- ACCURACY: Want to preserve teacher's knowledge in smaller model
- RESOURCE CONSTRAINTS: Mobile devices, edge computing, etc.

TEMPERATURE SCALING IN KD:
=========================
- Temperature T > 1 makes softmax "softer" (more uniform probabilities)
- Soft targets reveal teacher's confidence and decision boundaries
- Student learns not just "what" teacher predicts, but "how confident"
- T² scaling in loss compensates for temperature scaling

DISTILLATION LOSS FORMULA:
=========================
L_KD = α × L_CE(hard_labels) + (1-α) × T² × KL(student_T || teacher_T)

Where:
- L_CE: Cross-entropy loss on ground truth labels
- KL: Kullback-Leibler divergence between student and teacher distributions
- α: Weighting factor (0.3 means 30% hard labels, 70% soft targets)
- T: Temperature parameter (higher = softer distributions)

PIPELINE OVERVIEW:
=================
1. Load SST-2 dataset (Stanford Sentiment Treebank)
2. Teacher (DistilBERT) generates soft targets at temperature T
3. Student: TF-IDF features → small MLP
4. Train student with combined KD loss
5. Compare accuracy and latency between teacher and student

Run example:
    python kd_tfidf_student.py --train_size 3000 --val_size 800 --T 2.0 --alpha 0.3 --epochs 6
"""

import argparse
import time
from dataclasses import dataclass

import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


# =============================================================================
# CONFIGURATION AND HYPERPARAMETERS
# =============================================================================
def get_args():
    """
    Parse command-line arguments for knowledge distillation experiment.

    KNOWLEDGE DISTILLATION HYPERPARAMETERS:
    - T (temperature): Controls softness of teacher's probability distribution
      * T=1: Normal softmax (sharp distributions)
      * T>1: Softer distributions, more information about teacher's uncertainty
      * Typical range: [2.0, 4.0] for good knowledge transfer

    - alpha: Weighting between hard labels and soft targets
      * alpha=0.0: Pure distillation (no ground truth labels)
      * alpha=1.0: Pure supervised learning (no teacher knowledge)
      * Typical range: [0.1, 0.7] for balanced learning

    - max_features: TF-IDF vocabulary size (controls student model capacity)
      * Larger = more expressive but slower
      * Smaller = faster but less capacity to learn from teacher
    """
    p = argparse.ArgumentParser()
    p.add_argument(
        "--teacher",
        type=str,
        default="distilbert-base-uncased-finetuned-sst-2-english",
        help="Pre-trained teacher model for knowledge distillation",
    )
    p.add_argument(
        "--train_size",
        type=int,
        default=1500,
        help="Number of training examples (smaller = faster experiment)",
    )
    p.add_argument(
        "--val_size", type=int, default=400, help="Number of validation examples"
    )
    p.add_argument(
        "--max_features",
        type=int,
        default=20000,
        help="TF-IDF vocabulary size (student model capacity)",
    )
    p.add_argument(
        "--T",
        type=float,
        default=2.0,
        help="KD temperature: higher = softer teacher distributions",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=0.3,
        help="Weight for hard labels vs soft targets (0.3 = 30% hard, 70% soft)",
    )
    p.add_argument("--batch_size", type=int, default=64, help="Training batch size")
    p.add_argument("--epochs", type=int, default=6, help="Number of training epochs")
    p.add_argument(
        "--lr", type=float, default=2e-3, help="Learning rate for student model"
    )
    p.add_argument(
        "--hidden", type=int, default=256, help="Hidden layer size in student MLP"
    )
    p.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    return p.parse_args()


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def set_seed(seed: int):
    """
    Set random seeds for reproducibility across all libraries.

    DISTILLATION CONTEXT: Reproducible results are crucial for:
    - Comparing different hyperparameter settings
    - Validating that distillation improvements are real
    - Ensuring fair comparison between teacher and student models
    """
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_numpy_logits(logits: torch.Tensor) -> np.ndarray:
    """Convert PyTorch logits to NumPy array for analysis."""
    return logits.detach().cpu().numpy()


def softmax_temperature(logits: torch.Tensor, T: float) -> torch.Tensor:
    """
    Apply temperature scaling to softmax: softmax(logits / T)

    KNOWLEDGE DISTILLATION CONTEXT:
    - Temperature scaling is the key innovation in knowledge distillation
    - T=1: Normal softmax (sharp, confident predictions)
    - T>1: Softer distributions that reveal teacher's uncertainty
    - Soft targets contain more information than hard labels
    - Student learns teacher's decision boundaries, not just final predictions

    Example:
    - Hard target: [0, 1] (definitely positive)
    - Soft target (T=2): [0.3, 0.7] (teacher is confident but not certain)
    - Soft target (T=3): [0.4, 0.6] (teacher is less confident)
    """
    return F.softmax(logits / T, dim=-1)


@dataclass
class KDData:
    """
    Container for knowledge distillation dataset.

    KNOWLEDGE DISTILLATION DATA STRUCTURE:
    - X_train/X_val: TF-IDF features (student's input representation)
    - y_train/y_val: Hard labels (ground truth for supervised learning)
    - t_train_probs_T/t_val_probs_T: Soft targets from teacher at temperature T

    The soft targets are the key to knowledge distillation - they contain
    the teacher's probability distributions over classes, which encode
    much more information than simple hard labels.
    """

    # Student model inputs (TF-IDF features)
    X_train: np.ndarray
    X_val: np.ndarray

    # Ground truth labels (for hard loss component)
    y_train: np.ndarray
    y_val: np.ndarray

    # Teacher's soft targets at temperature T (for distillation loss)
    t_train_probs_T: np.ndarray
    t_val_probs_T: np.ndarray


# =============================================================================
# STUDENT MODEL ARCHITECTURE
# =============================================================================
class StudentMLP(nn.Module):
    """
    Simple Multi-Layer Perceptron student model.

    KNOWLEDGE DISTILLATION DESIGN PRINCIPLES:
    - SIMPLICITY: Much smaller than teacher (TF-IDF vs transformer)
    - SPEED: Fast inference for deployment scenarios
    - CAPACITY: Just enough to learn from teacher's knowledge

    ARCHITECTURE CHOICES:
    - Input: TF-IDF features (bag-of-words representation)
    - Hidden: Single hidden layer with ReLU activation
    - Dropout: Prevents overfitting to limited training data
    - Output: 2 classes (positive/negative sentiment)

    SIZE COMPARISON:
    - Teacher (DistilBERT): ~67M parameters
    - Student (this MLP): ~5M parameters (with 20k TF-IDF features)
    - Compression ratio: ~13x smaller!
    """

    def __init__(self, in_dim: int, hidden: int = 256, out_dim: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),  # Input layer: TF-IDF → hidden
            nn.ReLU(),  # Non-linearity
            nn.Dropout(p=0.2),  # Regularization
            nn.Linear(hidden, out_dim),  # Output layer: hidden → 2 classes
        )

    def forward(self, x):
        """Forward pass: TF-IDF features → sentiment logits"""
        return self.net(x)


# =============================================================================
# KNOWLEDGE DISTILLATION LOSS FUNCTIONS
# =============================================================================
def kd_loss(
    student_logits: torch.Tensor, teacher_probs_T: torch.Tensor, T: float
) -> torch.Tensor:
    """
    Knowledge Distillation Loss: KL(student_T || teacher_T)

    DISTILLATION LOSS EXPLANATION:
    - Measures how well student's softened predictions match teacher's
    - KL divergence captures difference between probability distributions
    - T² scaling compensates for temperature scaling in gradients

    MATHEMATICAL FORMULA:
    L_KD = T² × KL(softmax(student_logits/T), teacher_probs_T)

    WHERE:
    - student_logits/T: Student's logits scaled by temperature
    - teacher_probs_T: Teacher's probabilities at temperature T
    - T²: Gradient scaling factor (Hinton et al., 2015)

    WHY T² SCALING?
    - Temperature scaling reduces gradient magnitudes by factor of T
    - T² scaling restores proper gradient magnitudes
    - Ensures distillation loss has appropriate weight in combined loss
    """
    # Compute log-softmax of student at temperature T
    log_p_s_T = F.log_softmax(student_logits / T, dim=-1)

    # KL divergence expects log-probabilities as input, probabilities as target
    kld = F.kl_div(log_p_s_T, teacher_probs_T, reduction="batchmean")

    # Apply T² scaling for proper gradient magnitudes
    return (T * T) * kld


def hard_ce_loss(
    student_logits: torch.Tensor, hard_labels: torch.Tensor
) -> torch.Tensor:
    """
    Standard Cross-Entropy Loss on Ground Truth Labels.

    KNOWLEDGE DISTILLATION CONTEXT:
    - Provides direct supervision from ground truth labels
    - Prevents student from completely ignoring true labels
    - Balances distillation with traditional supervised learning
    - α parameter controls relative importance vs soft targets
    """
    return F.cross_entropy(student_logits, hard_labels)


# =============================================================================
# DATASET CONSTRUCTION WITH TEACHER SOFT TARGETS
# =============================================================================
def build_dataset(
    teacher_name: str,
    train_size: int,
    val_size: int,
    max_features: int,
    T: float,
    seed: int,
) -> KDData:
    """
    Build knowledge distillation dataset with teacher soft targets.

    KNOWLEDGE DISTILLATION PIPELINE:
    1. Load SST-2 dataset (Stanford Sentiment Treebank)
    2. Extract TF-IDF features for student model
    3. Generate teacher soft targets at temperature T
    4. Combine hard labels and soft targets for distillation

    KEY INSIGHTS:
    - Teacher soft targets contain rich information about decision boundaries
    - TF-IDF provides simple but effective feature representation
    - Temperature scaling reveals teacher's confidence patterns
    """
    set_seed(seed)

    # Load Stanford Sentiment Treebank dataset
    ds = load_dataset("glue", "sst2")
    train = ds["train"].shuffle(seed=seed).select(range(train_size))
    val = ds["validation"].shuffle(seed=seed).select(range(val_size))

    # Load teacher model (DistilBERT)
    teacher_tok = AutoTokenizer.from_pretrained(teacher_name)
    teacher = AutoModelForSequenceClassification.from_pretrained(teacher_name)
    teacher.eval()  # Set to evaluation mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher.to(device)

    # Extract TF-IDF features for student model
    # KNOWLEDGE DISTILLATION: Simple features force student to learn from teacher
    vec = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),  # Include bigrams for richer features
        stop_words="english",
    )
    X_train = vec.fit_transform(train["sentence"]).astype(np.float32).toarray()
    X_val = vec.transform(val["sentence"]).astype(np.float32).toarray()

    # Ground truth labels
    y_train = np.array(train["label"], dtype=np.int64)
    y_val = np.array(val["label"], dtype=np.int64)

    # Generate teacher soft targets at temperature T
    def teacher_probs_T(texts):
        """
        Generate teacher's probability distributions at temperature T.

        DISTILLATION PROCESS:
        - Teacher processes text through transformer layers
        - Outputs logits for each class
        - Temperature scaling creates soft probability distributions
        - Soft targets encode teacher's confidence and uncertainty
        """
        batch_size = 32  # Process in batches for efficiency
        probs_out = []
        with torch.no_grad():  # No gradients needed for teacher
            for i in range(0, len(texts), batch_size):
                chunk = texts[i : i + batch_size]
                # Tokenize text for teacher model
                tok = teacher_tok(
                    chunk, padding=True, truncation=True, return_tensors="pt"
                ).to(device)
                # Get teacher's logits
                logits = teacher(**tok).logits  # [batch_size, 2]
                # Apply temperature scaling for soft targets
                probsT = softmax_temperature(logits, T)
                probs_out.append(probsT.cpu())
        return torch.cat(probs_out, dim=0).numpy()

    # Generate soft targets for training and validation sets
    t_train_probs_T = teacher_probs_T(train["sentence"])
    t_val_probs_T = teacher_probs_T(val["sentence"])

    # Evaluate teacher baseline performance
    def teacher_predict(texts):
        """Get teacher's hard predictions for baseline comparison."""
        preds = []
        with torch.no_grad():
            for i in range(0, len(texts), 32):
                chunk = texts[i : i + 32]
                tok = teacher_tok(
                    chunk, padding=True, truncation=True, return_tensors="pt"
                ).to(device)
                logits = teacher(**tok).logits
                preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())
        return np.array(preds, dtype=np.int64)

    teacher_val_preds = teacher_predict(val["sentence"])
    teacher_acc = accuracy_score(y_val, teacher_val_preds)
    print(f"[Teacher] Validation accuracy: {teacher_acc:.3f}")

    return KDData(
        X_train=X_train,
        y_train=y_train,
        t_train_probs_T=t_train_probs_T,
        X_val=X_val,
        y_val=y_val,
        t_val_probs_T=t_val_probs_T,
    )


# =============================================================================
# STUDENT MODEL TRAINING WITH KNOWLEDGE DISTILLATION
# =============================================================================
def train_student(
    data: KDData,
    hidden: int,
    T: float,
    alpha: float,
    lr: float,
    batch_size: int,
    epochs: int,
):
    """
    Train student model using knowledge distillation loss.

    KNOWLEDGE DISTILLATION TRAINING PROCESS:
    1. Initialize small student model (TF-IDF → MLP)
    2. For each batch:
       - Forward pass through student model
       - Compute hard loss (CE on ground truth)
       - Compute soft loss (KL divergence with teacher)
       - Combine losses with weighting factor α
       - Backpropagate and update student parameters

    HYPERPARAMETER EFFECTS:
    - α=0.0: Pure distillation (student learns only from teacher)
    - α=1.0: Pure supervised learning (no teacher knowledge)
    - α=0.3: Balanced learning (30% hard labels, 70% soft targets)

    TEMPERATURE EFFECTS:
    - T=1.0: Sharp distributions (less information transfer)
    - T=2.0-4.0: Soft distributions (rich information transfer)
    - T>4.0: Too soft (information loss)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert data to PyTorch tensors
    Xtr = torch.from_numpy(data.X_train).to(device)
    ytr = torch.from_numpy(data.y_train).to(device)
    ttr = torch.from_numpy(data.t_train_probs_T).to(device)

    Xva = torch.from_numpy(data.X_val).to(device)
    yva = torch.from_numpy(data.y_val).to(device)
    tva = torch.from_numpy(data.t_val_probs_T).to(device)

    # Initialize student model
    model = StudentMLP(in_dim=Xtr.shape[1], hidden=hidden, out_dim=2).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # Create data loader for batch training
    train_ds = TensorDataset(Xtr, ytr, ttr)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # Training loop with early stopping
    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        losses, kd_losses, ce_losses = [], [], []

        # Training loop
        for xb, yb, tb in train_dl:
            opt.zero_grad()
            logits = model(xb)  # Student predictions [batch_size, 2]

            # Compute knowledge distillation loss components
            loss_kd = kd_loss(logits, tb, T)  # Soft target loss
            loss_ce = hard_ce_loss(logits, yb)  # Hard label loss

            # Combined loss with weighting
            loss = alpha * loss_ce + (1.0 - alpha) * loss_kd

            # Backpropagation
            loss.backward()
            opt.step()

            # Track losses for monitoring
            losses.append(loss.item())
            kd_losses.append(loss_kd.item())
            ce_losses.append(loss_ce.item())

        # Validation evaluation
        model.eval()
        with torch.no_grad():
            val_logits = model(Xva)
            val_preds = torch.argmax(val_logits, dim=-1)
            val_acc = accuracy_score(yva.cpu().numpy(), val_preds.cpu().numpy())

            # Track KD loss on validation set
            val_kd = kd_loss(val_logits, tva, T).item()

        # Print training progress
        print(
            f"Epoch {epoch:02d}: "
            f"train_loss={np.mean(losses):.4f} "
            f"(kd={np.mean(kd_losses):.4f}, ce={np.mean(ce_losses):.4f}) "
            f"| val_acc={val_acc:.4f} kd_val={val_kd:.4f}"
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"[Student] Best val accuracy: {best_val_acc:.3f}")
    return model


# =============================================================================
# INFERENCE SPEED COMPARISON
# =============================================================================
def time_inference_teacher(teacher_name: str, sample_text: str, n=20):
    """
    Benchmark teacher model inference speed.

    KNOWLEDGE DISTILLATION CONTEXT:
    - Teacher models are typically slow due to transformer architecture
    - Multiple attention layers and large parameter count
    - This motivates the need for distillation to smaller models
    """
    tok = AutoTokenizer.from_pretrained(teacher_name)
    mdl = AutoModelForSequenceClassification.from_pretrained(teacher_name).eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mdl.to(device)

    # Warmup runs (exclude from timing)
    with torch.no_grad():
        _ = mdl(**tok(sample_text, return_tensors="pt").to(device))

    # Time inference
    t = []
    for _ in range(n):
        with torch.no_grad():
            t0 = time.perf_counter()
            _ = mdl(**tok(sample_text, return_tensors="pt").to(device))
            t.append(time.perf_counter() - t0)
    return np.mean(t), np.std(t)


def time_inference_student(
    model: nn.Module, vectorizer: TfidfVectorizer, sample_text: str, n=100
):
    """
    Benchmark student model inference speed.

    KNOWLEDGE DISTILLATION BENEFITS:
    - Student models should be significantly faster than teacher
    - Simple MLP architecture enables fast inference
    - TF-IDF preprocessing is much faster than transformer tokenization
    - Enables deployment in resource-constrained environments
    """
    device = next(model.parameters()).device

    # Warmup runs
    with torch.no_grad():
        _ = model(
            torch.from_numpy(
                vectorizer.transform([sample_text]).astype(np.float32).toarray()
            ).to(device)
        )

    # Time inference
    t = []
    for _ in range(n):
        with torch.no_grad():
            t0 = time.perf_counter()
            # TF-IDF feature extraction
            X = vectorizer.transform([sample_text]).astype(np.float32).toarray()
            X = torch.from_numpy(X).to(device)
            # Student model inference
            _ = model(X)
            t.append(time.perf_counter() - t0)
    return np.mean(t), np.std(t)


# =============================================================================
# MAIN EXPERIMENT PIPELINE
# =============================================================================
def main():
    """
    Main knowledge distillation experiment pipeline.

    EXPERIMENTAL WORKFLOW:
    1. Build dataset with teacher soft targets
    2. Train student model with KD loss
    3. Evaluate accuracy comparison
    4. Benchmark inference speed comparison
    5. Report distillation effectiveness

    EXPECTED RESULTS:
    - Student should achieve 80-90% of teacher's accuracy
    - Student should be 10-50x faster than teacher
    - Student should be 10-20x smaller than teacher
    - Demonstrates successful knowledge transfer
    """
    args = get_args()
    set_seed(args.seed)

    print("Args:", args)

    # Step 1: Build knowledge distillation dataset
    print("\n=== Building KD Dataset ===")
    kddata = build_dataset(
        teacher_name=args.teacher,
        train_size=args.train_size,
        val_size=args.val_size,
        max_features=args.max_features,
        T=args.T,
        seed=args.seed,
    )

    # Step 2: Train student model with knowledge distillation
    print("\n=== Training Student with KD ===")
    student = train_student(
        data=kddata,
        hidden=args.hidden,
        T=args.T,
        alpha=args.alpha,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
    )

    # Step 3: Final accuracy evaluation
    print("\n=== Final Evaluation ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student.to(device)
    with torch.no_grad():
        logits = student(torch.from_numpy(kddata.X_val).to(device))
        preds = torch.argmax(logits, dim=-1).cpu().numpy()
    val_acc = accuracy_score(kddata.y_val, preds)
    print(f"[Student] Final val accuracy: {val_acc:.3f}")

    # Step 4: Inference speed comparison
    print("\n=== Speed Comparison ===")
    sample = "I absolutely love how simple this app is to use!"

    # Teacher speed
    t_mean, t_std = time_inference_teacher(args.teacher, sample_text=sample, n=10)
    print(f"[Teacher] per-example latency: {t_mean:.4f}s ± {t_std:.4f}")

    # Student speed
    vec = TfidfVectorizer(
        max_features=args.max_features, ngram_range=(1, 2), stop_words="english"
    )
    vec.fit(
        load_dataset("glue", "sst2")["train"]
        .shuffle(seed=args.seed)
        .select(range(args.train_size))["sentence"]
    )
    s_mean, s_std = time_inference_student(student, vec, sample_text=sample, n=100)
    print(f"[Student] per-example latency: {s_mean:.5f}s ± {s_std:.5f}")

    # Speedup calculation
    speedup = t_mean / s_mean
    print(f"[Distillation] Speedup: {speedup:.1f}x faster")

    # Hyperparameter tuning tips
    print("\n=== Knowledge Distillation Tips ===")
    print("Hyperparameter tuning guidelines:")
    print("- Temperature T: Start with 2.0, try [1.5, 3.0] range")
    print("- Alpha α: Start with 0.3, try [0.1, 0.7] range")
    print("- Higher T: Softer distributions, more information transfer")
    print("- Higher α: More weight on ground truth labels")
    print("- Lower α: More reliance on teacher knowledge")


if __name__ == "__main__":
    main()
