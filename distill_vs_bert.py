import time
import statistics as stats
import torch
import warnings
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

# Suppress urllib3 warnings
warnings.filterwarnings("ignore", message=".*urllib3.*")
warnings.filterwarnings("ignore", message=".*NotOpenSSLWarning.*")

# Repro-ish CPU runs
torch.set_num_threads(1)
device = 0 if torch.cuda.is_available() else -1

distil = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=device,
)
model = AutoModelForSequenceClassification.from_pretrained(
    "textattack/bert-base-uncased-SST-2",
    id2label={0: "NEGATIVE", 1: "POSITIVE"},
    label2id={"NEGATIVE": 0, "POSITIVE": 1},
)
tokenizer = AutoTokenizer.from_pretrained("textattack/bert-base-uncased-SST-2")
bert = pipeline(
    "sentiment-analysis",
    model="textattack/bert-base-uncased-SST-2",
    tokenizer=tokenizer,
    device=device,
)

TEXT = "I absolutely love the new features of this AI application!"
# warmup
distil(TEXT)
bert(TEXT)


def bench(p, text, n=15):
    times = []
    with torch.no_grad():
        for _ in range(n):
            t0 = time.perf_counter()
            _ = p(text)
            times.append(time.perf_counter() - t0)
    return stats.mean(times), stats.stdev(times)


print("About to benchmark")
d_mean, d_std = bench(distil, TEXT)
b_mean, b_std = bench(bert, TEXT)

print("DistilBERT:", distil(TEXT))
print("BERT     :", bert(TEXT))
print(f"DistilBERT mean {d_mean:.3f}s ±{d_std:.3f}")
print(f"BERT      mean {b_mean:.3f}s ±{b_std:.3f}")
print(f"Speedup ~{b_mean/d_mean:.2f}x (BERT/Distil)")


def count_params(model):
    return sum(p.numel() for p in model.parameters())


print("Distil params:", count_params(distil.model))
print("BERT   params:", count_params(bert.model))
print("Distil layers:", len(distil.model.distilbert.transformer.layer))
print("BERT   layers:", len(bert.model.bert.encoder.layer))
