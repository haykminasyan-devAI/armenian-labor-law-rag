# How to Run RAG Pipeline with Llama 3-8B

## ✅ What's Ready:
- BM25 index built (286 chunks from labor law)
- Llama 3-8B configured to run on GPU
- SLURM job script prepared

---

## 🚀 Option 1: Run on GPU (Recommended)

### Submit to SLURM:
```bash
cd /home/hayk.minasyan/Project/NLP_proj
sbatch run_rag_gpu.sh
```

### Check job status:
```bash
squeue -u hayk.minasyan
```

### View output:
```bash
# Output will be in logs/rag_JOBID.out
# Errors will be in logs/rag_JOBID.err
tail -f logs/rag_*.out
```

---

## 🖥️ Option 2: Interactive GPU Session

### Request GPU:
```bash
srun -p scalar6000q --gres=gpu:1 --mem=32G --time=1:00:00 --pty bash
```

### Run the pipeline:
```bash
cd /home/hayk.minasyan/Project/NLP_proj
source venv/bin/activate
python scripts/test_rag.py
```

---

## 📝 Test Questions (Armenian):

The script will test these questions:
1. "Որո՞նք են նվազագույն աշխատավարձի կանոնները։" (Minimum wage rules)
2. "Ինչպե՞ս է տրամադրվում տարեկան արձակուրդը։" (Annual vacation)
3. "Որո՞նք են աշխատանքային ժամերի սահմանափակումները։" (Working hours limits)

---

## ⚙️ Configuration:

**Model:** meta-llama/Meta-Llama-3-8B-Instruct
**Retrieval:** BM25 (top-3 articles)
**Temperature:** 0.1 (deterministic)
**Max tokens:** 500

---

## 🔧 Troubleshooting:

### Out of Memory?
Reduce batch size or use smaller model in `scripts/test_rag.py`:
```python
model_name="google/flan-t5-xxl"  # Smaller model
```

### Model download slow?
First run downloads ~16GB model. Subsequent runs use cache.

### GPU not detected?
Check with:
```bash
nvidia-smi
echo $CUDA_VISIBLE_DEVICES
```

---

## 📊 Expected Output:

```
Loading model: meta-llama/Meta-Llama-3-8B-Instruct
✅ Model loaded successfully on cuda:0
   GPU Memory: 15.2 GB

QUESTION 1: Որո՞նք են նվազագույն աշխատավարձի կանոնները։
Retrieved Articles: [145, 146, 147]
Scores: [8.52, 4.21, 3.18]

ANSWER:
Համաձայն Հոդված 145-ի...
```

---

## 🎯 Next Steps:

1. Test with your own questions
2. Try Dense retrieval (semantic search)
3. Build Hybrid retriever (BM25 + Dense)
4. Create evaluation metrics
5. Build web interface (Streamlit/Gradio)
