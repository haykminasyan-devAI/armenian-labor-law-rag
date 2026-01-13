# How to Run the Chatbot Interface

## ✅ What's Available:

1. **CLI Chatbot** - Simple command-line interface
2. **Streamlit Web App** - Beautiful web interface

---

## 🖥️ Option 1: CLI Chatbot (Simple)

### Run with BM25:
```bash
cd /home/hayk.minasyan/Project/NLP_proj
source venv/bin/activate
python interface/chatbot.py
```

### Run with Dense:
```bash
python interface/chatbot.py --method dense
```

### Usage:
```
💬 Ձեր հարցը: Քանի՞ արձակուրդային օր կա։

🔍 Փնտրում եմ համապատասխան հոդվածներ...
📊 Գտնված հոդվածներ: [160, 159, 158]
📊 Վստահության միավորներ: [0.70, 0.70, 0.67]

💡 ՊԱՏԱՍԽԱՆ:
Հոդված 159-ի 1-ին մասի համաձայն...
```

---

## 🌐 Option 2: Streamlit Web App (Beautiful UI)

### Run:
```bash
cd /home/hayk.minasyan/Project/NLP_proj
source venv/bin/activate
streamlit run interface/streamlit_app.py
```

### Access:
```
Local URL: http://localhost:8501
Network URL: http://your-ip:8501
```

### Features:
- ✅ Beautiful web interface
- ✅ Dropdown to switch between BM25/Dense
- ✅ Slider to adjust number of articles
- ✅ Shows retrieved articles with scores
- ✅ Expandable context view
- ✅ Example questions

---

## 🎯 Example Questions:

1. Որո՞նք են նվազագույն աշխատավարձի կանոնները։
2. Քանի՞ արձակուրդային օր կա։
3. Ինչպե՞ս է սահմանվում գործուղման օրապահիկը։
4. Ի՞նչ իրավունքներ ունի աշխատողը երբ իրեն կրճատում են։
5. Ինչ է կարգավորում Աշխատանքային օրենսգրքի 1-ին հոդվածը։

---

## 🔧 Requirements:

Both interfaces require:
- ✅ Virtual environment activated
- ✅ NVIDIA API key (already configured)
- ✅ BM25 or Dense index built
- ✅ Internet connection (for NVIDIA API)

---

## 💡 Tips:

- Use **BM25** for specific article queries
- Use **Dense** for conceptual/semantic questions
- CLI is faster and simpler
- Streamlit is better for demos and presentations

Enjoy your chatbot! 🚀
