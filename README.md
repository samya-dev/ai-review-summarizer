# 🧠 Sentify AI v2

> Transform raw customer feedback into structured strategic intelligence using Google Gemini AI.

Sentify AI v2 is an advanced customer sentiment analysis platform built with Streamlit, Gemini 2.5, Plotly, and Pydantic. It converts customer reviews, survey responses, support tickets, and social media feedback into actionable business insights through structured AI analysis.

---

## ✨ Features

### 📊 Structured AI Output
- Type-safe responses using Pydantic schemas
- Consistent JSON outputs
- Sentiment confidence scoring
- Executive summaries generated automatically

### 🔍 Smart Topic Extraction
- Detects trending customer topics
- Extracts key phrases automatically
- Identifies sentiment associated with each topic
- Estimates topic frequency across reviews

### 🚀 Token Optimization
- Automatic review cleaning
- Emoji removal
- Blank-line reduction
- Intelligent batching for large datasets

### 🎯 Industry-Specific Analysis
Choose analysis personas tailored for:

- E-Commerce & Retail
- SaaS & Technology
- Hospitality & Travel
- Healthcare & Wellness
- Financial Services
- Food & Beverage
- Education & E-Learning
- General Business

### 📈 Interactive Dashboard
- Sentiment donut charts
- KPI cards
- Topic tag clouds
- Strategic recommendations
- Executive reports

### 📄 Export Capabilities
- Corporate Markdown Reports
- Plain Text Reports
- Structured JSON data

---

## 🖼️ Dashboard Preview

### Sentiment Analytics
- Positive / Negative / Neutral distribution
- Confidence scoring
- Interactive visualization

### Strategic Intelligence Report
- Executive Summary
- Customer Compliments
- Customer Complaints
- Actionable Recommendations
- Trending Topics

---

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|----------|
| Streamlit | Web Application |
| Google Gemini 2.5 | AI Analysis Engine |
| Pydantic | Structured Output Validation |
| Plotly | Interactive Visualizations |
| Pandas | Data Processing |
| Python | Backend Logic |

---

## 📦 Installation

### Clone the Repository

```bash
git clone https://github.com/yourusername/sentify-ai.git
cd sentify-ai
```

### Create Virtual Environment

```bash
python -m venv venv
```

Activate:

**Windows**

```bash
venv\Scripts\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install streamlit google-genai plotly pandas pydantic python-dotenv
```

---

## 🔑 Configuration

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

Get your Gemini API key from:

https://aistudio.google.com/

---

## ▶️ Running the Application

```bash
streamlit run web_app.py
```

The application will start at:

```text
http://localhost:8501
```

---

## 📂 Supported Input Formats

### Text Input

Paste:

- Customer reviews
- Survey responses
- Support tickets
- Social media comments

### CSV Upload

Example:

```csv
review
Amazing service and fast delivery.
The app crashes frequently.
Customer support was very helpful.
```

Sentify AI automatically detects text columns.

---

## 🧠 AI Analysis Pipeline

```text
Raw Reviews
      ↓
Review Cleaning
      ↓
Batch Processing
      ↓
Sentiment Analysis
      ↓
Topic Extraction
      ↓
Strategic Recommendations
      ↓
Executive Report
```

---

## 📊 Output Example

### Sentiment Overview

```json
{
  "positive": 72.4,
  "negative": 15.6,
  "neutral": 12.0,
  "dominant": "positive",
  "confidence": 0.91
}
```

### Strategic Insights

```json
{
  "compliments": [
    "Fast delivery",
    "Excellent customer support"
  ],
  "complaints": [
    "Mobile app instability"
  ],
  "advice": [
    "Improve app reliability",
    "Expand support coverage"
  ]
}
```

---

## 🎯 Use Cases

### Business Intelligence
- Customer feedback analysis
- Brand monitoring
- Voice-of-customer programs

### Product Management
- Feature feedback analysis
- Product sentiment tracking
- User experience evaluation

### Customer Success
- Support ticket intelligence
- Customer satisfaction analysis
- Retention improvement strategies

### Marketing
- Campaign sentiment analysis
- Reputation management
- Audience insight generation

---

## 🔒 Privacy

- Customer data is not stored by the application.
- Reviews are processed only for analysis.
- API communication occurs directly with Google Gemini services.
- Reports are generated locally.

---

## 🚧 Future Roadmap

- Multi-file analysis
- PDF support
- PowerPoint export
- Comparative sentiment tracking
- Historical trend analysis
- Multi-model AI support
- Team collaboration features
- Real-time review monitoring

---

## 🤝 Contributing

Contributions are welcome.

1. Fork the repository
2. Create a feature branch

```bash
git checkout -b feature/new-feature
```

3. Commit your changes

```bash
git commit -m "Add new feature"
```

4. Push to GitHub

```bash
git push origin feature/new-feature
```

5. Open a Pull Request

---

## 📜 License

MIT License

---

## 👨‍💻 Author

Built with Python, Streamlit, and Google Gemini AI to help businesses turn customer feedback into strategic intelligence.
