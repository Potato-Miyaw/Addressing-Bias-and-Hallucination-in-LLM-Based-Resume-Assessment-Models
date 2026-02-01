# Addressing Bias and Hallucination in LLM-Based Resume Assessment Models

A comprehensive AI-powered recruitment system that addresses bias detection and hallucination verification in resume assessment using multiple Large Language Models (LLMs) and machine learning techniques.

## Overview

This project implements an end-to-end recruitment pipeline that combines natural language processing, bias detection, claim verification, and fair ranking algorithms to create a more equitable and accurate hiring process. The system evaluates resumes against job descriptions while actively monitoring and mitigating potential biases across demographic groups.

## Key Features

### 1. Job Description Analysis
- Automated extraction of key requirements, skills, and qualifications
- NLP-based parsing of job descriptions
- Structured data extraction for matching purposes

### 2. Resume Processing
- Multi-format document parsing (PDF, DOCX, DOC)
- Named Entity Recognition (NER) for resume information extraction
- Hybrid NER approach combining BERT-based and custom models
- Support for multiple resume formats and structures

### 3. Claim Verification
- Ground truth extraction from resumes
- BERTScore-based claim verification
- Detection and flagging of potential hallucinations or misrepresentations
- Verification confidence scoring

### 4. Intelligent Matching
- Semantic similarity matching between resumes and job descriptions
- TF-IDF and cosine similarity-based ranking
- Multi-factor scoring system

### 5. Fair Ranking with XGBoost
- XGBoost-based candidate ranking
- Fairness-aware machine learning using Fairlearn
- Demographic parity enforcement
- Bias mitigation through fairness constraints

### 6. Multi-Model Bias Detection
- Systematic evaluation of multiple LLMs (Gemma, Qwen, TinyLlama)
- Controlled demographic variation testing (8 demographic groups)
- Resume audit study methodology
- Statistical analysis of bias patterns
- Comprehensive bias reporting and metrics

### 7. Interactive Questionnaire System
- Dynamic questionnaire generation based on job requirements
- Candidate response collection and analysis
- Sentiment analysis of responses
- WhatsApp and email notification support
- Hiring prediction based on questionnaire responses

### 8. Multi-language Support
- Translation service for non-English resumes
- Language detection and automatic translation
- Support for diverse candidate pools

### 9. Feedback and Continuous Improvement
- HR feedback collection system
- Model performance tracking
- Iterative improvement based on real-world usage

## Technology Stack

### Backend
- **Framework**: FastAPI
- **Database**: MongoDB with Motor (async driver)
- **Machine Learning**: 
  - PyTorch
  - Transformers (HuggingFace)
  - XGBoost
  - scikit-learn
  - Fairlearn (bias mitigation)
  - AIF360 (fairness metrics)
- **NLP**: 
  - spaCy
  - NLTK
  - BERT-Score
  - KeyBERT
  - Sentence Transformers
- **Document Processing**: 
  - PyMuPDF
  - python-docx
  - pywin32

### Frontend
- **Framework**: Streamlit
- **Visualization**: Plotly, Matplotlib
- **Data Handling**: Pandas, NumPy

### Additional Services
- Translation: deep-translator
- Communication: Email (SMTP), WhatsApp integration
- Notifications: Microsoft Teams, Power Automate

## Project Structure

```
.
├── backend/
│   ├── app.py                 # Main FastAPI application
│   ├── database.py            # MongoDB connection and operations
│   ├── models.py              # Data models
│   ├── routers/               # API route handlers
│   │   ├── jd_router.py       # Job description endpoints
│   │   ├── resume_router.py   # Resume processing endpoints
│   │   ├── verification_router.py
│   │   ├── matching_router.py
│   │   ├── ranking_router.py
│   │   ├── bias_router.py     # Bias detection endpoints
│   │   ├── questionnaire_router.py
│   │   ├── feedback_router.py
│   │   └── ...
│   ├── services/              # Core business logic
│   │   ├── feature1_jd_extractor.py
│   │   ├── feature2_bert_ner.py
│   │   ├── feature2_hybrid_ner.py
│   │   ├── feature3_claim_verifier.py
│   │   ├── feature4_matcher.py
│   │   ├── feature5_xgb_ranker.py
│   │   ├── feature6_bias_detector.py
│   │   ├── question_generator.py
│   │   ├── answer_analyzer.py
│   │   ├── translation_service.py
│   │   └── ...
│   └── utils/                 # Utility functions
│       ├── document_parser.py
│       ├── file_extractor.py
│       └── id_generator.py
├── frontend/
│   ├── streamlit_app.py       # Main landing page
│   ├── pages/                 # Streamlit pages
│   │   ├── HR_Portal.py
│   │   ├── Candidate_Portal.py
│   │   └── Questionnaire_Response.py
│   └── tabs/                  # UI components
│       ├── job_description_tab.py
│       ├── resumes_tab.py
│       ├── matching_tab.py
│       ├── verification_tab.py
│       ├── bias_tab.py
│       └── ...
├── notebooks/
│   └── Train_Hiring_Predictor.ipynb
├── data/
│   └── feedback/              # Feedback data storage
├── requirements.txt           # Python dependencies
└── README.md
```

## Installation

### Prerequisites
- Python 3.8 or higher
- MongoDB instance (local or cloud)
- Windows OS (for full DOC file support via pywin32)

### Setup Steps

1. Clone the repository:
```bash
git clone https://github.com/Potato-Miyaw/Addressing-Bias-and-Hallucination-in-LLM-Based-Resume-Assessment-Models.git
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download required spaCy model:
```bash
python -m spacy download en_core_web_sm
```

5. Set up environment variables:
Create a `.env` file in the project root with the following variables:
```
MONGODB_URL=your_mongodb_connection_string
DATABASE_NAME=your_database_name
API_URL=your_api_url
STREAMLIT_URL=your_streamlit_url
POWER_AUTOMATE_URL=your_power_automate_url
```

## Running the Application

### Start the Backend (FastAPI)
```bash
cd backend
uvicorn app:app --reload --port 8000
```

The API will be available at `http://localhost:8000`
API documentation: `http://localhost:8000/docs`

### Start the Frontend (Streamlit)
```bash
cd frontend
streamlit run streamlit_app.py
```

The application will open in your browser at `http://localhost:8501`

## Usage

### For HR Users

1. **Upload Job Description**: Navigate to the Job Description tab and upload or paste the JD
2. **Upload Resumes**: Upload candidate resumes in PDF or DOCX format
3. **Run Matching**: Execute the matching algorithm to compare candidates against the JD
4. **Verify Claims**: Check for potential hallucinations or misrepresentations in resumes
5. **Review Rankings**: View fair rankings with bias mitigation applied
6. **Detect Bias**: Run multi-model bias analysis to ensure fairness
7. **Generate Questionnaire**: Create custom questionnaires for shortlisted candidates
8. **Review Feedback**: Track system performance and candidate responses

### For Candidates

1. **Access Candidate Portal**: Use the provided link to access your portal
2. **Submit Information**: Upload your resume and provide additional details
3. **Complete Questionnaire**: Answer generated questions (if invited)
4. **Receive Updates**: Get notifications via email or WhatsApp

## API Endpoints

### Job Descriptions
- `POST /jd/extract` - Extract information from job description
- `GET /jd/list` - List all job descriptions
- `GET /jd/{jd_id}` - Get specific job description

### Resumes
- `POST /resume/parse` - Parse and extract resume information
- `GET /resume/list` - List all resumes
- `GET /resume/{resume_id}` - Get specific resume

### Verification
- `POST /verification/verify` - Verify claims in resume
- `GET /verification/{resume_id}` - Get verification results

### Matching
- `POST /matching/match` - Match resumes against job description
- `GET /matching/results/{jd_id}` - Get matching results

### Ranking
- `POST /ranking/rank` - Rank candidates with fairness constraints
- `GET /ranking/results/{jd_id}` - Get ranking results

### Bias Detection
- `POST /bias/analyze` - Run multi-model bias analysis
- `GET /bias/report/{session_id}` - Get bias analysis report

### Questionnaire
- `POST /questionnaire/generate` - Generate questionnaire
- `POST /questionnaire/submit` - Submit responses
- `GET /questionnaire/{questionnaire_id}` - Get questionnaire details

## Fairness and Bias Mitigation

The system implements multiple layers of bias detection and mitigation:

1. **Demographic Parity**: Ensures equal representation across demographic groups
2. **Multi-Model Testing**: Evaluates multiple LLMs for systematic bias patterns
3. **Controlled Experiments**: Tests with randomized demographic variations
4. **Statistical Analysis**: Provides comprehensive bias metrics and reports
5. **Fairness Constraints**: Applies fairness-aware learning in ranking algorithms

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the terms specified in the LICENSE file.

## Acknowledgments

- Built as part of the EPITA Action Learning program 2026
- Special thanks to all contributors and researchers in fairness in AI
- Powered by HuggingFace Transformers, Fairlearn, and open-source ML community

## Contact

For questions or support, please open an issue in the GitHub repository or contact the maintainer at rayanalam5392@gmail.com

