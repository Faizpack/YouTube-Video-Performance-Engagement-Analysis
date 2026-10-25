# YouTube Video Performance & Engagement Analysis

Files
- `YouTube_Analysis_Full.ipynb` - full Colab notebook
- `app.py` - Streamlit dashboard
- `requirements.txt` - dependencies for deployment
- Put your own `youtube_videos_dataset.csv` next to `app.py` 

Run locally
    pip install -r requirements.txt
    streamlit run app.py

Deploy free on Streamlit Community Cloud
1. Create a GitHub repo and upload app.py, requirements.txt and youtube_videos_dataset.csv
2. Go to https://share.streamlit.io and sign in with GitHub
3. New app -> choose the repo, branch main, main file path app.py -> Deploy
