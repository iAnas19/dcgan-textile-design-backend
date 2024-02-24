# Backend for Textile Design Generation App

## How to run

`uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload`

## Folder Structure

```
project_root/
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   └── models/
│       ├── __init__.py
│       └── dcgan.py
│
├── nets/
│   ├── __init__.py
│   └── netG.pth
│   └── others
│
├── readme.md
├── requirements.txt
└── uvicorn_run.sh
```
