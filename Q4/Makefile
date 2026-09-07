install:
	python -m pip install -r requirements.txt
prepare:
	python prepare_models.py
test:
	python -m pytest -q
demo:
	python main.py demo --seed 42
benchmark:
	python main.py benchmark
streamlit:
	streamlit run app.py
submission-runs:
	python run_submission.py
