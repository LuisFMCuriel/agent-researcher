# agent-researcher

python -m src.data.init_db
python -m src.eval.generate_synthetic_data
python -m src.rag.build_faiss --rebuild
python -m src.rag.rag_answer "What is the accuracy of exp_001?"
python -m src.eval.evaluate_rag