# python main.py --file_type="smartphone" --model_mode="bert" --program_mode="run" --stage_model="first" --epoch=30 --model_type="multitask" --embed_dropout=0.1

# python main.py --file_type="smartphone" --model_mode="bert" --program_mode="run" --stage_model="second" --epoch=30 --model_type="multitask" --embed_dropout=0.1 --factor=0.3

python main.py --file_type="smartphone" --model_mode="bert" --program_mode="test" --epoch=30 --model_type="multitask"