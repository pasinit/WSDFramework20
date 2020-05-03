for lang in it es fr de
do
	echo $lang
	PYTHONPATH=. python src/evaluation/evaluate_model.py --config config/config_${lang}_onesec.yaml --checkpoint_path data/models/${lang}_onesec_new_evaluation/bert-base-multilingual-cased/checkpoints/model_state_epoch_4.th --output_path data/models/${lang}_onesec_new_evaluation/bert-base-multilingual-cased/evaluation/ --debug
done
