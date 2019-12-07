echo "BERT-LARGE SEMCOR+GLOSSE WN ID TRAINING"
python src/test/evaluate_model.py --config config/config_en_semcor+glosse_manual.wnoffsets.bert-large.yaml --checkpoint data/models/en_semcor_gloss_manual_wnoffsets_training/bert-large-cased/checkpoints/model_state_epoch_8.th --output_path data/models/en_semcor_gloss_manual_wnoffsets_training/bert-large-cased/evaluation

echo "BERT-BASE SEMCOR+GLOSSE WN ID TRAINING"
python src/test/evaluate_model.py --config config/config_en_semcor+glosse_manual.wnoffsets.bert-base.yaml --checkpoint data/models/en_semcor_gloss_manual_wnoffsets_training/bert-base-cased/checkpoints/model_state_epoch_23.th --output_path data/models/en_semcor_gloss_manual_wnoffsets_training/bert-base-cased/evaluation


echo "BERT-LARGE SEMCOR WN ID TRAINING"
python src/test/evaluate_model.py --config config/config_en_semcor_wnoffsets_training.yaml --checkpoint data/models/en_semcor_wnoffsets_training/bert-large-cased/checkpoints/model_state_epoch_9.th --output_path data/models/en_semcor_wnoffsets_training/bert-large-cased/evaluation

