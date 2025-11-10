python ./load_model/kfold_predict.py \
     --fold_dir ./train_kfold/lg_transformer_kfold_runs/sheet_HOV/fold_1 \
     --xlsx_train ./data/revision/descriptors_all.xlsx   \
     --sheet_train HOV   \
     --xlsx_new ./data/predict_data/new_test_descriptors.xlsx   \
     --sheet_new HOV   \
     --mapping ./descriptors_group/descriptorsMap/descriptorsMapping.json   \
     --device cuda \
     --out_csv ./predictions_with_HOV_fold1.csv
