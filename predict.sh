python ./load_model/kfold_predict.py \
     --fold_dir ./train_kfold/lg_transformer_kfold_runs/sheet_EOV/fold_1 \
     --xlsx_train ./data/revision/descriptors_all.xlsx   \
     --sheet_train EOV   \
     --xlsx_new ./data/predict_data/new_test_descriptors.xlsx   \
     --sheet_new EOV   \
     --mapping ./descriptors_group/descriptorsMap/descriptorsMapping.json   \
     --device cuda \
     --out_csv ./predictions_with_EOV_fold1.csv
