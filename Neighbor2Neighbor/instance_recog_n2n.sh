## add noise and convert noisy image to denoised dataset in the form (roxford5k|rparis6k)_(noisetype)
python eval_parisoxford.py --state_dict=TRAINED_WEIGHTS_PATH --noise=gauss_50 --test_dataset=roxford5k --data_root=OXFORDPARIS_DATA_PATH
python eval_parisoxford.py --state_dict=TRAINED_WEIGHTS_PATH --noise=gauss_100 --test_dataset=roxford5k --data_root=OXFORDPARIS_DATA_PATH
python eval_parisoxford.py --state_dict=TRAINED_WEIGHTS_PATH --noise=gauss_255 --test_dataset=roxford5k --data_root=OXFORDPARIS_DATA_PATH
python eval_parisoxford.py --state_dict=TRAINED_WEIGHTS_PATH --noise=speckle_0.4 --test_dataset=roxford5k --data_root=OXFORDPARIS_DATA_PATH
python eval_parisoxford.py --state_dict=TRAINED_WEIGHTS_PATH --noise=speckle_0.7 --test_dataset=roxford5k --data_root=OXFORDPARIS_DATA_PATH
python eval_parisoxford.py --state_dict=TRAINED_WEIGHTS_PATH --noise=speckle_1.0 --test_dataset=roxford5k --data_root=OXFORDPARIS_DATA_PATH
python eval_parisoxford.py --state_dict=TRAINED_WEIGHTS_PATH --noise=shot_10 --test_dataset=roxford5k --data_root=OXFORDPARIS_DATA_PATH
python eval_parisoxford.py --state_dict=TRAINED_WEIGHTS_PATH --noise=shot_3 --test_dataset=roxford5k --data_root=OXFORDPARIS_DATA_PATH
python eval_parisoxford.py --state_dict=TRAINED_WEIGHTS_PATH --noise=shot_1 --test_dataset=roxford5k --data_root=OXFORDPARIS_DATA_PATH

python eval_parisoxford.py --state_dict=TRAINED_WEIGHTS_PATH --noise=gauss_50 --test_dataset=rparis6k --data_root=OXFORDPARIS_DATA_PATH
python eval_parisoxford.py --state_dict=TRAINED_WEIGHTS_PATH --noise=gauss_100 --test_dataset=rparis6k --data_root=OXFORDPARIS_DATA_PATH
python eval_parisoxford.py --state_dict=TRAINED_WEIGHTS_PATH --noise=gauss_255 --test_dataset=rparis6k --data_root=OXFORDPARIS_DATA_PATH
python eval_parisoxford.py --state_dict=TRAINED_WEIGHTS_PATH --noise=speckle_0.4 --test_dataset=rparis6k --data_root=OXFORDPARIS_DATA_PATH
python eval_parisoxford.py --state_dict=TRAINED_WEIGHTS_PATH --noise=speckle_0.7 --test_dataset=rparis6k --data_root=OXFORDPARIS_DATA_PATH
python eval_parisoxford.py --state_dict=TRAINED_WEIGHTS_PATH --noise=speckle_1.0 --test_dataset=rparis6k --data_root=OXFORDPARIS_DATA_PATH
python eval_parisoxford.py --state_dict=TRAINED_WEIGHTS_PATH --noise=shot_10 --test_dataset=rparis6k --data_root=OXFORDPARIS_DATA_PATH
python eval_parisoxford.py --state_dict=TRAINED_WEIGHTS_PATH --noise=shot_3 --test_dataset=rparis6k --data_root=OXFORDPARIS_DATA_PATH
python eval_parisoxford.py --state_dict=TRAINED_WEIGHTS_PATH --noise=shot_1 --test_dataset=rparis6k --data_root=OXFORDPARIS_DATA_PATH