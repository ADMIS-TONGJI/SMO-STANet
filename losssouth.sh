#CUDA_VISIBLE_DEVICES=0 nohup python -u main_informer.py --model LSTM --data_path final_new.csv --dimension 1 --enc_in 2304 > LSTM-south1.log 2>&1

#CUDA_VISIBLE_DEVICES=0 nohup python -u main_informer.py --model LSTM --data_path final_new.csv --dimension 1 --enc_in 2304 > LSTM-south2.log 2>&1

#CUDA_VISIBLE_DEVICES=0 nohup python -u main_informer.py --model LSTM --data_path final_new.csv --dimension 1 --enc_in 2304 > LSTM-south3.log 2>&1

#CUDA_VISIBLE_DEVICES=0 nohup python -u main_informer.py --model LSTM --data_path final_new.csv --dimension 1 --enc_in 2304 > LSTM-south4.log 2>&1

#CUDA_VISIBLE_DEVICES=0 nohup python -u main_informer.py --model LSTM --data_path final_new.csv --dimension 1 --enc_in 2304 > LSTM-south5.log 2>&1



#CUDA_VISIBLE_DEVICES=1 nohup python -u main_informer.py --data_path chlor-east.csv --dimension 1 --enc_in 2304 > eastonedimension1.log 2>&1

#CUDA_VISIBLE_DEVICES=1 nohup python -u main_informer.py --data_path chlor-east.csv --dimension 1 --enc_in 2304 > eastonedimension2.log 2>&1

#CUDA_VISIBLE_DEVICES=1 nohup python -u main_informer.py --data_path chlor-east.csv --dimension 1 --enc_in 2304 > eastonedimension3.log 2>&1

#CUDA_VISIBLE_DEVICES=1 nohup python -u main_informer.py --data_path chlor-east.csv --dimension 1 --enc_in 2304 > eastonedimension4.log 2>&1

#CUDA_VISIBLE_DEVICES=1 nohup python -u main_informer.py --data_path chlor-east.csv --dimension 1 --enc_in 2304 > eastonedimension5.log 2>&1


#南海Informer+loss

CUDA_VISIBLE_DEVICES=1 nohup python -u main_informer.py --conv1 True --conv2 True --theta 0.5 --data_path final_east.csv > southloss1.log 2>&1

CUDA_VISIBLE_DEVICES=1 nohup python -u main_informer.py --conv1 True --conv2 True --theta 0.5 --data_path final_east.csv > southloss2.log 2>&1

CUDA_VISIBLE_DEVICES=1 nohup python -u main_informer.py --conv1 True --conv2 True --theta 0.5 --data_path final_east.csv > southloss3.log 2>&1

CUDA_VISIBLE_DEVICES=1 nohup python -u main_informer.py --conv1 True --conv2 True --theta 0.5 --data_path final_east.csv > southloss4.log 2>&1

CUDA_VISIBLE_DEVICES=1 nohup python -u main_informer.py --conv1 True --conv2 True --theta 0.5 --data_path final_east.csv > southloss5.log 2>&1

CUDA_VISIBLE_DEVICES=1 nohup python -u main_informer.py --conv1 True --conv2 True --theta 0.5 --data_path final_east.csv > southloss6.log 2>&1


CUDA_VISIBLE_DEVICES=1 nohup python -u main_informer.py --conv1 True --conv2 True --theta 1.0 --data_path final_east.csv > southloss1.log 2>&1

CUDA_VISIBLE_DEVICES=1 nohup python -u main_informer.py --conv1 True --conv2 True --theta 1.0 --data_path final_east.csv > southloss1.log 2>&1

CUDA_VISIBLE_DEVICES=1 nohup python -u main_informer.py --conv1 True --conv2 True --theta 1.0 --data_path final_east.csv > southloss1.log 2>&1

CUDA_VISIBLE_DEVICES=1 nohup python -u main_informer.py --conv1 True --conv2 True --theta 1.0 --data_path final_east.csv > southloss1.log 2>&1

CUDA_VISIBLE_DEVICES=1 nohup python -u main_informer.py --conv1 True --conv2 True --theta 1.0 --data_path final_east.csv > southloss1.log 2>&1

CUDA_VISIBLE_DEVICES=1 nohup python -u main_informer.py --conv1 True --conv2 True --theta 1.0 --data_path final_east.csv > southloss1.log 2>&1


CUDA_VISIBLE_DEVICES=1 nohup python -u main_informer.py --conv1 True --conv2 True --theta 1.5 --data_path final_east.csv > southloss1.log 2>&1

CUDA_VISIBLE_DEVICES=1 nohup python -u main_informer.py --conv1 True --conv2 True --theta 1.5 --data_path final_east.csv > southloss2.log 2>&1

CUDA_VISIBLE_DEVICES=1 nohup python -u main_informer.py --conv1 True --conv2 True --theta 1.5 --data_path final_east.csv > southloss3.log 2>&1

CUDA_VISIBLE_DEVICES=1 nohup python -u main_informer.py --conv1 True --conv2 True --theta 1.5 --data_path final_east.csv > southloss4.log 2>&1

CUDA_VISIBLE_DEVICES=1 nohup python -u main_informer.py --conv1 True --conv2 True --theta 1.5 --data_path final_east.csv > southloss5.log 2>&1

CUDA_VISIBLE_DEVICES=1 nohup python -u main_informer.py --conv1 True --conv2 True --theta 1.5 --data_path final_east.csv > southloss5.log 2>&1

CUDA_VISIBLE_DEVICES=1 nohup python -u main_informer.py --conv1 True --conv2 True --theta 1.5 --data_path final_east.csv > southloss1.log 2>&1



CUDA_VISIBLE_DEVICES=2 nohup python -u main_informer.py --conv1 True --conv2 True --theta 2.0 --data_path final_new.csv > eastloss2.log 2>&1

CUDA_VISIBLE_DEVICES=2 nohup python -u main_informer.py --conv1 True --conv2 True --theta 2.0 --data_path final_new.csv > eastloss3.log 2>&1

CUDA_VISIBLE_DEVICES=2 nohup python -u main_informer.py --conv1 True --conv2 True --theta 2.0 --data_path final_new.csv > eastloss4.log 2>&1

CUDA_VISIBLE_DEVICES=2 nohup python -u main_informer.py --conv1 True --conv2 True --theta 2.0 --data_path final_new.csv > eastloss1.log 2>&1

CUDA_VISIBLE_DEVICES=2 nohup python -u main_informer.py --conv1 True --conv2 True --theta 2.0 --data_path final_new.csv > eastloss2.log 2>&1

CUDA_VISIBLE_DEVICES=2 nohup python -u main_informer.py --conv1 True --conv2 True --theta 2.0 --data_path final_new.csv > eastloss3.log 2>&1

CUDA_VISIBLE_DEVICES=2 nohup python -u main_informer.py --conv1 True --conv2 True --theta 2.0 --data_path final_new.csv > eastloss4.log 2>&1

CUDA_VISIBLE_DEVICES=2 nohup python -u main_informer.py --conv1 True --conv2 True --theta 1.0 --data_path final_new.csv > eastloss5.log 2>&1

CUDA_VISIBLE_DEVICES=2 nohup python -u main_informer.py --conv1 True --conv2 True --theta 1.0 --data_path final_new.csv > eastloss3.log 2>&1

CUDA_VISIBLE_DEVICES=2 nohup python -u main_informer.py --conv1 True --conv2 True --theta 1.0 --data_path final_new.csv > eastloss4.log 2>&1

CUDA_VISIBLE_DEVICES=2 nohup python -u main_informer.py --conv1 True --conv2 True --theta 1.0 --data_path final_new.csv > eastloss5.log 2>&1
