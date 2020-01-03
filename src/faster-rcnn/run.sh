CUDA_VISIBLE_DEVICES=0,1 /home/duxiaowey/miniconda3/envs/myps/bin/python \
trainval_net.py --dataset voc_2013 --net res101  --lr 1e-3 \
--bs 12 --nw 4 --cuda --mGPUs \
--r True --checksession 1 --checkepoch 83 --checkpoint 613


# CUDA_VISIBLE_DEVICES=0,1 /home/duxiaowey/miniconda3/envs/myps/bin/python \
# trainval_net_n.py --dataset voc_2013 --net res101  --lr 1e-3 \
# --bs 12 --nw 4 --cuda --mGPUs \
# --r True --checksession 1 --checkepoch 8 --checkpoint 306