export CUDA_VISIBLE_DEVICES=0
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps0
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-log0
sudo nvidia-smi -i 0 -c EXCLUSIVE_PROCESS
nvidia-cuda-mps-control -d
unset CUDA_VISIBLE_DEVICES

python run.py
