# If you need to load modules, these are the versions used:
module load python/3.11
module load gcc arrow/21.0.0 cuda 
module load scipy-stack opencv

python -m venv vlevalenv
source vlevalenv/bin/activate

pip install torch==2.7.1 torchvision==0.22.1 transformers==4.52.3 evaluate datasets
pip install flash_attn
pip install numpy triton
pip install tqdm imageio easydict ftfy imageio-ffmpeg decord einops pycocotools timm peft onnxruntime scikit-image