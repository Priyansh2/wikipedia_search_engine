#!/bin/bash
#SBATCH -A research
#SBATCH -n 5
#SBATCH --qos=medium
#SBATCH --gres=gpu:0
#SBATCH -p long
#SBATCH --mem-per-cpu=3G
#SBATCH --time=4-00:00:00

export DATA_DUMP="enwiki-latest.xml" ##path of data
export PY_FILE="/home/priyansh.agrawal/ire/index.py"
mkdir -p /scratch/$USER/
if [ -f /scratch/$USER/$DATA_DUMP ]; then
echo "$DATA_DUMP exist"
else
rsync -avzP ada:/share1/$USER/$DATA_DUMP /scratch/$USER/
fi
rsync -avzP ada:$PY_FILE /scratch/$USER/
cd /scratch/$USER/
source /home/$USER/p3.6/bin/activate
pwd
python index.py $DATA_DUMP index1
rsync -avzP /scratch/$USER/index1 ada:/share1/$USER/index

