#!/bin/bash
#SBATCH --mail-user=ar.aamer@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL
#SBATCH --job-name=bn_train
#SBATCH --output=%x-%j.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=32
#SBATCH --mem=200000M
#SBATCH --time=0-05:00
#SBATCH --account=rrg-ebrahimi

nvidia-smi

source ~/hist_cbn/bin/activate

echo "------------------------------------< Data preparation>----------------------------------"
echo "Copying the source code"
date +"%T"
cd $SLURM_TMPDIR
cp -r ~/scratch/CBN_Camelyon .

echo "Copying the datasets"
date +"%T"
cd CBN_Camelyon
cd dataset
cp -r ~/scratch/FS_WSI_Datasets/camylon .
cd ..

echo "----------------------------------< End of data preparation>--------------------------------"
date +"%T"
echo "--------------------------------------------------------------------------------------------"

echo "---------------------------------------<Run the program>------------------------------------"
date +"%T"
cd $SLURM_TMPDIR

cd CBN_Camelyon

experiment = "BN"

python train_bn.py -exp_name $experiment -batch_size 32


echo "-----------------------------------<End of run the program>---------------------------------"
date +"%T"
echo "--------------------------------------<backup the result>-----------------------------------"
date +"%T"
cd $SLURM_TMPDIR
cp -r $SLURM_TMPDIR/CBN_Camelyon/logs/$experiment ~/scratch/CBN_Camelyon/logs
