# ADAGAN
# 0. Dependencies
Create a conda environment with python version = 3.6
pytorch
# 1. dataset
Mayo dataset：[Mayo dataset](http://www.aapm.org/GrandChallenge/LowDoseCT/)
piglet：http://homepage.usask.ca/protect%20$elax%20sim%20$xiy525/
download mayo dataset from the link //www.aapm.org/GrandChallenge/LowDoseCT/
convert .dcm to .png 
# 2. Training
python train.py --dataset facades --cuda ###--dataset need to be modified.
# 3. Evaluation
python test.py --dataset facades --cuda ###--dataset need to be modified.
# 4. Reference
