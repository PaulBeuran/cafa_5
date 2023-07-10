# cafa_5

## How to use

- Install Visual Studio Code
- If on Windows, install WSL and Ubuntu on WSL as instructed: https://learn.microsoft.com/en-us/windows/wsl/install
- Install WSL extension for Visual Studio Code
- Launch an Ubuntu terminal (either from VS code, wsl.exe or Ubuntu.exe)
- Install CUDA support for WSL from here: https://docs.nvidia.com/cuda/wsl-user-guide/index.html#cuda-support-for-wsl-2
- Install Anaconda package manager as instructed: https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html
- Create package environment `cafa_5` in Anaconda as such:
```
conda create --name cafa_5
conda activate cafa_5
```
- Install followings packages as such:
```
conda install numpy scipy pandas scikit-learn matplotlib seaborn
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install fasta
```
- Clone the repo and move inside (considered as root)
- Dowload data from Kaggle here: https://www.kaggle.com/competitions/cafa-5-protein-function-prediction/data
- Unzip the data into `./kaggle/input/`
- Enjoy !
