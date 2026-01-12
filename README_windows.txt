On Windows DeepSpeed requirement couldn't be installed by requirements.txt, it needs to be compiled.

As referenced in https://github.com/deepspeedai/DeepSpeed/issues/4729#issue-2010860892:

1) Download the latest release of DeepSpeed (compatible with Python 3.9 & CUDA 12.1. For Windows the 0.13.5 is the adviced one), extract it to a folder.
2) Install Visual C++ build tools, such as VS2019 C++ x64/x86 build tools.
3) Download and install the Nvidia Cuda Toolkit 12.1
4) Edit your Windows environment variables to ensure that CUDA_HOME and CUDA_PATH are set to your Nvidia Cuda Toolkit path. (The folder above the bin folder that nvcc.exe is installed in).

5) Navigate to your deepspeed folder in the Command Prompt: $ cd c:\deepspeed (wherever you extracted it to)
6) set DS_BUILD_AIO=0 
7) set DS_BUILD_SPARSE_ATTN=0
8) set DS_BUILD_EVOFORMER_ATTN=0
9) build_win.bat (This reqires admin privileges: it will start building your wheel file and may take a while)
10) Now cd dist to go into your dist folder and you can now pip install deepspeed-YOURFILENAME.whl (or whatever your WHL file is called).