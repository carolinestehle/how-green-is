import subprocess
import sys


def __isConda()-> bool:
    is_conda = False
    try:
        envs = subprocess.check_output('conda env list')
        if "conda" in str(envs):
            is_conda = True
    except Exception as excp:
        print("No conda")
    return is_conda



def installModule(package):
    
    packageManager = "pip"

    if __isConda() == True:
        packageManager = "conda"

    subprocess.check_call([sys.executable, "-m", packageManager, "install", package])

     
 

    