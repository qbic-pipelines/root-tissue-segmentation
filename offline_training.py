import os

import sys


def offline_trainer(script) -> str:
    """
    Runs Training offline on VM.
    :param script: Python to train offline on VM.
    :return: Message that script is running on VM.
    """
    systemvar = "nohup sh -c '"
    sysvars = sys.executable + f' {os.getcwd()}/{script} '
    systemvar += f"{sysvars} > log.txt 2>&1' &"
    os.system(systemvar)
    print(systemvar)
    out = "Offline training started view process with ps -ef |grep python."
    return out


def main():
    print(offline_trainer("hpo.py --n-trials 300 --n-startup-trials 10"))


if __name__ == '__main__':
    main()
