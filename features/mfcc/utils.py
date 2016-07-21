import subprocess

shell = lambda command: subprocess.Popen(command, shell=True, stdout=subprocess.PIPE).communicate()[0]
