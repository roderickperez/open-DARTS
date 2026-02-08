import os
import sys

def wheels_list(platform, dir='.'):
    '''
    returna a list of wheel (.whl) filenames for a scpecified platform (lin/win) located in a specified directory
    '''
    wheels_list = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file.endswith('.whl') and platform in file:
                wheels_list.append(file)
    return wheels_list

if __name__ == '__main__':
    wl = wheels_list(platform=sys.argv[1], dir=sys.argv[2])
    wl.sort()
    latest_wheel = wl[-1]
    print(latest_wheel)