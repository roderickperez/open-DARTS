# Installation

## Where to get the latest DARTS wheels

If you are still lucky to have NetID, then while connected to TU Delft network you can map new Windows drive:
~~~
net use Y: \\darts-ci.citg.tudelft.nl\darts-private-artifacts ILoveDARTS2019! /user:WORKGROUP\darts-dev
~~~
If you are outside of the campus, just use VPN (https://vpn.tudelft.nl). In Linux, you can use command-line tools like `smbclient` or your file manager to connect to the samba share. As you might have already guessed, the address is `darts-ci.citg.tudelft.nl\darts-private-artifacts`, the username is `WORKGROUP\darts-dev` and the password is `ILoveDARTS2019!!`. 

This network share provides the latest built wheels, as well as separate libraries - intermediate results of DARTS build process.


## Where to get the docker image

The Docker imaged is located in `\\darts-ci.citg.tudelft.nl\darts-private-artifacts\docker` . Look [here](Where to get the latest DARTS wheels) how to get there.

## How to deploy DARTS via Python wheels

DARTS is distributed as Python wheel package, compiled for win_amd64 and linux_86_64 platforms.
Python 3.6, 3.7, 3.8 and 3.9 are currently supported.

To install DARTS, you need a regular 64-bit Python distribution with pip package manager. Follow https://pip.pypa.io/en/stable/installing/ if you don`t have pip.

If you don`t have python, get one! Look for a 64-bit Python distribution from https://www.python.org/downloads/. If you install for windows, tick the option "Add Python 3.X to PATH" to be able to run interpreter (and pip) from any location using plain 'python' or 'pip' commands.

Now, get the wheel according to your Python version and platform. For example, filename darts-0.1.0-cp36-cp36m-linux_x86_64.whl corresponds to Python 3.6 on Linux platform.

The following command installs DARTS to a system directory of Python 3.5 on Linux platform:

	> pip install darts-0.1.0-cp36-cp36m-linux_x86_64.whl

Alternatively, you can call python binary if the location of pip is unknown:
	
	> python -m pip install darts-0.1.0-cp36-cp36m-linux_x86_64.whl

To upgrade DARTS, use:

	> pip install --force-reinstall --no-deps darts-0.1.0-cp36-cp36m-linux_x86_64.whl

To install DARTS into a user directory (e.g., in case you don`t have root rights on a system), use:

	> pip install --user darts-0.1.0-cp36-cp36m-linux_x86_64.whl

pip will install most of the requirements for DARTS automatically, in case you don`t have them already.
After the installation is complete, you should be able to run DARTS! Start python interpreter and enter:

    >> import darts.engines 

No output would mean correct installation.

## How to deploy DARTS via Docker on Windows

Prerequisites:
1. [Docker for Windows](https://docs.docker.com/docker-for-windows/install/)
   Note: if you are unable to run Docker after installation and reboot, try to run it under a local Administrator account. 
2. X-server ([VcXsrv](https://sourceforge.net/projects/vcxsrv/), for example. When installed, use provided config.xlaunch to start the VcXsrv X server).

Install docker image with DARTS:
1. Launch Docker
2. Download all files from `\\darts-ci.citg.tudelft.nl\darts-private-artifacts\docker` to a local directory. Look [here](Where to get the latest DARTS wheels) how to get them.
3. Open command prompt and go to that folder
4. Make sure Docker has already reached 'running' state. Import darts-training-docker-image.tar to docker by:
`docker load -i darts-training-docker-image.tar`
5. Once the process is completed, it is done! You only need to install the Docker image once. 

Start docker container:
1. Launch Xserver (you can start VcXsrv with double-click on the provided config.xlaunch file) - an icon should appear in the system tray.
2. Launch Docker, if it's not yet running.
3. Open command prompt and run the image using `docker run -e DISPLAY=host.docker.internal:0 --rm darts/training:v1`
4. If everything is nominal, you should see a Pycharm window with darts models

## How to deploy DARTS via Docker on MacOS

Prerequisites:
1. [Docker for Mac](https://docs.docker.com/docker-for-mac/install/)
2. X-server ([XQuartz](https://www.xquartz.org/index.html). When installed, run it, and activate the option ‘Allow connections from network clients’ in XQuartz settings. Restart XQuartz to enable the new setting.
Install docker image with DARTS:
1. Launch Docker
2. Download all files from `\\darts-ci.citg.tudelft.nl\darts-private-artifacts\docker` to a local directory. Look [here](Where to get the latest DARTS wheels) how to get them.
3. Open command prompt and go to that folder
4. Make sure Docker has already reached 'running' state. Import darts-training-docker-image.tar to docker by:
`docker load -i darts-training-docker-image.tar`
5. Once the process is completed, it is done! You only need to install the Docker image once. 

Start docker container:
1. Launch XQuartz.
2. Launch Docker, if it's not yet running.
3. Open command prompt and type `xhost + 127.0.0.1` to allow access from localhost.
4. Finally, run the image using `docker run -e DISPLAY=host.docker.internal:0 --rm darts/training:v1`. If everything is nominal, you should see a Pycharm window with darts models