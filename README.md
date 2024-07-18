# TheAgency

* https://docs.omniverse.nvidia.com/isaacsim/latest/installation/install_container.html

$ docker login nvcr.io
Username: $oauthtoken
Password: 

$ docker pull nvcr.io/nvidia/isaac-sim:4.0.0

Maybe need to install 'apt install ros-humble-vision-msgs'
If it is not found, follow https://stackoverflow.com/questions/72591758/unable-to-locate-ros-packages


## Running Python Code from External Editors
* https://docs.omniverse.nvidia.com/isaacsim/latest/advanced_tutorials/tutorial_advanced_code_editors.html
VScode extension - https://marketplace.visualstudio.com/items?itemName=NVIDIA.isaacsim-vscode-edition


## Verify no-cgroups Setting
Ensure that the no-cgroups setting in the /etc/nvidia-container-runtime/config.toml file is correctly set to false:

no-cgroups = false