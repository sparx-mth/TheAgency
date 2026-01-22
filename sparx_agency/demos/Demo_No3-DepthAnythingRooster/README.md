docker run --runtime nvidia -it     --network host     --privileged     --env ROS_DOMAIN_ID=2     --volume /home/user/GIT/TheAgency/sparx_agency:/home/user/GIT/TheAgency/sparx_agency     --volume /home/user/rqs_iai_ws:/home/user/rqs_iai_ws     --volume /etc/cyclonedds.xml:/etc/cyclonedds.xml     rooster_mapping_jetson:depth_anything


docker run --runtime nvidia -it --network host --privileged --name jetson_depth_anything -p 2222:22 --env ROS_DOMAIN_ID=2 rooster_mapping_jetson:depth_anything 
