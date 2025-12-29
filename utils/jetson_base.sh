#!/bin/bash


echo "Verify apt sources"
if command -v apt >/dev/null; then
    if [ ! -d "/etc/apt" ]; then
        mkdir -p /etc/apt/sources.list.d
    fi
    if [ ! -f "/etc/apt/sources.list" ]; then
        echo "deb http://ports.ubuntu.com/ubuntu-ports $(grep UBUNTU_CODENAME /etc/os-release | cut -d= -f2) main universe" > /etc/apt/sources.list
    fi
fi   

echo "Update and Upgrade"
apt update && apt upgrade -y

echo "Install networking stuff"
apt install iputils-ping
apt install inetutils-traceroute

echo "Install python dev stuff"
apt install python3-venv

