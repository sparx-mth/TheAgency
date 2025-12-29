#!/usr/bin/env bash
set -euo pipefail

if [[ $EUID -eq 0 ]]; then
  echo "Please run this script as a normal user (without sudo)."
  echo "The script will use sudo internally when needed."
  exit 1
fi

INSTALL_ROS2=false
INSTALL_PYCHARM=false

usage() {
  cat <<EOF
Usage: $0 [OPTIONS]

Options:
  --ros2         Install ROS 2 Jazzy and its prerequisites
  --pycharm      Install PyCharm Community (via snap)
  --no-ros2      Explicitly disable ROS 2 install (overrides previous flags)
  --no-pycharm   Explicitly disable PyCharm install
  -h, --help     Show this help

Examples:
  $0                       # base setup (no ROS2, no PyCharm)
  $0 --ros2               # base + ROS2
  $0 --pycharm            # base + PyCharm
  $0 --ros2 --pycharm     # base + ROS2 + PyCharm
EOF
}

# Simple flag parsing
while [[ $# -gt 0 ]]; do
  case "$1" in
    --ros2)
      INSTALL_ROS2=true
      ;;
    --pycharm)
      INSTALL_PYCHARM=true
      ;;
    --no-ros2)
      INSTALL_ROS2=false
      ;;
    --no-pycharm)
      INSTALL_PYCHARM=false
      ;;
    --no-docker)
      INSTALL_DOCKER=false
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      usage
      exit 1
      ;;
  esac
  shift
done

log() {
  echo
  echo "=== $* ==="
}

log "Updating apt index"
sudo apt update

#######################################
# Terminator
#######################################
log "Installing Terminator"
sudo apt install -y terminator

#######################################
# SSH & Network Tools
#######################################
log "Installing SSH & network tools"
sudo apt install -y \
  openssh-client \
  openssh-server \
  iputils-ping \
  traceroute \
  net-tools \
  dnsutils
sudo systemctl enable --now ssh || true

#######################################
# Core Dev Tools
#######################################
log "Installing core dev tools"
sudo apt install -y \
  build-essential \
  curl \
  wget \
  ca-certificates \
  gnupg \
  lsb-release \
  htop \
  tree \
  zip \
  unzip

#######################################
# Text editors: nano, gedit, vim
#######################################
log "Installing text editors (nano, gedit, vim)"
sudo apt install -y \
  nano \
  gedit \
  vim

#######################################
# Docker (official repository)
#######################################
if [[ "$INSTALL_DOCKER" == true ]]; then
	log "Installing Docker prerequisites"
	sudo apt install -y ca-certificates curl gnupg

	log "Configuring Docker APT repository"
	sudo install -m 0755 -d /etc/apt/keyrings
	if [[ ! -f /etc/apt/keyrings/docker.asc ]]; then
	  sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
	  sudo chmod a+r /etc/apt/keyrings/docker.asc
	fi

	if [[ ! -f /etc/apt/sources.list.d/docker.list ]]; then
	  . /etc/os-release
	  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu ${UBUNTU_CODENAME:-$VERSION_CODENAME} stable" | \
	    sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
	fi

	log "Updating apt index (with Docker repo)"
	sudo apt update

	log "Installing Docker Engine + CLI + Buildx + Compose"
	sudo apt install -y \
	  docker-ce \
	  docker-ce-cli \
	  containerd.io \
	  docker-buildx-plugin \
	  docker-compose-plugin

	log "Enabling Docker service and configuring 'docker' group"
	sudo systemctl enable --now docker || true
	if ! getent group docker > /dev/null; then
	  sudo groupadd docker
	fi
	sudo usermod -aG docker "$USER"
	echo "NOTE: you must log out and log back in (or reboot) so 'docker' group takes effect."
fi
#######################################
# Ensure curl + wget (redundant-safe)
#######################################
log "Re-ensuring curl + wget are installed"
sudo apt install -y curl wget

#######################################
# GStreamer stack
#######################################
log "Installing GStreamer stack"
sudo apt install -y \
  gstreamer1.0-tools \
  gstreamer1.0-plugins-base \
  gstreamer1.0-plugins-good \
  gstreamer1.0-plugins-bad \
  gstreamer1.0-plugins-ugly \
  gstreamer1.0-libav \
  libgstreamer1.0-dev \
  libgstreamer-plugins-base1.0-dev

#######################################
# Video tools: VLC + ffmpeg + SimpleScreenRecorder
#######################################
log "Installing video tools (VLC, ffmpeg, SimpleScreenRecorder)"
sudo apt install -y \
  vlc \
  ffmpeg \
  simplescreenrecorder

#######################################
# VS Code
#######################################
log "Installing VS Code (snap)"
if ! command -v snap &>/dev/null; then
  log "snap not found, installing snapd"
  sudo apt install -y snapd
fi

if ! snap list 2>/dev/null | grep -q "^code "; then
  sudo snap install code --classic
else
  echo "VS Code already installed via snap."
fi

#######################################
# PyCharm (Community) - optional
#######################################
if [[ "$INSTALL_PYCHARM" == true ]]; then
  log "Installing PyCharm Community (snap)"
  if ! snap list 2>/dev/null | grep -q "^pycharm-community "; then
    sudo snap install pycharm-community --classic
  else
    echo "PyCharm Community already installed via snap."
  fi
else
  log "Skipping PyCharm (flag --pycharm not set)"
fi

#######################################
# ROS 2 Jazzy (optional)
#######################################
if [[ "$INSTALL_ROS2" == true ]]; then
  log "Setting up ROS 2 Jazzy apt repository"
  sudo apt install -y software-properties-common
  sudo add-apt-repository -y universe || true

  sudo apt update
  sudo apt install -y curl

  # Get latest ros2-apt-source release
  ROS_APT_SOURCE_VERSION=$(
    curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest |
    grep -F "tag_name" | awk -F\" '{print $4}'
  )

  curl -L -o /tmp/ros2-apt-source.deb \
    "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.$(. /etc/os-release && echo ${UBUNTU_CODENAME:-${VERSION_CODENAME}})_all.deb"

  sudo dpkg -i /tmp/ros2-apt-source.deb || true

  log "Installing ROS 2 Jazzy desktop + dev tools"
  sudo apt update
  sudo apt upgrade -y
  sudo apt install -y \
    ros-jazzy-desktop \
    ros-dev-tools \
    python3-colcon-common-extensions

  if ! grep -q "source /opt/ros/jazzy/setup.bash" "$HOME/.bashrc"; then
    echo 'source /opt/ros/jazzy/setup.bash' >> "$HOME/.bashrc"
    echo "Added 'source /opt/ros/jazzy/setup.bash' to ~/.bashrc"
  fi
else
  log "Skipping ROS 2 Jazzy (flag --ros2 not set)"
fi

log "All done!"
echo
echo "Remember:"
echo "- Reboot or log out/log in so Docker group membership is applied."
echo "- Open a new terminal so any new ~/.bashrc lines (ROS 2) are loaded."
echo "- Test Docker with:  docker run hello-world"
if [[ "$INSTALL_ROS2" == true ]]; then
  echo "- Test ROS 2 with:   ros2 doctor"
fi

