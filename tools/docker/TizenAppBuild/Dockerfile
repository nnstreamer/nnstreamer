FROM ubuntu:18.04

ARG DEBIAN_FRONTEND=noninteractive
ENV LC_ALL=C.UTF-8

RUN \
apt-get update; \
apt-get install -yqq gnupg apt-transport-https wget curl apt-utils sudo git zip unzip

RUN \
groupadd --gid 1000 sdk; \
useradd --uid 1000 --gid sdk --shell /bin/bash --create-home sdk; \
echo "sdk ALL=NOPASSWD: ALL" >> /etc/sudoers

USER sdk
ENV HOME="/home/sdk/"

# TS prerequisite
RUN sudo apt-get install -yqq pciutils openjdk-8-jdk
# MOBILE-6.0 prerequisite
RUN sudo apt-get install -yqq libcurl3-gnutls libxcb-xfixes0 libxcb-render-util0 python2.7 libwebkitgtk-1.0-0 libxcb-image0 acl libsdl1.2debian libv4l-0 libxcb-randr0 libpython2.7 libxcb-icccm4 gettext rpm2cpio cpio make bridge-utils openvpn

# install TS webcli
RUN \
curl -fsSL "http://download.tizen.org/sdk/Installer/tizen-studio_4.1/web-cli_Tizen_Studio_4.1_ubuntu-64.bin" -o /tmp/cli.bin; \
chmod +x /tmp/cli.bin; \
/tmp/cli.bin --accept-license ~/tizen-studio; \
~/tizen-studio/package-manager/package-manager-cli.bin install --accept-license MOBILE-6.0;

# Install dotnet
RUN \
set -xe; \
pwd; \
wget https://packages.microsoft.com/config/ubuntu/18.04/packages-microsoft-prod.deb -O /tmp/packages-microsoft-prod.deb; \
sudo dpkg -i /tmp/packages-microsoft-prod.deb; \
sudo apt-get update; \
sudo apt-get install -y apt-transport-https && \
sudo apt-get update && \
sudo apt-get install -y dotnet-sdk-5.0

# Install code-server
RUN \
set -xe; \
curl -fsSL "https://github.com/cdr/code-server/releases/download/v3.10.2/code-server_3.10.2_amd64.deb" -o /tmp/code-server.deb; \
sudo dpkg -i /tmp/code-server.deb; \
rm -rf /tmp/code-server.deb

# Install Tizen extension
# Using tizen-csharp instead of vscodeext/vscode-tizen-web-csharp-1.3.0.vsix refer to https://marketplace.visualstudio.com/items?itemName=tizen.vscode-tizen-csharp&utm_source=www.vsixhub.com
RUN \
set -xe; \
curl -fsSL "https://f2.vsixhub.com/file.php?version=1.7.0&ext_name=vscode-tizen-csharp&post_id=3443&app_id=72bd7627-b87d-4aaf-bf66-2b269501e04f&token=fab88c9cfaddf500db1df79ea54175161704870824" -o /tmp/tizen.vsix; \
code-server --install-extension /tmp/tizen.vsix --extensions-dir $HOME/config/extensions/; \
rm -rf /tmp//tmp/tizen.vsix;

# code-server setting file
COPY --chown=sdk:sdk settings.json $HOME/config/data/User/settings.json

# code-server docker entrypoint
RUN \
set -xe; \
echo '#!/usr/bin/env bash' | sudo tee $HOME/entrypoint.sh; \
echo "mkdir -p $HOME/config/{extensions,data,workspace,.ssh}" | sudo tee -a $HOME/entrypoint.sh; \
echo "exec /usr/bin/code-server --bind-addr 0.0.0.0:8443 --user-data-dir $HOME/config/data --extensions-dir $HOME/config/extensions --disable-telemetry --auth none $HOME/config/workspace" | sudo tee -a $HOME/entrypoint.sh; \
echo 'exec "$@"' | sudo tee -a $HOME/entrypoint.sh; \
sudo chmod +x $HOME/entrypoint.sh;

WORKDIR /home/sdk/
ENTRYPOINT ["/bin/bash", "/home/sdk/entrypoint.sh"]
EXPOSE 8443
