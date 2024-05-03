# VScode dev containers isn't supported because this image is using Ubuntu 18.04
# https://code.visualstudio.com/docs/remote/faq#_can-i-run-vs-code-server-on-older-linux-distributions

# So I'm mounting the folder in the image and editing there with micro

docker run -it -d --name imgclas-devenv --mount src=/home/iheredia/ignacio/projects/deephdc/apps/ai4os-image-classification-tf,target=/srv/imgclas-mounted,type=bind -p 5000:5000 -p 6006:6006 -p 8008:8008 ai4oshub/ai4os-image-classification-tf
docker exec -i imgclas-devenv bash < setup_devenv.sh
docker exec -it imgclas-devenv bash
