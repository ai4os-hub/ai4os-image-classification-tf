# VScode dev containers isn't supported because this image is using Ubuntu 18.04
# https://code.visualstudio.com/docs/remote/faq#_can-i-run-vs-code-server-on-older-linux-distributions

# So I'm mounting the folder in the image and editing with micro there with micro

cd /srv/

# Uninstall default repo
mv ai4os-image-classification-tf old-repo

# Install local repo
# https://stackoverflow.com/questions/58854822/python-setup-py-sdist-error-versioning-for-this-project-requires-either-an-sdist
cd imgclas-mounted
PBR_VERSION=1.2.3 python setup.py sdist
pip install -e .

# Install micro
apt-get install micro
