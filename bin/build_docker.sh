# get the file path first. Then the root folder is 1 level up
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$(dirname "$DIR")"
cd $ROOT_DIR
# Dockerfile is docker/Dockerfile
docker build -t "san" -f docker/Dockerfile .
