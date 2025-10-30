# get the file path first. Then the root folder is 1 level up
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$(dirname "$DIR")"
cd $ROOT_DIR
# dev images
docker build -t "lbarberi/sam:dev" -f docker/sam.Dockerfile.dev .
docker build -t "lbarberi/sam2:dev" -f docker/sam2.Dockerfile.dev .