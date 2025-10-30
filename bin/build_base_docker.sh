# get the file path first. Then the root folder is 1 level up
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$(dirname "$DIR")"
cd $ROOT_DIR
# SAN Dockerfile is docker/Dockerfile
docker build -t "lbarberi/san" -f docker/Dockerfile .
echo "Docker image lbarberi/san built successfully."

# grounded sam base image:
docker build -t "lbarberi/sam:base" -f docker/grounded_sam.Dockerfile .
echo "Docker image lbarberi/sam:base built successfully."

# grounded sam2 base image:
docker build -t "lbarberi/sam2:base" -f docker/grounded_sam2.Dockerfile .
echo "Docker image lbarberi/sam2:base built successfully."

