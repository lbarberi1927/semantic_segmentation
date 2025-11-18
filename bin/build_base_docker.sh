# get the file path first. Then the root folder is 1 level up
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$(dirname "$DIR")"
cd $ROOT_DIR
# SAN server image:
docker build -t "lbarberi/san" -f docker/san.Dockerfile .
echo "Docker image lbarberi/san built successfully."

# grounded sam base image:
docker build -t "lbarberi/sam:base" -f docker/grounded_sam.Dockerfile .
echo "Docker image lbarberi/sam:base built successfully."

# grounded sam2 base image:
docker build -t "lbarberi/sam2:base" -f docker/grounded_sam2.Dockerfile .
echo "Docker image lbarberi/sam2:base built successfully."

# rayfronts server image:
docker build -t "lbarberi/rayfronts" -f docker/rayfronts.Dockerfile .
echo "Docker image lbarberi/rayfronts built successfully."

# OpenWorldSAM server image:
docker build -t "lbarberi/owsam" -f docker/owsam.Dockerfile .
echo "Docker image lbarberi/owsam built successfully."
