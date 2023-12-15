docker build -t alpaca .
docker build -f alpaca.dockerfile . -t alpaca-docker

docker run --rm --gpus all -v /home/alourido/Desktop/alpaca-classification:/alpaca-classification alpaca
