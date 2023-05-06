##
# ECE408
#
# @file
# @version 0.1

all:
	@echo "DIR=`pwd`" |tee .env
	@docker compose run cuda

docker-image: ./Dockerfile
	@echo "DIR=`pwd`" |tee .env
	@docker compose build

# end
