##
# ECE408
#
# @file
# @version 0.1

host:
	bash run.sh

docker:
	@echo "DIR=`pwd`" > .env # Setting environment variables for docker-compose.yaml
	@echo "USER_NAME=`id -u ${USER}`" >> .env
	@echo "GROUP=`id -g ${USER}`" >> .env
	@docker compose run cuda

docker-image: Dockerfile
	@docker compose build

# end
