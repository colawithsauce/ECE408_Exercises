##
# ECE408
#
# @file
# @version 0.1

host:
	@echo "Making project via host ..."
	@bash run.sh

docker:
	@echo "Making project via docker ..."
	@echo "DIR=`pwd`" > .env # Setting environment variables for docker-compose.yaml
	@echo "USER_NAME=`id -u ${USER}`" >> .env
	@echo "GROUP=`id -g ${USER}`" >> .env
	@docker compose run cuda

docker-image:
	@echo "Building docker image ..."
	@docker compose build

# end
