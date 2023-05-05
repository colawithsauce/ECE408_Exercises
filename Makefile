##
# ECE408
#
# @file
# @version 0.1

all:
	@echo "DIR=`pwd`" |tee .env
	@docker-compose up

# end
