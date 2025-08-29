.PHONY: build up down restart logs shell clean train

build:
	docker-compose build

up:
	docker-compose up -d

down:
	docker-compose down

restart:
	docker-compose restart

logs:
	docker-compose logs -f

shell:
	docker-compose exec ticket_prediction /bin/bash

clean:
	docker-compose down --volumes --remove-orphans
	docker system prune -f

dev:
	docker-compose up

stop:
	docker-compose stop

remove:
	docker-compose down --volumes --rmi all

train:
	docker-compose exec ticket_prediction python train.py

