BLACK_CONFIG=-t py36 -l 100
BLACK_TARGETS=src
ISORT_CONFIG=--atomic -l 100 --trailing-comma --remove-redundant-aliases --multi-line 3
ISORT_TARGETS=src

.PHONY: help install update clean format

help:
	@echo "The following make targets are available:"
	@echo "	install		  install all dependencies for environment with conda"
	@echo "	install-gpu	  install all dependencies for environment-gpu with conda"
	@echo "	update		  update all dependencies for environment with conda"
	@echo "	update-gpu	  update all dependencies for environment-gpu with conda"
	@echo " clean 		  clean previously built files"
	@echo "	format		  run black and isort on files"

install:
	conda env create -f environment.yml

install-gpu:
	conda env create -f environment_gpu.yml

update:
	conda env update -f environment.yml --prune

update-gpu:
	conda env update -f environment_gpu.yml --prune

clean:
	rm -rf dist *.egg-info build

format:
	black $(BLACK_CONFIG) $(BLACK_TARGETS)
	isort $(ISORT_CONFIG) $(ISORT_TARGETS)
