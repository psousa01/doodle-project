.DEFAULT_GOAL := default
#################### PACKAGE ACTIONS ###################
run_train:
	python -c 'from doodle.interface.main import train; train()'