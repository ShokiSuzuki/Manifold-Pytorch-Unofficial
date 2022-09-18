#! /bin/bash

docker build ./ --force-rm --no-cache -t vit/pytorch:cuda11.4-ubuntu20.04
