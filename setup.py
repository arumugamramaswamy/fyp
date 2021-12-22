#!/usr/bin/env python3

from setuptools import setup

setup(name='fyp_experiments',
      version='0.2.0',
      description='fyp experiments',
      author='Arumugam Ramaswamy',
      author_email='rm.arumugam.2000@gmail.com',
      packages=['experiments'],
      scripts=['scripts/run_experiment'],
      install_requires=[
          "stable-baselines3==1.2.0",
          "gym==0.21.0",
          "pettingzoo[mpe]==1.13.1",
          "supersuit==3.3.0",
          "torch"
      ]
     )
