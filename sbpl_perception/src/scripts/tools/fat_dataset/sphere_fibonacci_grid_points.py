#! /usr/bin/env python
#
import math
import numpy as np
import random

def sphere_fibonacci_grid_points(ng):

  rnd = 1.
  samples  = ng
  randomize = False
  if randomize:
      rnd = random.random() * samples

  points = []
  offset = 2./samples
  increment = math.pi * (3. - math.sqrt(5.));

  for i in range(samples):
      y = ((i * offset) - 1) + (offset / 2);
      r = math.sqrt(1 - pow(y,2))

      phi = ((i + rnd) % samples) * increment

      x = math.cos(phi) * r
      z = math.sin(phi) * r

      points.append([x,y,z])

  return np.array(points)  

def sphere_fibonacci_grid_points_with_sym_metric (ng, half_whole):

  if (half_whole == 1):
    rnd = 1.
    samples  = ng
    randomize = False
    if randomize:
        rnd = random.random() * samples

    points = []
    offset = 2./samples
    increment = math.pi * (3. - math.sqrt(5.));

    for i in range(samples):
        y = ((i * offset) - 1) + (offset / 2);
        r = math.sqrt(1 - pow(y,2))

        phi = ((i + rnd) % samples) * increment

        x = math.cos(phi) * r
        z = math.sin(phi) * r

        points.append([x,y,z])

    return np.array(points)  
  else: # half_whole == 0
    rnd = 1.
    samples  = ng
    randomize = False
    if randomize:
        rnd = random.random() * samples

    points = []
    offset = 2./samples
    # offset = 1./samples
    increment = math.pi * (3. - math.sqrt(5.));

    # for i in range(math.ceil(samples/2)):
    for i in range(round(samples/2)):
        y = ((i * offset) - 1) + (offset / 2);
        r = math.sqrt(1 - pow(y,2))

        phi = ((i + rnd) % samples) * increment

        x = math.cos(phi) * r
        z = math.sin(phi) * r

        points.append([x,y,z])

    return np.array(points)
