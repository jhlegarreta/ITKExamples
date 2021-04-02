#!/usr/bin/env python

# Copyright NumFOCUS
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import itk
import argparse

if len(sys.argv) != 2:
    print("Usage: " + sys.argv[0] + "<InputFileName>")
    print("Prints image moment information from binary image.")
    sys.exit(1)

parser = argparse.ArgumentParser(description="Calculate Image Moments.")
parser.add_argument("input_image")

args = parser.parse_args()
input_image = itk.imread(args.input_image)
momentor = itk.ImageMomentsCalculator.New(Image=input_image)
momentor.Compute()
print(momentor)
