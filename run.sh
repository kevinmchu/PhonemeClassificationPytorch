#!/usr/bin/env bash

cd feature_extraction
matlab -nodisplay -nosplash -nodesktop -r "run('extractFeaturesTestingData.m');exit;"
