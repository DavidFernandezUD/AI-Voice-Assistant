# AI VOICE ASSISTANT

AI VOICE ASSISTANT is a simple, RNN based, voice assistant consisting on a wake-word recognition model,
that continuously lsitens for a specific keyword, and a speech recognition model that classifies
the received audio into an action, from a set of actions.

The projec is also part of the "SocialTech" challenge, which consists in the making of an autonomous 
wheelchair for museum traversal. The purpose of this voice assistant is to allow users with low upper
body movility to choose between different prefabricated routes around the museum, via voice commands. 

## Project Overview

The starting point of the project is the wake-word detection model. This isin essence a binary classification
problem, so the model of choice will be a LSTM with a single sigmoid output for binary classification. The
model should be simple enough tu run continuously on an embedded system like a raspberry pi, without consuming
too much resources, but should also be robust enough so that other words aren't missinterpreted as the keyword.

Both the wake-word model and the command classification model will need a lsitener to continuously feed audio
to them. The listener will make use of a buffer to store and feed the audio signal to the model, and should be
able to work with different hardware.

Finally the command classification model will also consist on a LSTM, but with a softmax layer as output, since
it will be doing multiclass classification in thiscase. This model would only work after the previous model
has detected the keyword, and it should stop listening after a command has been detected with enough confidence,
or a certain ammount of time has passed without any command detected.

Finally the whole assistant will be introduced as a ROS node to comunicate with the wheelchair.

TODO:
- [ ] wake word model
- [ ] audio listener
- [ ] command recognition model
- [ ] AI assistant ROS-noetic node

## Repository Structure

- `src/`: Source of the project.
- `models/`: Trained deep learning models for wake-up and listener modules.

## Running on Local Machine

### Dependencies

- python3
- virtualenv (recommended)

**NOTICE: This might not work on windows, try to run it with docker or WSL instead!**

1. Set Up Virtual Enviroment
    1. `virtualenv venv`
    2. `source venv/bin/activate;

2. Install pip Packages
`pip install -r requirements.txt`

3. Training Model on Custom data
    1. `mkdir data`
    2. `cd data`
    3. Put dataset in data dir and modify load_data.py script for dataset

