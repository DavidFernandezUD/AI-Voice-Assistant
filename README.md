# AI VOICE ASSISTANT

![potato](img/potato_voice_assistant.jpg)

AI VOICE ASSISTANT is a simple, RNN based, voice assistant consisting on a wake-word recognition model,
that continuously lsitens for a specific keyword, and a speech recognition model that classifies
the received audio into an action, from a set of actions.

The projec is also part of the "SocialTech" challenge, which consists in the making of an autonomous 
wheelchair for museum traversal. The purpose of this voice assistant is to allow users with low upper
body movility to choose between different prefabricated routes around the museum, via voice commands. 

## Project Overview

The starting point of the project is the wake-word detection model. This is in essence a binary classification
problem, so the model of choice will be a LSTM with a single sigmoid output for binary classification. The
model should be simple enough tu run continuously on an embedded system like a raspberry pi, without consuming
too much resources, but should also be robust enough so that other words aren't missinterpreted as the keyword.

Both the wake-word model and the command classification model will need a lsitener to continuously feed audio
to them. The listener will make use of a buffer to store and feed the audio signal to the model, and should be
able to work with different hardware.

Finally the command classification model will also consist on a LSTM, but with a softmax layer as output, since
it will be doing multiclass classification in this case. This model would only work after the previous model
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
    2. `source venv/bin/activate`

2. Install pip Packages
    1. `pip install -r requirements.txt`

3. Training Model on Custom data
    1. `mkdir data`
    2. `cd data`
    3. `mv /path/to/dataset ./data`
    4. modify data loading script

## Datasets

- [Common Voice Corpus 16.1](https://commonvoice.mozilla.org/en/datasets)
- [CHIME-Home](https://archive.org/details/chime-home)
- [Speech Commands](https://arxiv.org/pdf/1804.03209.pdf)
- [AudioMNIST](https://github.com/soerenab/AudioMNIST?tab=readme-ov-file)

## Resources

- [Audio Deep Learning (part1)](https://towardsdatascience.com/audio-deep-learning-made-simple-sound-classification-step-by-step-cebc936bbe5)
- [Audio Deep Learning (part2)](https://towardsdatascience.com/audio-deep-learning-made-simple-part-2-why-mel-spectrograms-perform-better-aad889a93505)
- [Audio Deep Learning (part3)](https://towardsdatascience.com/audio-deep-learning-made-simple-part-3-data-preparation-and-augmentation-24c6e1f6b52)
- [Guide to Audio Classification Using Deep Learning](https://www.analyticsvidhya.com/blog/2022/04/guide-to-audio-classification-using-deep-learning/)
- [Speech Commands](https://arxiv.org/pdf/1804.03209.pdf)
- [AudioMNIST](https://arxiv.org/pdf/1807.03418.pdf)
- [Audio Classification](https://medium.com/@cgawande12/audio-classification-with-the-audio-mnist-dataset-0ad95c3fb713)
- [CNN for Audio Classification](https://www.mdpi.com/2076-3417/11/13/5796)
- [CNN vs RNN for Music Genre Classification](https://www.diva-portal.org/smash/get/diva2:1354738/FULLTEXT01.pdf)
- [Acoustic Scene Classification Using LSTM-CNN](https://dcase.community/documents/workshop2016/proceedings/Bae-DCASE2016workshop.pdf)
- [Comparison of Audio Preprocessing Methods](https://web.archive.org/web/20220612070124id_/https://repositum.tuwien.at/bitstream/20.500.12708/20348/1/Damboeck%20Maximilian%20-%202022%20-%20A%20Comparison%20of%20Audio%20Preprocessing%20Methods%20for...pdf)
