import torchaudio
import csv
import os


if __name__ == "__main__":

    with open("data/cv-corpus-16.1/test.tsv") as file:

        reader = csv.DictReader(file, delimiter="\t")
        
        i = 0
        for row in reader:
            
            path = os.path.join("data/cv-corpus-16.1/clips/", row["path"])
            wf, sr = torchaudio.load(path)

            torchaudio.save(f"data/wakeword_dataset/test/0/{i}.wav", wf, sr)

            i += 1
            if i >= 15000:
                break
