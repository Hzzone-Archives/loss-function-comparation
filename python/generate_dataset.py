import os
import about_lmdb
import pickle

def generate_txt(source, target):
    with open(target, "w") as f:
        for person_index, ims_path in enumerate(os.listdir(source)):
            path = os.path.join(source, ims_path)
            for im in os.listdir(path):
                # line = "%s %s\n" % (os.path.join(ims_path, im), person_index)
                line = "%s %s\n" % ("/"+ims_path +"/" + im, person_index)
                f.write(line)
                print(line)

def generate_predict_sequence(source, save_path):
    _same, _diff = about_lmdb.generate_siamese_dataset(source, totals=200000)
    samples = []
    for t in _same:
        t.append(1)
        samples.append(t)
    for t in _diff:
        t.append(0)
        samples.append(t)
    f = file(save_path, "wb")
    pickle.dump(samples, f, True)


if __name__=="__main__":
    generate_txt("/home/bw/loss-function-comparation/CASIA-WebFace", "/home/bw/loss-function-comparation/train.txt")
