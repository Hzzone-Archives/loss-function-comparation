import os

def generate_txt(source, target):
    with open(target, "w") as f:
        for person_index, ims_path in enumerate(os.listdir(source)):
            path = os.path.join(source, ims_path)
            for im in os.listdir(path):
                # line = "%s %s\n" % (os.path.join(ims_path, im), person_index)
                line = "%s %s\n" % ("/"+ims_path +"/" + im, person_index)
                f.write(line)
                print(line)


if __name__=="__main__":
    generate_txt("/home/bw/loss-function-comparation/CASIA-WebFace", "/home/bw/loss-function-comparation/train.txt")
