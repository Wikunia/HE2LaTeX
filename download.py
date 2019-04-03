# download all files necessary for training
import requests, os, zipfile
base_url = "http://he2latex.opensourc.es/"

files = ["train_images.npy", "train_labels.npy", "test_images.npy", "test_labels.npy", "iseq_n.npy", "oseq_n.npy", "normalized.zip", "formulas.zip"]

for fname in files:
    if not os.path.exists(fname): 
        print("Downloading: ", fname)
        url = base_url + fname
        r = requests.get(url)
        open(fname , 'wb').write(r.content)
    else:
        print("%s exists already " % fname)

for fname in files:
    if fname[-4:] == ".zip" and not os.path.exists(fname[:-4]+"/"):
        print("Extracting zip file %s" % fname)
        zip_ref = zipfile.ZipFile(fname, 'r')
        zip_ref.extractall(".")
        zip_ref.close()
