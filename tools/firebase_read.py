# A script to download images from firebase cloud storage
import pyrebase
import os
    
def fb_pull():
    filelist = [f for f in os.listdir("./firebase_pull/") if f.endswith(".jpg") ]
    for f in filelist:
        os.remove(os.path.join("./firebase_pull/", f)) 
        
    config = {
      "apiKey": "AIzaSyALSUia2pj6SSZmwhnJHuN-18DGGcQEy6I",
      "authDomain": "thesis-781ac.firebaseapp.com",
      "projectId": "thesis-781ac",
      "storageBucket": "thesis-781ac.appspot.com",
      "serviceAccount": "./tools/serviceaccountkey.json",
      "databaseURL": ""
    }

    firebase_storage = pyrebase.initialize_app(config)
    storage = firebase_storage.storage()

    #storage.child("Guitar.JPG").put("Guitar.JPG")

    #storage.child("Guitar.JPG").download("Guitar.JPG")

    all_files = storage.list_files()
    
    print("Pulling images from firebase to ./firebase_pull/")
    for file in all_files:
        file.download_to_filename('./firebase_pull/'+file.name)
    print("Completed!")