First of all, install the following dependencies: 

   1.OpenCV
   
   2.OpenCV-contrib
   
   3.SQLite
   
   4.numpy
   
   5.pillow
   
   


To install these dependencies, you can use the commands below:

    pip3 install opencv-python

    pip3 install opencv-contrib-python

    pip3 install Pillow
    
    
  In this part of the tutorial, we are going to focus on how to write the necessary code implementation for recording and training the face recognition program. We can further divide this part into:

    Create database for face recognition
    Record faces
    Train Recognizer

Create Database for face recognition

We are going to first create a database which stores the name of the corresponding faces. We will be using SQLite 3 for this purpose. Make a file named create_database.py in the working directory and copy paste the code below:
   
    import sqlite3
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    sql = """
    DROP TABLE IF EXISTS users;
    CREATE TABLE users (
               id integer unique primary key autoincrement,
               name text
    );
    """
    c.executescript(sql)
    conn.commit()
    conn.close()
    
    Record Faces

Now, we are going to prepare the dataset for face recognition. We will be using haarcascade_frontalface_default.xml file provided in the opencv/data/haarcascades directory of the opencv repo in github. Download the file and place it in the working directory. After that, make a file named record_face.py in the working directory and copy paste the code below:

    import cv2
    import numpy as np 
    import sqlite3
    import os
    conn = sqlite3.connect('database.db')
    if not os.path.exists('./dataset'):
        os.makedirs('./dataset')
    c = conn.cursor()
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    uname = input("Enter your name: ")
    c.execute('INSERT INTO users (name) VALUES (?)', (uname,))
    uid = c.lastrowid
    sampleNum = 0
    while True:
      ret, img = cap.read()
      gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      faces = face_cascade.detectMultiScale(gray, 1.3, 5)
      for (x,y,w,h) in faces:
        sampleNum = sampleNum+1
        cv2.imwrite("dataset/User."+str(uid)+"."+str(sampleNum)+".jpg",gray[y:y+h,x:x+w])
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
        cv2.waitKey(100)
      cv2.imshow('img',img)
      cv2.waitKey(1);
      if sampleNum > 20:
        break
    cap.release()
    conn.commit()
    conn.close()
    cv2.destroyAllWindows()

The code above, when run, will ask you to enter the name for the face first. It will then use the haarcascade to find the face in the camera stream. It will look for 20 samples each at the interval of 100ms. Once 20 sample faces have been found, it stores the sample data in the ‘dataset’ directory inside the working directory. In the next step, we are going to train recognizer for face recognition.

Train Recognizer

OpenCV provides three methods of face recognition:

    Eigenfaces
    Fisherfaces
    Local Binary Patterns Histograms (LBPH)

All three methods perform the recognition by comparing the face to be recognized with some training set of known faces. In the training set, we supply the algorithm faces and tell it to which person they belong.

Eigenfaces and Fisherfaces find a mathematical description of the most dominant features of the training set as a whole. LBPH analyzes each face in the training set separately and independently. The LBPH method is somewhat simpler, in the sense that we characterize each image in the dataset locally; and when a new unknown image is provided, we perform the same analysis on it and compare the result to each of the images in the dataset. We will be using the LBPH Face recognizer for our purpose. To do so, create a file named trainer.py in the working directory and copy paste the code below:

    import os
    import cv2
    import numpy as np 
    from PIL import Image
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    path = 'dataset'
    if not os.path.exists('./recognizer'):
        os.makedirs('./recognizer')
    def getImagesWithID(path):
      imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
      faces = []
      IDs = []
      for imagePath in imagePaths:
        faceImg = Image.open(imagePath).convert('L')
        faceNp = np.array(faceImg,'uint8')
        ID = int(os.path.split(imagePath)[-1].split('.')[1])
        faces.append(faceNp)
        IDs.append(ID)
        cv2.imshow("training",faceNp)
        cv2.waitKey(10)
      return np.array(IDs), faces
    Ids, faces = getImagesWithID(path)
    recognizer.train(faces,Ids)
    recognizer.save('recognizer/trainingData.yml')
    cv2.destroyAllWindows()

Run it using the command python3 trainer.pyThis will create a file named trainingData.yml inside the ‘recognizer’ directory inside the working directory.

This brings us to end of this part of  tutorial series, face recognition using OpenCV.  In this part of the series, we created 3 files:

    create_database.py: To create database and table
    record_face.py: To capture face images and record the corresponding name in the database.
    trainer.py: Use of OpenCV’s LBPH Face Recognizer to train the dataset that outputs trainingData.yml file that we’ll be using later in the tutorial for face recognition.

If you are following along with the tutorial series face recognition using OpenCV, by the end of previous part  of the series, you should have created three files:

    create_database.py: To create database and table
    record_face.py: To capture face images and record the corresponding name in the database.
    trainer.py: Use of OpenCV’s LBPH Face Recognizer to train the dataset that outputs trainingData.yml file that we’ll be using for face recognition.

You should already have trainingData.yml file inside the ‘recognizer’ directory in the working directory. If you don’t you might want to recheck the previous Part of the tutorial series. You might also remember from Previous Part of the series that we used LBPH Face recognizer to train our data. If you are curious about how LBPH works, you can refer to this article here.
Face Recognition and fetch data from SQLite

Now, we will be using the file we prepared during training to recognize whose face is it in front of the camera. We already have our virtual environment activated and the necessary dependencies installed. So, let’s get right to it. Make a file named detector.py in the working directory and copy paste the code below:

    import cv2
    import numpy as np 
    import sqlite3
    import os
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    fname = "recognizer/trainingData.yml"
    if not os.path.isfile(fname):
      print("Please train the data first")
      exit(0)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(fname)
    while True:
      ret, img = cap.read()
      gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      faces = face_cascade.detectMultiScale(gray, 1.3, 5)
      for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
        ids,conf = recognizer.predict(gray[y:y+h,x:x+w])
        c.execute("select name from users where id = (?);", (ids,))
        result = c.fetchall()
        name = result[0][0]
        if conf < 50:
          cv2.putText(img, name, (x+2,y+h-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (150,255,0),2)
        else:
          cv2.putText(img, 'No Match', (x+2,y+h-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)
      cv2.imshow('Face Recognizer',img)
      k = cv2.waitKey(30) & 0xff
      if k == 27:
        break
    cap.release()
    cv2.destroyAllWindows()
    
 Now, run this program and your face recognition app is now ready.
