# 🗳️ Face Recognition Voting System

This is a Python-based voting system that uses **facial recognition** for voter identification and authentication. Users can **register their face**, and then **vote for a party** using their facial identity. Each vote is recorded in a `.csv` file for transparency and record-keeping.

---

## 🎯 Features

- 🔐 Voter authentication using **face recognition**
- 📝 Register new faces to the system
- 🗳️ Cast a vote for a political party
- 📁 Votes recorded in a CSV file (`votes.csv`)
- ❌ Prevents multiple votes from the same face
- 📷 Real-time webcam interface for face detection

---

## 🛠️ Tech Stack

- Python 3.x
- OpenCV
- face_recognition (dlib-based)
- NumPy
- Pandas (for vote recording)

---

---

## 🚀 Getting Started

### 1. Clone the Repository
     ```bash
     git clone https://github.com/yourusername/face-voting-system.git
     cd face-voting-system

  2. Create a Virtual Environment
     ```bash
        python -m venv env
       source env/bin/activate  # On Windows: env\Scripts\activate
  3. Install Dependencies
     ```bash
      pip install -r requirements.txt
   4. Register a New Face
   Run the following script and follow the on-screen instructions:

     ```bash
    python register.py
It will capture your image using the webcam and store it in the dataset/ folder.

 5. Cast a Vote
Run:

    ```bash
      python recognizer.py
The system will detect your face.

If you’re a registered user and haven’t voted yet, you can select your party.

Your vote will be saved in votes.csv.

📄 Vote Record Format (votes.csv)
Voter Name	Party Voted	Timestamp
John Doe	Party A	2025-07-16 14:32:10

⚠️ Important Notes
Ensure your webcam is connected and functional.

Voting is one-time per registered face (prevents multiple votes).

All face data is stored locally and not uploaded anywhere.

📸 Dependencies
Make sure these are installed via requirements.txt:

opencv-python

face_recognition

numpy

pandas

🛡️ Security & Ethics
This system is for educational and experimental purposes only.

Real-world use of face recognition in elections must comply with privacy laws and ethical guidelines.

👨‍💻 Developed By
Tashik middha

Python project using face recognition for secure and smart voting experiences. Contributions welcome!

📄 License
MIT License 


