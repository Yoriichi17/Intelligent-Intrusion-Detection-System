<h1>Intelligent-Intrusion-Detection-System</h1>

<p>An AI-powered Cyber Threat Intelligence platform that captures real-time network traffic, detects potential threats using machine learning, and visualizes insights through a secure, full-stack dashboard.</p>

<h2>Overview:</h2>
<p>This project implements an intelligent intrusion detection system designed to enhance network security. It performs real-time traffic analysis using a trained AI model to classify network packets as benign or malicious, then stores and displays results on a dashboard built with FastAPI, ReactJS, and MySQL. The dashboard includes secure user login authentication.</p>
<h2>Features:</h2>
<ul>
  <li>🔴 Real-time network packet capture</li>

  <li>🤖 AI-based threat classification (Benign or Malicious)</li>

  <li>📊 Web dashboard with live updates and threat visualization</li>

  <li>🔐 User authentication with login system</li>

  <li>💾 MySQL backend to store analysis logs and user data</li>
</ul>

<h2>🛠️ Tech Stack:</h2>
<table>
  <thead>
    <tr>
      <th>Layer</th>
      <th>Technology</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Frontend</td>
      <td>ReactJS, Custom CSS</td>
    </tr>
    <tr>
      <td>Backend API</td>
      <td>FastAPI (Python)</td>
    </tr>
    <tr>
      <td>Database</td>
      <td>MySQL</td>
    </tr>
    <tr>
      <td>ML Model</td>
      <td>Random Forest, Gradient Boost & Extreme Gradient Boosting classifiers</td>
    </tr>
    <tr>
      <td>Traffic Capture</td>
      <td>T-shark</td>
    </tr>
  </tbody>
</table>

<h2>🖥️ Dashboard Screens</h2>
<ul>
  <li>
    <strong>Login Page</strong> – Secure login for users<br>
    <img src="https://drive.google.com/uc?export=view&id=1hwu79r-2GSn7HNJ6FKfp8XdBrNZa8jFk" alt="Login Page Screenshot" width="400" style="margin-top:10px;">
  </li>
  <li>
    <strong>Threat Dashboard</strong> – Real-time list of analyzed packets with a chart and stats of threat types over time<br>
    <img src="https://drive.google.com/uc?export=view&id=1nak1jvzgprrg7W9OtLfjyD0DQRgCnk4d" alt="Threat Dashboard Screenshot" width="400" style="margin-top:10px;">
  </li>
  <li>
    <strong>User Management</strong> – (Optional) Admin-level view for managing users<br>
    <img src="https://drive.google.com/uc?export=view&id=14pru7p4K-d0xoTEzBjA-KX9Q8bMhOSl3" alt="User Management Screenshot" width="400" style="margin-top:10px;">
  </li>
</ul>

<h2>🌐 Project Structure:</h2>
Dashboard/  <br/>
├── backend/  <br/>
│   ├── convert.py  <br/>
│   ├── db_utils.py  <br/>
│   ├── extract.py  <br/>
│   ├── login.py  <br/>
│   ├── main.py  <br/>
│   ├── realtime_predict.py  <br/>
│   ├── registration.py  <br/>
│   └── requirements.txt  <br/>
│  <br/>
├── frontend/  <br/>
│   ├── public/  <br/>
│   └── src/  <br/>
│       ├── components/  <br/>
│       │   ├── AddUserModal.js  <br/>
│       │   └── Header.js  <br/>
│       ├── pages/  <br/>
│       │   ├── Dashboard.js  <br/>
│       │   ├── Login.js  <br/>
│       │   └── Register.js  <br/>
│       ├── services/  <br/>
│       │   ├── ProtectedRoute.js  <br/>
│       │   └── api.js  <br/>
│       ├── App.css  <br/>
│       ├── App.js  <br/>
│       ├── App.test.js  <br/>
│       ├── index.css  <br/>
│       ├── index.js  <br/>
│       ├── logo.svg  <br/>
│       ├── reportWebVitals.js  <br/>
│       ├── setupTests.js  <br/>
│       ├── styles.css  <br/>
│       └── utils.js  <br/>
 <br/>
├── Detection-Models/  <br/>
│   ├── data/  <br/>
│   │   └── training_data.csv  <br/>
│   └── Models/  <br/>
│       ├── GBC.py  <br/>
│       ├── RFC.py  <br/>
│       ├── XGB.py  <br/>
│       ├── captureNetworkTraffic.py  <br/>
│       └── predict.py  <br/>
│  <br/>
├── .gitignore  <br/>
└── README.md  

<h2>🌱 Set it up in your System:</h2>
<h3>Requirements:</h3>
  <ul>
    <li>Python 3.10.0</li>
    <li>NodeJS</li>
    <li>MySQL</li>
    <li>Wireshark and T-shark</li>
  </ul> 

  <h3>1. Clone the Repo:</h3>
    a. Open the terminal in the directory of your choice,</li><br/>
    b. copy and paste the below command: <br/>
    
      git clone https://github.com/ankush07261/Intelligent-Intrusion-Detection-System.git

  <h3>2. Set up the environment:</h3>
    a. Navigate inside the project directory

      cd Intelligent-Intrusion-Detection-System

  b. Navigate to backend

     cd Dashboard/backend

  c. Create a virtual env:
  
     python -m venv venv

d. Activate the venv:

    venv\Scripts\activate

e.  Install dependencies

    python install -r requirements.txt

f. now, open a terminal in the root project folder and navigate to frontend dir:

    cd Dashboard/backend

g. Install node modules in the frontend folder:

    npm i

h. Open a new terminal in the root project folder and navigate to Detection-Models:

    cd Dashboard/Detection-Models

i. Install dependecies again to run the models:

    python install -r requirements.txt

<h2>🚅 Running the software:</h2>

First, replace "<your_password>" in the method get_db() with your mysql password in the following files:
<ul>
  <li>db_utils.py</li>
  <li>convert.py</li>
  <li>login.py</li>
  <li>realtime_predict.py</li>
  <li>registration.py</li>
</ul>
Then... <br/><br/>
a. Open terminal in Detection-Models and run any of the 3 models (Train the models).<br/>
b. 4 .pkl files will be created, copy these pkl files and paste them in the backend folder (This is an important step). <br/>
c. You can capture your network traffic by running the captureNetworkTraffic.py file and test the model by running the predict.py file.<br/><br/>
Command to run a pyton file (replace "filename" with the name of the file you want to run in the below command):<br/>

    python filename.py

Open two terminals in the backend folder and run the below commands in each of the terminals (with venv activated) to capture the real time network traffic and predict the packets flowing:

    uvicorn main:app --reload
and

    python realtime_predict.py

Now, open a terminal in the frontend dir and enter the below command:

    npm start

The dashboard UI will start on automatically on your default web browser on the port 3000. If not, open a browser and browse "localhost:3000"<br/><br/>
Now you may register yourself and login to view the dashboard (unauthorized registration of a new user will be implemented soon).

<h2>🐋 Running the project using docker image:</h2>
Pull the docker image and run:

    docker pull ankus263/intelligent-intrusion-detection-system:latest
and then:

    docker run -p 5000:5000 ankus263/intelligent-intrusion-detection-system:latest

<h2>📈 Future Enhancements</h2>
<ul>
  <li>Increasing model accuracy</li>
  <li>Multi-user roles (admin/analyst)</li>
  <li>Alert system via email or Slack</li>
  <li>Dockerized deployment</li>
</ul>

<h2>🤝 Contributing</h2>
<p>
  Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to change.
</p>
