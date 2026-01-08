To run the **AutoJudge** project locally using GitHub Codespaces, please follow these formalized steps. This guide ensures the environment is correctly configured and the Flask application is deployed successfully.

---

### **1. Accessing the Development Environment**

* **Navigate to the Repository:** Open the main page of your project repository on GitHub.
* **Launch Codespace:** Click the green **Code** button located at the top right of the file list.
* **Select Environment:** Switch to the **Codespaces** tab and select the existing configuration named **laughing-carnival** to initialize your virtual environment.

### **2. Environment Setup and Installation**

Once the terminal in your Codespace is ready, you must ensure all Python dependencies are installed:

* **Install Dependencies:** In the terminal, execute the following command:
`pip install -r requirements.txt`
*(If you do not have a requirements file, ensure you have installed `flask`, `pandas`, `scikit-learn`, and `joblib` individually.)*

### **3. Executing the Application**

* **Run the Server:** Start the Flask application by running the main execution file:
`python app.py`
* **Monitor Output:** Ensure the terminal indicates the server is active. It will typically state that the app is "Running on [http://0.0.0.0:5001](https://www.google.com/search?q=http://0.0.0.0:5001)".

### **4. Accessing the Web Interface**

* **Local Address:** Once the script is running, you can access the UI by navigating to:
**[http://127.0.0.1:5001](http://127.0.0.1:5001)**
* **Codespace Redirection:** GitHub Codespaces will often provide a pop-up notification in the bottom right corner with a button labeled **"Open in Browser."** Clicking this will automatically handle the port forwarding and take you to the live website.

> **Note:** Ensure that your model files (`tfidf_vectorizer.pkl`, `clf_model.pkl`, and `reg_model.pkl`) are present in the directory before running `app.py`, as the application requires them to generate predictions.
