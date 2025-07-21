# 🦷 Teeth Disease Classifier

This project is a deep learning application for classifying 7 different teeth-related diseases using a fine-tuned **Xception** model. The model was trained on a custom dataset with data augmentation and fine-tuning for improved accuracy.

---

🚀 **Live Demo**  
👉 [Try the Streamlit App Here](https://dentaldiagnosisapp-caodgnw4mmk47momoy9n9w.streamlit.app/)

---

📂 **Project Structure**
- `app.py` — Streamlit app for image upload and prediction  
- `Xception_FineTuned_Model/` — The trained model in TensorFlow Xception_FineTuned_Model
- `requirements.txt` — Dependencies to run the project  

---

🧠 **Model Details**
- **Base model:** Xception (pretrained on ImageNet)  
- **Fine-tuning:** Top layers unfrozen  
- **Input size:** 224x224 RGB images  
- **Number of classes:** 7  
- **Classes:** CaS, CoS, Gum, MC, OC, OLP, OT  
- **Framework:** TensorFlow 2.10.1
