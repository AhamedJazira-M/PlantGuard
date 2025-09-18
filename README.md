# 🌱 PlantGuard: Mobile AI System for Explainable Multi-Disease Crop Detection

# Abstract 

Agricultural productivity is significantly threatened by crop diseases that reduce yield and directly impact food security. Existing solutions for plant disease detection often face practical limitations, such as dependence on cloud-based computation requiring stable internet, restriction to single-disease classification, and the absence of interpretability in predictions, which makes them less trustworthy for farmers. Moreover, most tools provide generic results in a single language, overlooking accessibility for diverse farming communities. These limitations restrict the adoption and effectiveness of current systems, particularly in rural regions where reliability, transparency, and usability are critical. To address these challenges, this work introduces PlantGuard, Mobile AI system for explainable multi-disease crop detection designed to provide accurate, explainable, and offline crop disease detection.

The objectives of PlantGuard are threefold: to detect multiple crop diseases from a single leaf image using lightweight deep learning models, to generate interpretable outputs through heatmap visualizations that highlight infected regions, and to deliver multilingual, farmer-friendly explanations and treatment recommendations. A multi-label classification pipeline, trained on publicly available datasets such as PlantVillage and Plant Pathology, enables robust detection of co-infections. Additionally, synthetic progression modeling is applied to estimate how infections may evolve over time, supporting preventive decision-making in disease management. By overcoming the limitations of internet dependency, black-box predictions, and language exclusivity, PlantGuard improves both the practicality and adoption potential of AI-based crop disease diagnosis.

The system is implemented using TensorFlow/Keras for model training, Grad-CAM for explainability, TensorFlow Lite for mobile optimization, and Flutter for cross-platform app development. Offline inference ensures reliability in regions with limited connectivity, while multilingual natural language processing enhances accessibility across diverse communities. In conclusion, PlantGuard delivers accurate, interpretable, and accessible disease detection in a lightweight mobile application. By addressing key shortcomings of existing systems, it empowers farmers with reliable decision-support tools that improve yield management and promote sustainable agriculture.

# 🚀 Innovations & New Features

- 📶 Works Fully Offline
Unlike many cloud-dependent systems, PlantGuard runs entirely on-device using TensorFlow Lite, making it reliable in rural areas with limited or no internet.

- 🩺 Multi-Disease Detection
Supports multi-label classification, meaning it can identify more than one disease from a single crop leaf image — a step beyond single-disease models.

- 🧠 Explainable AI (XAI)
Provides Grad-CAM visualizations that highlight the infected regions of the leaf, helping farmers and researchers understand why the AI made its decision.

- 🌍 Multilingual Support
Delivers explanations, results, and treatment suggestions in multiple languages, breaking language barriers for diverse farming communities.

- 📈 Disease Progression Forecasting
Uses synthetic progression modeling to estimate how infections may evolve, helping farmers take preventive measures before critical yield losses occur.

- 📱 Mobile-Optimized AI
Deploys lightweight CNN models directly into a Flutter mobile app, ensuring smooth performance across devices.

- 👨‍🌾 Farmer-Centric Design
Focused on accessibility and usability with a clean interface, simple explanations, and practical guidance for real-world agricultural practices.
