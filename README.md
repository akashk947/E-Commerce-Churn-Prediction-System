# 🛒 E-Commerce Customer Churn Prediction App

A machine learning web application that predicts whether a customer is likely to **churn** (stop using the service) based on their behavior and profile.

Built using:
- ✅ Python
- ✅ scikit-learn & XGBoost (for modeling)
- ✅ Gradio (for UI)
- ✅ Hugging Face Spaces (for deployment)
- ✅ GitHub (for version control)

---

## 🚀 Live Demo

👉 [Try the App on Hugging Face Spaces](https://huggingface.co/spaces/akashk947/eCommerceChurnPredictionSystem)  

---

## 🧠 Model Overview

This app uses a trained **Random Forest** or **XGBoost** model to predict customer churn based on features like:

- Preferred login device
- Payment mode
- Gender, marital status
- App usage (hours spent, frequency of orders)
- Cashback received
- Complaints & customer satisfaction
- And more...


---


## 🌐 Deploy to Hugging Face Spaces

To deploy this app on Hugging Face Spaces:

1. Go to [Hugging Face Spaces](https://huggingface.co/spaces)
2. Click **Create New Space**
3. Set:

   * SDK: `Gradio`
   * Visibility: `Public` or `Private`
   * Repo source: Select **"Use existing GitHub repository"**
4. Connect your GitHub and choose the repo
5. Deploy — that's it!

---

## 📋 Sample Inputs

| Feature              | Example Value |
| -------------------- | ------------- |
| PreferredLoginDevice | Mobile Phone  |
| Gender               | Female        |
| CityTier             | 2             |
| WarehouseToHome      | 12.5          |
| SatisfactionScore    | 4             |
| OrderCount           | 10            |
| CashbackAmount       | 450.75        |
| Tenure               | 18            |
| PreferredPaymentMode | Credit Card   |

---

## 📦 Requirements

Install dependencies using:

```txt
gradio
pandas
scikit-learn
xgboost
```

---

## 🙏 Acknowledgments

* Hugging Face for free app hosting
* Gradio for UI
* scikit-learn & XGBoost for ML modeling
```
