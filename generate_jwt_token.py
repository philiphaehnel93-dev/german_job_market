import jwt
import datetime

# This must exactly match the SECRET_KEY in your FastAPI app
SECRET_KEY = os.getenv("API_SECRET_KEY")

payload = {
    "service": "streamlit_dashboard",
    # Setting an expiration date 10 years in the future for the service token
    "exp": datetime.datetime.utcnow() + datetime.timedelta(days=3650)
}

token = jwt.encode(payload, SECRET_KEY, algorithm="HS256")
print(f"Your STREAMLIT_JWT for the .env file: \n{token}")